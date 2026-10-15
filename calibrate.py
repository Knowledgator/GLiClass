"""Fit a post-hoc calibrator for a trained GLiClass model with its backbone frozen.

The frozen model is run once over the calibration data; the cached training pairs are then reweighted so that
every (raw confidence bin, correct / wrong) cell carries the same total weight, and the calibrator MLP is trained
on the cached (text repr, label repr, logits, gold) tuples by minimizing the weighted NLL of
sigmoid(beta * logit + bias) plus a prior towards a global temperature / Platt scaling fit (see
gliclass/calibration.py). Texts with exactly one gold label are calibrated as single-label: their NLL is the
cross-entropy of softmax(beta * logit + bias) over their labels, matching classification_type="single-label".
The saved model returns calibrated logits and keeps the raw ones in `uncalibrated_logits`; the pipeline reports
both scores.

Example:
    python calibrate.py --model_path models/checkpoint-1000 --data_path data/val.json
        --save_path models/checkpoint-1000-calibrated

For multi-modal decoder-kv models, data items may carry "images" / "audio" lists as in training.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import random
import argparse

import torch
from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import DataLoader

from gliclass import GLiClassModel
from gliclass.calibration import (
    fit_calibrator,
    fit_global_scaling,
    evaluate_calibrator,
    collect_calibration_examples,
)
from gliclass.data_processing import GLiClassDataset, AugmentationConfig, DataCollatorWithPadding

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def mark_single_label(examples):
    """Flag texts whose targets are one-hot (exactly one gold label among 2+), so they are calibrated with softmax.

    Returns the number of flagged texts.
    """
    count = 0
    for ex in examples:
        one_hot = ((ex.targets == 0) | (ex.targets == 1)).all() and ex.targets.sum() == 1
        ex.single_label = ex.single_label or bool(one_hot and len(ex.targets) > 1)
        count += ex.single_label
    return count


def confidence_cells(example, num_bins):
    """Cell id of every pair.

    Multi-label text: per pair, confidence bin (max(p, 1 - p) of the raw sigmoid, num_bins equal-width bins over
    [0.5, 1]) * 2 + whether the raw decision at 0.5 is correct; ids in [0, 2 * num_bins).
    Single-label text: one cell for the whole text from the raw softmax, top-1 probability binned over [0, 1] and
    whether the top-1 label is gold; ids in [2 * num_bins, 4 * num_bins).
    """
    if example.single_label:
        probs = torch.softmax(example.logits, dim=-1)
        confidence, top = probs.max(dim=-1)
        conf_bin = min(int(confidence * num_bins), num_bins - 1)
        cell = 2 * num_bins + conf_bin * 2 + int(example.targets[top] >= 0.5)
        return torch.full_like(example.logits, cell, dtype=torch.long)
    probs = torch.sigmoid(example.logits)
    confidence = torch.maximum(probs, 1 - probs)
    bins = torch.clamp(((confidence - 0.5) * 2 * num_bins).long(), max=num_bins - 1)
    correct = ((probs >= 0.5) == (example.targets >= 0.5)).long()
    return bins * 2 + correct


def cell_mass(examples, num_bins, weighted):
    """Loss mass of every cell: a pair counts weight / #labels of its text, so a text has mass 1 in total."""
    mass = torch.zeros(num_bins * 4)
    for ex in examples:
        weights = ex.weights if weighted and ex.weights is not None else torch.ones_like(ex.logits)
        mass.index_add_(0, confidence_cells(ex, num_bins), weights / len(ex.logits))
    return mass


def describe_cells(examples, num_bins, weighted):
    """Share of the loss mass in every cell, within the multi-label (sigmoid) and single-label (softmax) groups."""
    mass = cell_mass(examples, num_bins, weighted)
    lines = []
    groups = (("multi-label pairs, sigmoid", 0, 0.5), ("single-label texts, softmax top-1", 2 * num_bins, 0.0))
    for title, offset, low in groups:
        group = mass[offset : offset + 2 * num_bins]
        if group.sum() == 0:
            continue
        group = group / group.sum()
        lines.append(f" {title}:")
        for b in range(num_bins):
            lo, hi = low + b * (1 - low) / num_bins, low + (b + 1) * (1 - low) / num_bins
            lines.append(f"  conf [{lo:.2f}, {hi:.2f}): correct {group[2 * b + 1]:7.2%}  wrong {group[2 * b]:7.2%}")
    return "\n".join(lines)


def balance_examples(examples, num_bins, max_weight):
    """Set per-pair weights so that, within the multi-label and within the single-label group, every non-empty
    (confidence bin, correctness) cell has the same total loss mass, keeping each group's total mass (and so the
    loss scale and the multi- / single-label proportion) unchanged. Weights are clipped to max_weight so that a
    handful of pairs in a rare cell cannot dominate the fit.
    """
    mass = cell_mass(examples, num_bins, weighted=False)
    cell_weights = torch.zeros_like(mass)
    for offset in (0, 2 * num_bins):
        group = mass[offset : offset + 2 * num_bins]
        if group.sum() == 0:
            continue
        target = group.sum() / (group > 0).sum()
        weights = torch.where(group > 0, target / group.clamp_min(1e-12), torch.zeros_like(group))
        weights = weights.clamp(max=max_weight)
        cell_weights[offset : offset + 2 * num_bins] = weights * group.sum() / (weights * group).sum()
    for ex in examples:
        ex.weights = cell_weights[confidence_cells(ex, num_bins)]
    return examples


def build_dataset(data, model, tokenizer, args):
    config = model.config
    labels_tokenizer = None
    if config.label_model_name is not None:
        labels_tokenizer = AutoTokenizer.from_pretrained(config.label_model_name)
    return GLiClassDataset(
        data,
        tokenizer,
        AugmentationConfig(enabled=False),
        max_length=args.max_length,
        problem_type=config.problem_type,
        architecture_type=config.architecture_type,
        prompt_first=config.prompt_first,
        labels_tokenizer=labels_tokenizer,
        shuffle_labels=False,
    )


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = GLiClassModel.from_pretrained(args.model_path, torch_dtype=DTYPES[args.dtype]).to(device)
    # multi-modal decoder-kv: the processor also prepares the images / audio of each data item
    tokenizer_class = AutoProcessor if getattr(model.config, "multimodal", False) else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_path)
    model.eval()
    model.requires_grad_(False)

    with open(args.data_path) as f:
        data = json.load(f)
    random.shuffle(data)
    if args.max_examples is not None:
        data = data[: args.max_examples]

    # split by text so labels of one text never land on both sides
    num_eval = max(1, int(len(data) * args.eval_ratio))
    train_data, eval_data = data[num_eval:], data[:num_eval]
    print(f"Calibration texts: {len(train_data)} train, {len(eval_data)} held-out")

    collator = DataCollatorWithPadding(device="cpu", config=model.config)

    def collect(split):
        loader = DataLoader(
            build_dataset(split, model, tokenizer, args),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )
        return collect_calibration_examples(model, loader, device)

    train_examples = collect(train_data)
    eval_examples = collect(eval_data)

    if args.softmax_single_label:
        for name, examples in (("train", train_examples), ("held-out", eval_examples)):
            print(f"Single-label (softmax) {name} texts: {mark_single_label(examples)} / {len(examples)}")

    if args.balance:
        print("Training loss mass by raw confidence / correctness:")
        print(describe_cells(train_examples, args.balance_bins, weighted=False))
        balance_examples(train_examples, args.balance_bins, args.balance_max_weight)
        print("After balancing:")
        print(describe_cells(train_examples, args.balance_bins, weighted=True))

    base_beta, base_bias = fit_global_scaling(train_examples, fit_bias=args.use_bias)
    method = "Platt" if args.use_bias else "temperature"
    print(f"Global {method} scaling: beta_0 = {base_beta:.4f}, b_0 = {base_bias:.4f}")

    calibrator = model.add_calibrator(
        hidden_size=args.hidden_size, beta_max=args.beta_max, dropout=args.dropout, use_bias=args.use_bias
    )
    fit_calibrator(
        calibrator,
        train_examples,
        eval_examples,
        device,
        base_beta=base_beta,
        base_bias=base_bias,
        prior_weight=args.prior_weight,
        bias_prior_weight=args.bias_prior_weight,
        num_epochs=args.num_epochs,
        batch_size=args.calibrator_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )

    report = {
        "base_beta": base_beta,
        "base_bias": base_bias,
        "train": evaluate_calibrator(calibrator, train_examples, device, base_beta, base_bias),
        "held_out": evaluate_calibrator(calibrator, eval_examples, device, base_beta, base_bias),
        "args": vars(args),
    }
    print(json.dumps({key: report[key] for key in ("base_beta", "base_bias", "held_out")}, indent=2))

    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    with open(os.path.join(args.save_path, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved calibrated model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a GLiClass calibrator with a frozen backbone")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="JSON in the training data format")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="Share of texts held out for early stopping")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8, help="Backbone batch size for feature collection")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float32", choices=list(DTYPES))

    parser.add_argument(
        "--softmax_single_label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calibrate texts with exactly one gold label with softmax over their labels instead of per-label "
        "sigmoids (always on for single_label_classification models)",
    )
    parser.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reweight training pairs so every (raw confidence bin, correct / wrong) cell has equal loss mass",
    )
    parser.add_argument("--balance_bins", type=int, default=5, help="Equal-width raw confidence bins over [0.5, 1]")
    parser.add_argument(
        "--balance_max_weight", type=float, default=10.0, help="Upper bound of a pair's weight before renormalizing"
    )

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--beta_max", type=float, default=3.0, help="Upper bound of the inverse temperature")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--prior_weight",
        type=float,
        default=0.01,
        help="Weight of (log beta - log beta_0)^2; large -> temperature scaling",
    )
    parser.add_argument(
        "--use_bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also predict a per-pair logit bias (fixes base-rate miscalibration; may flip decisions at 0.5)",
    )
    parser.add_argument(
        "--bias_prior_weight", type=float, default=0.01, help="Weight of (bias - b_0)^2, b_0 from global Platt scaling"
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--calibrator_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
