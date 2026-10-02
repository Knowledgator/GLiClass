"""Fit a post-hoc calibrator for a trained GLiClass model with its backbone frozen.

The frozen model is run once over the calibration data; the calibrator MLP is then trained on the cached
(text repr, label repr, logits, gold) tuples by minimizing the binary NLL of sigmoid(beta * logit + bias) plus
a prior towards a global temperature / Platt scaling fit (see gliclass/calibration.py). The saved model returns
calibrated logits and keeps the raw ones in `uncalibrated_logits`; the pipeline reports both scores.

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
