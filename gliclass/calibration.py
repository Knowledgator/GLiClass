"""Post-hoc calibration of GLiClass scores with a frozen backbone.

The calibrator predicts an inverse temperature beta in (0, beta_max) for every (text, label) pair and,
optionally, a bias b; the calibrated probability is sigmoid(beta * logit + b). It is fitted by minimizing the
binary NLL on held-out data: for a pair the backbone gets wrong the NLL decreases as beta -> 0 (probability
-> sigmoid(b)), for a correct pair it decreases as beta grows, so the MLP learns where the backbone can be
trusted. A prior pulls (log beta, b) towards a global fit (log beta_0, b_0), which keeps the per-pair optimum
from collapsing to {0, beta_max}.

Without the bias, beta > 0 never flips the sign of a logit: decisions at 0.5 are unchanged and a distrusted
pair goes to 0.5. Scaling alone can only shrink towards 0.5, so it cannot fix a backbone that systematically
over- or under-predicts positives; the bias shifts the base rate (global Platt scaling sigmoid(a * z + c) is
then the baseline), at the cost of possibly flipping decisions and pulling distrusted pairs to sigmoid(b)
instead of 0.5.
"""

import math
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

NUM_LOGIT_FEATURES = 6


def calibrator_representation_size(config):
    """Size of the text/label representations the architecture exposes to the calibrator."""
    if config.architecture_type == "decoder-kv":
        return config.hidden_size
    return config.encoder_config.hidden_size


def logit_features(logits, label_mask):
    """Per-pair confidence features: (batch, num_labels, NUM_LOGIT_FEATURES).

    [z, |z|, z - max_valid(z), z - mean_valid(z), mean binary entropy of the text's labels, log(#labels)]
    """
    mask = label_mask.bool()
    maskf = mask.to(logits.dtype)
    num_labels = maskf.sum(dim=-1, keepdim=True).clamp_min(1.0)

    max_logit = logits.masked_fill(~mask, float("-inf")).amax(dim=-1, keepdim=True)
    max_logit = torch.where(mask.any(dim=-1, keepdim=True), max_logit, torch.zeros_like(max_logit))
    mean_logit = (logits * maskf).sum(dim=-1, keepdim=True) / num_labels

    probs = torch.sigmoid(logits)
    entropy = -(probs * F.logsigmoid(logits) + (1 - probs) * F.logsigmoid(-logits))
    mean_entropy = (entropy * maskf).sum(dim=-1, keepdim=True) / num_labels

    features = [
        logits,
        logits.abs(),
        logits - max_logit,
        logits - mean_logit,
        mean_entropy.expand_as(logits),
        num_labels.log().expand_as(logits),
    ]
    return torch.stack(features, dim=-1) * maskf.unsqueeze(-1)


class GLiClassCalibrator(nn.Module):
    """MLP over (text repr, label repr, logit statistics) -> inverse temperature beta in (0, beta_max)
    and, with use_bias, an additive logit bias b. Calibrated logit: beta * z + b.
    """

    def __init__(self, representation_size, hidden_size=256, beta_max=3.0, dropout=0.1, use_bias=False):
        super().__init__()
        self.beta_max = beta_max
        self.use_bias = use_bias
        self.text_norm = nn.LayerNorm(representation_size)
        self.label_norm = nn.LayerNorm(representation_size)
        self.mlp = nn.Sequential(
            nn.Linear(4 * representation_size + NUM_LOGIT_FEATURES, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2 if use_bias else 1),
        )
        self.reset_to_global_scaling(1.0)

    def reset_to_global_scaling(self, base_beta, base_bias=0.0):
        """Zero the output layer so (beta, b) == (base_beta, base_bias) everywhere: training starts from the
        global temperature / Platt scaling fit.
        """
        ratio = min(max(base_beta / self.beta_max, 1e-4), 1 - 1e-4)
        output = self.mlp[-1]
        with torch.no_grad():
            output.weight.zero_()
            output.bias[0] = math.log(ratio / (1 - ratio))
            if self.use_bias:
                output.bias[1] = base_bias

    def forward(self, text_repr, label_repr, logits, label_mask):
        """Predict the inverse temperature of every (text, label) pair.

        Args:
            text_repr: (batch, dim)
            label_repr: (batch, num_labels, dim)
            logits: (batch, num_labels) uncalibrated logits
            label_mask: (batch, num_labels), 1 for real labels

        Returns:
            beta: (batch, num_labels) inverse temperatures (1.0 on padded labels)
            bias: (batch, num_labels) logit biases (0.0 on padded labels and without use_bias)
        """
        dtype = self.mlp[0].weight.dtype
        num_labels = logits.shape[-1]
        label_repr = label_repr[:, :num_labels]
        label_mask = label_mask[:, :num_labels].bool()

        text = self.text_norm(text_repr.to(dtype)).unsqueeze(1).expand_as(label_repr)
        label = self.label_norm(label_repr.to(dtype))
        stats = logit_features(logits.float(), label_mask).to(dtype)
        features = torch.cat([text, label, text * label, (text - label).abs(), stats], dim=-1)

        output = self.mlp(features).float()
        beta = self.beta_max * torch.sigmoid(output[..., 0])
        bias = output[..., 1] if self.use_bias else torch.zeros_like(beta)
        return beta.masked_fill(~label_mask, 1.0), bias.masked_fill(~label_mask, 0.0)


def build_calibrator(config):
    return GLiClassCalibrator(
        calibrator_representation_size(config),
        hidden_size=config.calibrator_hidden_size,
        beta_max=config.calibrator_beta_max,
        dropout=config.calibrator_dropout,
        use_bias=getattr(config, "calibrator_use_bias", False),
    )


# ---------------------------------------------------------------------------------------------------------
# Fitting on cached backbone outputs
# ---------------------------------------------------------------------------------------------------------


@dataclass
class CalibrationExample:
    text_repr: torch.Tensor  # (dim,)
    label_repr: torch.Tensor  # (num_labels, dim)
    logits: torch.Tensor  # (num_labels,)
    targets: torch.Tensor  # (num_labels,) in [0, 1]


def _targets_from_labels(labels, num_labels, problem_type):
    if problem_type == "single_label_classification" or labels.dim() == 1:
        return F.one_hot(labels.long().view(-1), num_classes=num_labels).float()
    return labels[:, :num_labels].float()


@torch.no_grad()
def collect_calibration_examples(model, dataloader, device, show_progress=True):
    """Run the frozen model once over a dataloader and cache per-text calibration inputs on CPU.

    Uses the uncalibrated logits, so it also works for re-fitting a model that already has a calibrator.
    """
    from tqdm import tqdm

    model.eval()
    examples = []
    iterator = tqdm(dataloader, desc="Collecting calibration features") if show_progress else dataloader
    for batch in iterator:
        labels = batch["labels"]
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
            if key not in {"labels", "labels_text", "input_texts"}
        }
        outputs = model.model(
            **inputs,
            output_text_embeddings=True,
            output_class_embeddings=True,
            return_dict=True,
        )
        logits = outputs.logits.float()
        num_labels = min(logits.shape[-1], labels.shape[-1]) if labels.dim() > 1 else logits.shape[-1]
        targets = _targets_from_labels(labels, logits.shape[-1], model.config.problem_type)[:, :num_labels]
        label_mask = outputs.class_mask[:, :num_labels].bool().cpu()
        if labels.dim() > 1 and "labels_text" in batch:
            counts = torch.tensor([len(item) for item in batch["labels_text"]])
            label_mask &= torch.arange(num_labels).unsqueeze(0) < counts.unsqueeze(1)

        text_repr = outputs.text_embeddings.float().cpu()
        label_repr = outputs.class_embeddings.float().cpu()
        logits = logits.cpu()
        for i in range(logits.shape[0]):
            valid = label_mask[i]
            if not valid.any():
                continue
            examples.append(
                CalibrationExample(
                    text_repr=text_repr[i],
                    label_repr=label_repr[i, :num_labels][valid],
                    logits=logits[i, :num_labels][valid],
                    targets=targets[i][valid],
                )
            )
    return examples


def collate_calibration_examples(examples):
    label_repr = pad_sequence([ex.label_repr for ex in examples], batch_first=True)
    logits = pad_sequence([ex.logits for ex in examples], batch_first=True)
    targets = pad_sequence([ex.targets for ex in examples], batch_first=True)
    label_mask = pad_sequence([torch.ones_like(ex.logits, dtype=torch.bool) for ex in examples], batch_first=True)
    return {
        "text_repr": torch.stack([ex.text_repr for ex in examples]),
        "label_repr": label_repr,
        "logits": logits,
        "targets": targets,
        "label_mask": label_mask,
    }


def _masked_text_mean(values, label_mask):
    """Average over labels inside each text, then over texts: every text weighs the same."""
    maskf = label_mask.float()
    per_text = (values * maskf).sum(dim=-1) / maskf.sum(dim=-1).clamp_min(1.0)
    return per_text.mean()


def calibration_loss(
    beta,
    bias,
    logits,
    targets,
    label_mask,
    base_beta,
    base_bias=0.0,
    prior_weight=0.01,
    bias_prior_weight=0.01,
):
    """Binary NLL of sigmoid(beta * z + b)
    + prior_weight * (log beta - log base_beta)^2 + bias_prior_weight * (b - base_bias)^2.
    """
    nll = F.binary_cross_entropy_with_logits(beta * logits + bias, targets, reduction="none")
    prior = prior_weight * (beta.clamp_min(1e-6).log() - math.log(base_beta)) ** 2
    prior = prior + bias_prior_weight * (bias - base_bias) ** 2
    nll = _masked_text_mean(nll, label_mask)
    prior = _masked_text_mean(prior, label_mask)
    return nll + prior, nll


def fit_global_scaling(examples, fit_bias=False, max_iter=100):
    """The single (beta_0, b_0) minimizing the binary NLL of sigmoid(beta_0 * z + b_0).

    fit_bias=False is temperature scaling (b_0 = 0), fit_bias=True is Platt scaling with a positive slope.
    """
    logits = torch.cat([ex.logits for ex in examples])
    targets = torch.cat([ex.targets for ex in examples])
    log_beta = torch.zeros((), requires_grad=True)
    bias = torch.zeros((), requires_grad=fit_bias)
    params = [log_beta, bias] if fit_bias else [log_beta]
    optimizer = torch.optim.LBFGS(params, lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(log_beta.exp() * logits + bias, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_beta.detach().exp()), float(bias.detach())


def calibration_metrics(probs, targets, num_bins=15):
    """NLL, Brier, ECE and accuracy at 0.5 over flattened pairs."""
    probs = probs.clamp(1e-7, 1 - 1e-7)
    nll = F.binary_cross_entropy(probs, targets).item()
    brier = ((probs - targets) ** 2).mean().item()
    accuracy = ((probs >= 0.5).float() == (targets >= 0.5).float()).float().mean().item()

    confidence = torch.maximum(probs, 1 - probs)
    correct = ((probs >= 0.5) == (targets >= 0.5)).float()
    bins = torch.clamp(((confidence - 0.5) * 2 * num_bins).long(), max=num_bins - 1)
    ece = 0.0
    for b in range(num_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.float().mean().item() * abs(confidence[in_bin].mean().item() - correct[in_bin].mean().item())
    return {"nll": nll, "brier": brier, "ece": ece, "accuracy": accuracy}


@torch.no_grad()
def evaluate_calibrator(calibrator, examples, device, base_beta=None, base_bias=0.0, batch_size=256):
    """Metrics for the raw model, the global scaling fit (if base_beta) and the calibrator."""
    calibrator.eval()
    raw, scaled, calibrated, targets, betas, biases = [], [], [], [], [], []
    for start in range(0, len(examples), batch_size):
        batch = collate_calibration_examples(examples[start : start + batch_size])
        batch = {key: value.to(device) for key, value in batch.items()}
        beta, bias = calibrator(batch["text_repr"], batch["label_repr"], batch["logits"], batch["label_mask"])
        mask = batch["label_mask"]
        logits = batch["logits"][mask]
        raw.append(torch.sigmoid(logits).cpu())
        if base_beta is not None:
            scaled.append(torch.sigmoid(base_beta * logits + base_bias).cpu())
        calibrated.append(torch.sigmoid(beta[mask] * logits + bias[mask]).cpu())
        targets.append(batch["targets"][mask].cpu())
        betas.append(beta[mask].cpu())
        biases.append(bias[mask].cpu())

    targets = torch.cat(targets)
    results = {"uncalibrated": calibration_metrics(torch.cat(raw), targets)}
    if base_beta is not None:
        results["global_scaling"] = calibration_metrics(torch.cat(scaled), targets)
    results["calibrated"] = calibration_metrics(torch.cat(calibrated), targets)
    for name, values in (("beta", torch.cat(betas)), ("bias", torch.cat(biases))):
        results[name] = {"mean": values.mean().item(), "min": values.min().item(), "max": values.max().item()}
    return results


def fit_calibrator(
    calibrator,
    train_examples,
    eval_examples,
    device,
    base_beta,
    base_bias=0.0,
    prior_weight=0.01,
    bias_prior_weight=0.01,
    num_epochs=20,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.01,
    patience=3,
    seed=42,
    min_base_beta=0.05,
    log_fn=print,
):
    """Train the calibrator MLP on cached features; keeps the weights with the best held-out NLL.

    base_beta is floored at min_base_beta: near beta = 0 the sigmoid parameterization has no gradient,
    so starting from a collapsed global temperature would leave the MLP unable to move.
    """
    if base_beta < min_base_beta:
        log_fn(f"beta_0={base_beta:.3g} is below {min_base_beta}; using {min_base_beta} for the start and the prior")
        base_beta = min_base_beta
    rng = random.Random(seed)
    calibrator.to(device)
    if not calibrator.use_bias:
        base_bias = 0.0
    calibrator.reset_to_global_scaling(base_beta, base_bias)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=lr, weight_decay=weight_decay)

    def eval_nll():
        calibrator.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(eval_examples), 256):
                batch = collate_calibration_examples(eval_examples[start : start + 256])
                batch = {key: value.to(device) for key, value in batch.items()}
                beta, bias = calibrator(
                    batch["text_repr"], batch["label_repr"], batch["logits"], batch["label_mask"]
                )
                _, nll = calibration_loss(
                    beta, bias, batch["logits"], batch["targets"], batch["label_mask"], base_beta, base_bias
                )
                total += nll.item() * batch["logits"].shape[0]
                count += batch["logits"].shape[0]
        return total / max(count, 1)

    best_nll = eval_nll()
    best_state = {key: value.detach().clone() for key, value in calibrator.state_dict().items()}
    log_fn(f"epoch 0: held-out NLL {best_nll:.5f} (global scaling, beta_0={base_beta:.4f}, b_0={base_bias:.4f})")
    bad_epochs = 0

    for epoch in range(1, num_epochs + 1):
        calibrator.train()
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        train_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = collate_calibration_examples([train_examples[i] for i in order[start : start + batch_size]])
            batch = {key: value.to(device) for key, value in batch.items()}
            beta, bias = calibrator(batch["text_repr"], batch["label_repr"], batch["logits"], batch["label_mask"])
            loss, _ = calibration_loss(
                beta,
                bias,
                batch["logits"],
                batch["targets"],
                batch["label_mask"],
                base_beta,
                base_bias,
                prior_weight,
                bias_prior_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch["logits"].shape[0]

        held_out_nll = eval_nll()
        log_fn(f"epoch {epoch}: train loss {train_loss / len(order):.5f}, held-out NLL {held_out_nll:.5f}")
        if held_out_nll < best_nll - 1e-5:
            best_nll = held_out_nll
            best_state = {key: value.detach().clone() for key, value in calibrator.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    calibrator.load_state_dict(best_state)
    calibrator.eval()
    return calibrator
