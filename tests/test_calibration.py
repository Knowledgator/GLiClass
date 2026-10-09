"""Tests for the post-hoc calibrator."""

import math

import torch
import pytest
from torch.utils.data import DataLoader

from gliclass import GLiClassModel, GLiClassModelConfig
from gliclass.pipeline import BaseZeroShotClassificationPipeline
from gliclass.calibration import (
    CalibrationExample,
    GLiClassCalibrator,
    fit_calibrator,
    logit_features,
    calibration_loss,
    fit_global_scaling,
    evaluate_calibrator,
    collect_calibration_examples,
)


@pytest.mark.parametrize("use_bias", [False, True])
def test_calibrator_starts_at_global_scaling_and_ignores_padding(use_bias):
    calibrator = GLiClassCalibrator(8, hidden_size=16, beta_max=3.0, use_bias=use_bias)
    calibrator.reset_to_global_scaling(0.7, -1.5)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    beta, bias = calibrator(torch.randn(2, 8), torch.randn(2, 3, 8), torch.randn(2, 3), mask)

    torch.testing.assert_close(beta[mask.bool()], torch.full((3,), 0.7))
    torch.testing.assert_close(bias[mask.bool()], torch.full((3,), -1.5 if use_bias else 0.0))
    torch.testing.assert_close(beta[~mask.bool()], torch.ones(3))
    torch.testing.assert_close(bias[~mask.bool()], torch.zeros(3))


def test_beta_is_bounded_and_never_flips_decisions_without_bias():
    torch.manual_seed(0)
    calibrator = GLiClassCalibrator(8, hidden_size=16, beta_max=2.0)
    torch.nn.init.normal_(calibrator.mlp[-1].weight, std=1.0)
    logits = torch.randn(4, 5) * 5
    beta, bias = calibrator(torch.randn(4, 8), torch.randn(4, 5, 8), logits, torch.ones(4, 5))

    assert (bias == 0).all()
    assert (beta > 0).all() and (beta < 2.0).all()
    assert torch.equal(torch.sign(beta * logits), torch.sign(logits))


def test_logit_features_ignore_padded_labels():
    logits = torch.tensor([[2.0, -1.0, 100.0]])
    mask = torch.tensor([[1, 1, 0]])
    features = logit_features(logits, mask)

    assert features[0, 2].abs().sum() == 0
    torch.testing.assert_close(features[0, :2, 2], torch.tensor([0.0, -3.0]))  # z - max over real labels
    torch.testing.assert_close(features[0, :2, 5], torch.full((2,), math.log(2)))


def test_nll_pushes_beta_down_for_wrong_pairs_and_up_for_correct_ones():
    logits = torch.tensor([[3.0, 3.0]])
    targets = torch.tensor([[0.0, 1.0]])  # first pair is confidently wrong
    beta = torch.ones(1, 2, requires_grad=True)
    loss, _ = calibration_loss(
        beta, torch.zeros(1, 2), logits, targets, torch.ones(1, 2, dtype=torch.bool), 1.0, prior_weight=0.0
    )
    loss.backward()

    assert beta.grad[0, 0] > 0  # gradient descent lowers beta -> probability moves to 0.5
    assert beta.grad[0, 1] < 0


def make_synthetic_examples(num_texts, seed):
    """Logits that are right on 'easy' labels and sign-flipped on labels marked hard in their representation."""
    generator = torch.Generator().manual_seed(seed)
    examples = []
    for _ in range(num_texts):
        num_labels = int(torch.randint(2, 6, (1,), generator=generator))
        targets = torch.randint(0, 2, (num_labels,), generator=generator).float()
        hard = torch.rand(num_labels, generator=generator) < 0.3
        logits = (targets * 2 - 1) * 4.0
        logits = torch.where(hard, -logits, logits)
        label_repr = torch.randn(num_labels, 8, generator=generator) * 0.1
        label_repr[:, 0] = hard.float() * 3
        examples.append(CalibrationExample(torch.randn(8, generator=generator), label_repr, logits, targets))
    return examples


def test_fit_learns_to_lower_confidence_where_backbone_is_wrong():
    torch.manual_seed(0)
    train, held_out = make_synthetic_examples(400, seed=1), make_synthetic_examples(100, seed=2)
    base_beta, _ = fit_global_scaling(train)
    calibrator = fit_calibrator(
        GLiClassCalibrator(8, hidden_size=32),
        train,
        held_out,
        "cpu",
        base_beta=base_beta,
        num_epochs=30,
        lr=3e-3,
        log_fn=lambda *_: None,
    )
    report = evaluate_calibrator(calibrator, held_out, "cpu", base_beta)

    assert report["calibrated"]["nll"] < report["global_scaling"]["nll"] < report["uncalibrated"]["nll"]
    assert report["calibrated"]["accuracy"] == pytest.approx(report["uncalibrated"]["accuracy"])

    with torch.no_grad():
        example = held_out[0]
        hard = example.label_repr[:, 0] > 1
        beta = calibrator(
            example.text_repr[None], example.label_repr[None], example.logits[None], torch.ones(1, len(hard))
        )[0][0]
    if hard.any() and (~hard).any():
        assert beta[hard].max() < beta[~hard].min()


def make_overconfident_examples(num_texts, seed):
    """A backbone that calls most negatives positive: 1 positive among 8 labels, negatives at logit ~ +2,
    positives at ~ +5. Ranking is informative, but every logit is positive.
    """
    generator = torch.Generator().manual_seed(seed)
    examples = []
    for _ in range(num_texts):
        targets = torch.zeros(8)
        targets[int(torch.randint(0, 8, (1,), generator=generator))] = 1.0
        logits = 2.0 + 3.0 * targets + torch.randn(8, generator=generator) * 0.7
        label_repr = torch.randn(8, 8, generator=generator) * 0.1
        examples.append(CalibrationExample(torch.randn(8, generator=generator), label_repr, logits, targets))
    return examples


def test_global_platt_scaling_recovers_negative_bias():
    beta, bias = fit_global_scaling(make_overconfident_examples(300, seed=0), fit_bias=True)
    assert beta > 0 and bias < 0
    _, no_bias = fit_global_scaling(make_overconfident_examples(300, seed=0), fit_bias=False)
    assert no_bias == 0.0


def test_bias_fixes_base_rate_miscalibration_that_scaling_cannot():
    torch.manual_seed(0)
    train, held_out = make_overconfident_examples(400, seed=1), make_overconfident_examples(100, seed=2)
    reports = {}
    for use_bias in (False, True):
        base_beta, base_bias = fit_global_scaling(train, fit_bias=use_bias)
        calibrator = fit_calibrator(
            GLiClassCalibrator(8, hidden_size=32, use_bias=use_bias),
            train,
            held_out,
            "cpu",
            base_beta=base_beta,
            base_bias=base_bias,
            num_epochs=10,
            lr=3e-3,
            log_fn=lambda *_: None,
        )
        reports[use_bias] = evaluate_calibrator(calibrator, held_out, "cpu", base_beta, base_bias)

    # without the bias every logit stays positive, so accuracy is stuck at the positive rate
    assert reports[False]["calibrated"]["accuracy"] == pytest.approx(reports[False]["uncalibrated"]["accuracy"])
    assert reports[True]["calibrated"]["accuracy"] > 0.9
    assert reports[True]["calibrated"]["nll"] < 0.5 * reports[False]["calibrated"]["nll"]


@pytest.fixture
def tiny_uni_encoder():
    torch.manual_seed(0)
    encoder_config = {
        "model_type": "deberta-v2",
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 64,
    }
    config = GLiClassModelConfig(
        encoder_config=encoder_config,
        encoder_model="tiny-deberta",  # never loaded: the backbone is built from encoder_config
        class_token_index=60,
        text_token_index=61,
        architecture_type="uni-encoder",
        problem_type="multi_label_classification",
        max_num_classes=4,
    )
    model = GLiClassModel(config).eval()
    # two texts, the second with only two labels
    input_ids = torch.tensor(
        [
            [60, 5, 60, 6, 60, 7, 61, 10, 11, 12],
            [60, 5, 60, 6, 61, 13, 14, 0, 0, 0],
        ]
    )
    attention_mask = (input_ids != 0).long()
    return model, {"input_ids": input_ids, "attention_mask": attention_mask, "max_num_classes": 3}


def test_model_returns_calibrated_and_initial_logits(tiny_uni_encoder, tmp_path):
    model, inputs = tiny_uni_encoder
    with torch.no_grad():
        raw = model(**inputs).logits
    assert model(**inputs).uncalibrated_logits is None

    calibrator = model.add_calibrator(hidden_size=16, use_bias=True)
    calibrator.reset_to_global_scaling(0.5, -1.0)
    with torch.no_grad():
        output = model(**inputs)

    torch.testing.assert_close(output.uncalibrated_logits, raw)
    torch.testing.assert_close(output.class_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
    torch.testing.assert_close(output.logits[0], 0.5 * raw[0] - 1.0)
    torch.testing.assert_close(output.calibration_biases[1], torch.tensor([-1.0, -1.0, 0.0]))
    assert output.text_embeddings is None and output.class_embeddings is None

    model.save_pretrained(tmp_path)
    reloaded = GLiClassModel.from_pretrained(tmp_path).eval()
    assert reloaded.calibrator is not None and reloaded.calibrator.use_bias
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**inputs).logits, output.logits)


def test_collect_calibration_examples_uses_real_labels_only(tiny_uni_encoder):
    model, inputs = tiny_uni_encoder
    batch = dict(inputs, labels=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    examples = collect_calibration_examples(model, DataLoader([batch], batch_size=None), "cpu", show_progress=False)

    assert [len(ex.logits) for ex in examples] == [3, 2]
    torch.testing.assert_close(examples[1].targets, torch.tensor([0.0, 1.0]))
    assert examples[0].label_repr.shape == (3, model.config.encoder_config.hidden_size)


def test_postprocess_reports_initial_score():
    predictions, all_scores = BaseZeroShotClassificationPipeline._postprocess_logits(
        torch.tensor([0.5, -2.0]),
        ["a", "b"],
        "multi-label",
        0.5,
        initial_logits=torch.tensor([3.0, -4.0]),
    )
    assert predictions == [
        {"label": "a", "score": pytest.approx(torch.sigmoid(torch.tensor(0.5)).item()),
         "initial_score": pytest.approx(torch.sigmoid(torch.tensor(3.0)).item())}
    ]
    assert set(all_scores) == {"a", "b"}

    predictions, _ = BaseZeroShotClassificationPipeline._postprocess_logits(
        torch.tensor([0.5, -2.0]), ["a", "b"], "multi-label", 0.5
    )
    assert "initial_score" not in predictions[0]


@pytest.mark.parametrize("halt_threshold", [None, 1e-3, 1.1])
def test_decoder_kv_representations_match_final_logits(halt_threshold):
    from types import SimpleNamespace

    from gliclass.scorers import DecoderKVScorer

    torch.manual_seed(0)
    config = SimpleNamespace(
        hidden_size=64,
        scorer_encoder_num_layers=1,
        scorer_mlp_hidden_size=32,
        dropout=0.0,
        normalize_features=False,
        sep_token_index=98,
        class_token_index=99,
        problem_type="multi_label_classification",
        recurrent_steps=3,
    )
    scorer = DecoderKVScorer(config).eval()
    hidden = torch.randn(2, 5, 64)
    input_ids = torch.tensor([[10, 99, 11, 99, 98], [12, 99, 98, 0, 0]])
    steps, _, (text_repr, label_repr) = scorer.recurrent_forward(
        hidden, input_ids, input_ids.ne(0).long(), num_steps=4, halt_threshold=halt_threshold,
        return_representations=True,
    )

    combined = torch.cat([text_repr.unsqueeze(1).expand_as(label_repr), label_repr], dim=-1)
    torch.testing.assert_close(scorer.mlp(combined).squeeze(-1), steps[-1])
