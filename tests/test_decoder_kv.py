"""Tests for the decoder-KV architecture."""

from types import SimpleNamespace

import torch
import pytest

from gliclass.model import GLiClassOutput, GLiClassDecoderKV


def make_decoder_kv_stub():
    return SimpleNamespace(
        sep_token_id=98,
        config=SimpleNamespace(class_token_index=99),
    )


def test_gliclass_output_exposes_past_key_values():
    cache = object()
    output = GLiClassOutput(logits=torch.zeros(1, 1), past_key_values=cache)
    assert output.past_key_values is cache


def test_extract_label_section_pads_uneven_sections():
    model = make_decoder_kv_stub()
    hidden_states = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    input_ids = torch.tensor(
        [
            [10, 98, 11, 98, 20, 99, 98, 0, 0, 99],
            [98, 12, 98, 21, 22, 99, 23, 99, 98, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    )

    padded_hidden, padded_ids, label_mask = GLiClassDecoderKV._extract_label_section(
        model,
        hidden_states,
        input_ids,
        attention_mask,
    )

    expected_hidden = hidden_states.new_zeros(2, 6, 3)
    expected_hidden[0, :3] = hidden_states[0, 4:7]
    expected_hidden[1] = hidden_states[1, 3:9]
    expected_ids = input_ids.new_zeros(2, 6)
    expected_ids[0, :3] = input_ids[0, 4:7]
    expected_ids[1] = input_ids[1, 3:9]
    expected_mask = attention_mask.new_tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
    )

    torch.testing.assert_close(padded_hidden, expected_hidden)
    torch.testing.assert_close(padded_ids, expected_ids)
    torch.testing.assert_close(label_mask, expected_mask)


def test_extract_label_section_preserves_fallback_and_gradients():
    model = make_decoder_kv_stub()
    hidden_states = torch.randn(1, 5, 3, requires_grad=True)
    input_ids = torch.tensor([[1, 99, 2, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    padded_hidden, padded_ids, label_mask = GLiClassDecoderKV._extract_label_section(
        model,
        hidden_states,
        input_ids,
        attention_mask,
    )

    torch.testing.assert_close(padded_hidden, hidden_states[:, :3])
    torch.testing.assert_close(padded_ids, input_ids[:, :3])
    torch.testing.assert_close(label_mask, attention_mask[:, :3])

    padded_hidden.sum().backward()
    torch.testing.assert_close(hidden_states.grad[:, :3], torch.ones_like(hidden_states[:, :3]))
    torch.testing.assert_close(hidden_states.grad[:, 3:], torch.zeros_like(hidden_states[:, 3:]))


def make_recurrent_stub(**overrides):
    config = SimpleNamespace(
        problem_type="multi_label_classification",
        focal_loss_alpha=-1,
        focal_loss_gamma=-1,
        ignore_index=-100,
        recurrent_steps=4,
        recurrent_min_steps=2,
        recurrent_inference=True,
        recurrent_inference_max_steps=6,
        recurrent_halt_threshold=0.01,
        recurrent_improvement_coef=0.0,
        recurrent_improvement_margin=0.0,
    )
    vars(config).update(overrides)
    stub = SimpleNamespace(config=config, training=True, scorer=SimpleNamespace(recurrent=True))
    for name in ("_resolve_recurrence", "_per_sample_loss", "_per_sample_entropy", "get_recurrent_loss"):
        setattr(stub, name, getattr(GLiClassDecoderKV, name).__get__(stub))
    return stub


def make_scorer_config(**overrides):
    config = SimpleNamespace(
        hidden_size=64,
        scorer_encoder_num_layers=1,
        scorer_mlp_hidden_size=32,
        dropout=0.0,
        normalize_features=False,
        sep_token_index=98,
        class_token_index=99,
        problem_type="multi_label_classification",
        recurrent_steps=1,
    )
    vars(config).update(overrides)
    return config


def make_scorer_inputs():
    hidden = torch.randn(2, 5, 64)
    input_ids = torch.tensor([[10, 99, 11, 99, 98], [12, 99, 98, 0, 0]])
    return hidden, input_ids, input_ids.ne(0).long()


def test_recurrent_scorer_first_step_matches_plain_scorer():
    from gliclass.scorers import DecoderKVScorer

    torch.manual_seed(0)
    plain = DecoderKVScorer(make_scorer_config()).eval()
    recurrent = DecoderKVScorer(make_scorer_config(recurrent_steps=3)).eval()
    recurrent.load_state_dict(plain.state_dict(), strict=False)
    hidden, input_ids, mask = make_scorer_inputs()

    expected = plain(hidden, input_ids, mask)
    steps, steps_taken = recurrent.recurrent_forward(hidden, input_ids, mask, num_steps=3, return_all_steps=True)

    assert len(steps) == 3
    torch.testing.assert_close(steps[0], expected)
    torch.testing.assert_close(recurrent(hidden, input_ids, mask, num_steps=3), steps[-1])
    assert steps_taken.tolist() == [3, 3]


def test_recurrent_scorer_extrapolates_beyond_training_depth():
    from gliclass.scorers import DecoderKVScorer

    scorer = DecoderKVScorer(make_scorer_config(recurrent_steps=2)).eval()
    hidden, input_ids, mask = make_scorer_inputs()
    steps, steps_taken = scorer.recurrent_forward(hidden, input_ids, mask, num_steps=10, return_all_steps=True)
    assert len(steps) == 10
    assert steps_taken.tolist() == [10, 10]
    assert all(torch.isfinite(logits).all() for logits in steps)


def test_recurrent_scorer_halts_when_probabilities_stop_changing():
    from gliclass.scorers import DecoderKVScorer

    scorer = DecoderKVScorer(make_scorer_config(recurrent_steps=2)).eval()
    hidden, input_ids, mask = make_scorer_inputs()
    # probabilities can never change by more than 1, so every example halts after step 2
    steps, steps_taken = scorer.recurrent_forward(
        hidden, input_ids, mask, num_steps=10, halt_threshold=1.1, return_all_steps=True
    )
    assert len(steps) == 2
    assert steps_taken.tolist() == [2, 2]


def test_plain_scorer_rejects_multiple_steps():
    from gliclass.scorers import DecoderKVScorer

    scorer = DecoderKVScorer(make_scorer_config())
    hidden, input_ids, mask = make_scorer_inputs()
    with pytest.raises(ValueError):
        scorer(hidden, input_ids, mask, num_steps=2)


def test_resolve_recurrence_training_inference_and_overrides():
    stub = make_recurrent_stub()
    sampled = {stub._resolve_recurrence()[0] for _ in range(200)}
    assert sampled == {2, 3, 4}
    assert stub._resolve_recurrence()[1] is None
    assert stub._resolve_recurrence(use_recurrence=False) == (1, None)

    stub.training = False
    assert stub._resolve_recurrence() == (6, 0.01)
    assert stub._resolve_recurrence(max_recurrent_steps=12, halt_threshold=0.0) == (12, 0.0)
    assert stub._resolve_recurrence(use_recurrence=False) == (1, None)
    stub.config.recurrent_inference = False
    assert stub._resolve_recurrence() == (1, None)
    assert stub._resolve_recurrence(use_recurrence=True) == (6, 0.01)

    stub.scorer.recurrent = False
    assert stub._resolve_recurrence(use_recurrence=True) == (1, None)


def test_improvement_term_starts_at_step_three_and_spares_earlier_steps():
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # later steps are worse than earlier ones, so the hinge is active
    values = [
        [[2.0, -2.0], [-2.0, 2.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ]

    def step_grads(coef, num_steps):
        stub = make_recurrent_stub(recurrent_improvement_coef=coef)
        step_logits = [torch.tensor(value, requires_grad=True) for value in values[:num_steps]]
        loss, step_losses, _ = stub.get_recurrent_loss(step_logits, labels)
        loss.backward()
        assert step_losses.shape == (num_steps,)
        return [logits.grad for logits in step_logits]

    # two steps: no improvement term at all
    for with_hinge, without in zip(step_grads(1.0, 2), step_grads(0.0, 2)):
        torch.testing.assert_close(with_hinge, without)

    with_hinge, without = step_grads(1.0, 3), step_grads(0.0, 3)
    torch.testing.assert_close(with_hinge[0], without[0])
    torch.testing.assert_close(with_hinge[1], without[1])  # baseline of the step-3 hinge is stop-gradient
    assert with_hinge[2].abs().sum() > without[2].abs().sum()


def test_recurrent_loss_ignores_padded_labels():
    stub = make_recurrent_stub()
    labels = torch.tensor([[1.0, -100.0]])
    step_logits = [torch.tensor([[2.0, 50.0]]), torch.tensor([[3.0, -50.0]])]
    _, step_losses, _ = stub.get_recurrent_loss(step_logits, labels)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2.0, 3.0]), torch.ones(2), reduction="none"
    )
    torch.testing.assert_close(step_losses, expected)


def test_read_text_scorer_requires_and_uses_text():
    from gliclass.scorers import DecoderKVScorer

    torch.manual_seed(0)
    plain = DecoderKVScorer(make_scorer_config()).eval()
    scorer = DecoderKVScorer(make_scorer_config(recurrent_steps=3, recurrent_read_text=True)).eval()
    scorer.load_state_dict(plain.state_dict(), strict=False)
    hidden, input_ids, mask = make_scorer_inputs()
    text = torch.randn(2, 7, 64)
    text_mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3])

    with pytest.raises(ValueError):
        scorer(hidden, input_ids, mask, num_steps=2)

    steps, _ = scorer.recurrent_forward(
        hidden,
        input_ids,
        mask,
        num_steps=3,
        return_all_steps=True,
        text_hidden_states=text,
        text_attention_mask=text_mask,
    )
    torch.testing.assert_close(steps[0], plain(hidden, input_ids, mask))

    # recurrent steps depend on the text; padded text positions do not
    other_text = text.clone()
    other_text[0] += torch.randn(7, 64)
    other_text[1, 4:] += 100.0
    changed, _ = scorer.recurrent_forward(
        hidden,
        input_ids,
        mask,
        num_steps=3,
        return_all_steps=True,
        text_hidden_states=other_text,
        text_attention_mask=text_mask,
    )
    assert not torch.allclose(changed[-1][0], steps[-1][0])
    torch.testing.assert_close(changed[-1][1], steps[-1][1])


def test_extract_label_section_returns_text_section():
    model = make_decoder_kv_stub()
    hidden_states = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    input_ids = torch.tensor(
        [
            [10, 98, 11, 98, 20, 99, 98, 0, 0, 99],
            [98, 12, 98, 21, 22, 99, 23, 99, 98, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    )

    *_, text_hidden, text_mask = GLiClassDecoderKV._extract_label_section(
        model, hidden_states, input_ids, attention_mask, return_text_section=True
    )

    torch.testing.assert_close(text_mask, attention_mask.new_tensor([[1, 1, 1, 1], [1, 1, 1, 0]]))
    torch.testing.assert_close(text_hidden[0], hidden_states[0, :4])
    torch.testing.assert_close(text_hidden[1, :3], hidden_states[1, :3])


def test_recurrent_lr_moves_reasoning_cell_into_own_groups():
    from gliclass.training import Trainer

    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2)
    model.reasoning_cell = torch.nn.Linear(2, 2)
    groups = [
        {"params": [model.backbone.weight, model.reasoning_cell.weight], "weight_decay": 0.01, "lr": 1e-5},
        {"params": [model.backbone.bias, model.reasoning_cell.bias], "weight_decay": 0.0, "lr": 1e-5},
    ]
    stub = SimpleNamespace(args=SimpleNamespace(recurrent_lr=1e-3))

    split = Trainer._split_recurrent_groups(stub, model, groups)

    assert [len(group["params"]) for group in split] == [1, 1, 1, 1]
    assert [group["lr"] for group in split] == [1e-5, 1e-3, 1e-5, 1e-3]
    assert [group["weight_decay"] for group in split] == [0.01, 0.01, 0.0, 0.0]
    assert split[1]["params"][0] is model.reasoning_cell.weight


def test_per_sample_entropy_is_binary_entropy_over_real_labels():
    stub = make_recurrent_stub()
    logits = torch.tensor([[0.0, 30.0, 5.0]])
    valid = torch.tensor([[True, True, False]])
    entropy = stub._per_sample_entropy(logits, valid)
    # p=0.5 -> log 2, p~1 -> 0; the padded third label is ignored
    torch.testing.assert_close(entropy, torch.tensor([torch.log(torch.tensor(2.0)).item() / 2]), atol=1e-6, rtol=0)


def test_relative_confidence_loss_sharpens_later_step_only():
    labels = torch.tensor([[1.0, 0.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)

    def grads(coef, mode="relative"):
        stub = make_recurrent_stub(recurrent_confidence_coef=coef, recurrent_confidence_mode=mode)
        # step 2 is less certain than step 1, so the relative hinge is active
        step_logits = [torch.tensor([[2.0, -2.0]], requires_grad=True), torch.tensor([[0.5, -0.5]], requires_grad=True)]
        loss, _, entropies = stub.get_recurrent_loss(step_logits, labels, valid)
        loss.backward()
        return [logits.grad for logits in step_logits], entropies

    without, entropies = grads(0.0)
    with_conf, _ = grads(1.0)
    assert entropies[1] > entropies[0]
    # earlier step is a stop-gradient baseline
    torch.testing.assert_close(with_conf[0], without[0])
    # gradient descent moves step-2 logits further from 0 (probabilities away from 0.5)
    extra = with_conf[1] - without[1]
    assert extra[0, 0] < 0 and extra[0, 1] > 0

    absolute, _ = grads(1.0, mode="absolute")
    torch.testing.assert_close(absolute[0], without[0])
    assert (absolute[1] - without[1])[0, 0] < 0


def test_confidence_loss_inactive_when_later_step_already_more_certain():
    stub = make_recurrent_stub(recurrent_confidence_coef=1.0)
    labels = torch.tensor([[1.0, 0.0]])
    step_logits = [torch.tensor([[0.5, -0.5]]), torch.tensor([[3.0, -3.0]])]
    with_conf, _, _ = stub.get_recurrent_loss(step_logits, labels)
    stub.config.recurrent_confidence_coef = 0.0
    without, _, _ = stub.get_recurrent_loss(step_logits, labels)
    torch.testing.assert_close(with_conf, without)


def test_single_label_losses_accept_class_index_targets():
    stub = make_recurrent_stub(problem_type="single_label_classification", recurrent_confidence_coef=0.0)
    stub.get_loss = GLiClassDecoderKV.get_loss.__get__(stub)
    labels = torch.tensor([2, 0])
    logits = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.0, 0.0]])
    torch.testing.assert_close(stub.get_loss(logits, labels), torch.nn.functional.cross_entropy(logits, labels))
    _, step_losses, _ = stub.get_recurrent_loss([logits, logits], labels)
    torch.testing.assert_close(step_losses, torch.nn.functional.cross_entropy(logits, labels).repeat(2))
