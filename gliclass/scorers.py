import torch
from torch import nn

from .ops import attn_padded


class ScorerWeightedDot(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, **kwargs):
        super().__init__()

        self.proj_text = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_label = nn.Linear(hidden_size, hidden_size * 2)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, 1),  # start, end, score
        )

    def forward(self, text_rep, label_rep, **kwargs):
        batch_size, hidden_size = text_rep.shape
        num_classes = label_rep.shape[1]

        # (batch_size, 1, 3, hidden_size)
        text_rep = self.proj_text(text_rep).view(batch_size, 1, 1, 2, hidden_size)
        label_rep = self.proj_label(label_rep).view(batch_size, 1, num_classes, 2, hidden_size)

        # (2, batch_size, 1, num_classes, hidden_size)
        text_rep = text_rep.expand(-1, -1, num_classes, -1, -1).permute(3, 0, 1, 2, 4)
        label_rep = label_rep.expand(-1, 1, -1, -1, -1).permute(3, 0, 1, 2, 4)

        # (batch_size, 1, num_classes, hidden_size * 3)
        cat = torch.cat([text_rep[0], label_rep[0], text_rep[1] * label_rep[1]], dim=-1)

        # (batch_size, num_classes)
        scores = self.out_mlp(cat).view(batch_size, num_classes)

        return scores


class ScorerDot(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, text_rep, label_rep, **kwargs):
        # dot product with einsum
        scores = torch.einsum("BD,BCD->BC", text_rep, label_rep)
        return scores


class MLPScorer(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size=256, **kwargs):
        super().__init__()

        # Calculate the input size for the MLP
        total_input_size = hidden_size * 2

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(total_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size // 2, 1),
        )

    def forward(self, text_rep, label_rep, **kwargs):
        # Concatenate text and label representations
        batch_size, num_labels, dim = label_rep.shape
        text_rep = text_rep.unsqueeze(1).expand(batch_size, num_labels, dim)
        combined_rep = torch.cat([text_rep, label_rep], dim=-1)

        # Pass through MLP
        scores = self.mlp(combined_rep).squeeze(-1)

        return scores


class HopfieldScorer(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size=256, beta=4, num_iteration=1, **kwargs):
        super().__init__()

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size // 2, 1),
        )

        self.beta = beta
        self.num_iteration = num_iteration

    def forward(self, text_rep, label_rep, **kwargs):
        """
        text_rep: [batch_size, hidden_size]
        label_rep: [batch_size, num_labels, hidden_size].
        """
        for _i in range(self.num_iteration):
            # Expand text_rep to match label_rep's batch shape
            text_rep_expanded = text_rep.unsqueeze(1)  # [batch_size, 1, dim]

            # Compute Q, K, V
            query = self.q_proj(label_rep)  # [batch_size, num_labels, dim]
            key = self.k_proj(text_rep_expanded)  # [batch_size, 1, dim]
            value = self.v_proj(text_rep_expanded)  # [batch_size, 1, dim]

            attn = torch.bmm(query, key.transpose(1, 2))  # [b, num_labels, 1]
            attn = attn * self.beta  # optional beta scaling
            attn = torch.nn.functional.softmax(attn, dim=1)  # softmax over labels

            context = attn * value  # [b, num_labels, dim]

            label_rep = label_rep + context

        scores = self.mlp(label_rep).squeeze(-1)  # [b, num_labels]

        return scores


class CrossAttnScorer(nn.Module):
    def __init__(self, hidden_size, num_heads=16, attn_dropout=0.1, scorer_mlp_hidden_size=1024, **kwargs):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout

        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, scorer_mlp_hidden_size),
            nn.GELU(),
            nn.Linear(scorer_mlp_hidden_size, scorer_mlp_hidden_size // 2),
            nn.GELU(),
            nn.Linear(scorer_mlp_hidden_size // 2, 1),
        )

    def forward(self, text_rep, label_rep, text_mask=None, **kwargs):
        batch_size, _, hidden_size = text_rep.shape
        num_labels = label_rep.shape[1]

        if text_mask is None:
            text_mask = torch.ones(batch_size, text_rep.shape[1], dtype=torch.bool, device=text_rep.device)

        q = self.q(self.q_norm(label_rep)).view(batch_size, num_labels, self.num_heads, self.head_dim)
        k = self.k(self.kv_norm(text_rep)).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v(text_rep).view(batch_size, -1, self.num_heads, self.head_dim)

        dropout_p = self.attn_dropout if self.training else 0.0
        context = attn_padded(q, k, v, key_padding_mask=text_mask, dropout_p=dropout_p)
        context = self.norm(self.out(context.reshape(batch_size, num_labels, hidden_size)))

        return self.score_mlp(torch.cat([context, label_rep], dim=-1)).squeeze(-1)


class DecoderKVScorer(nn.Module):
    """
    Scorer for decoder-kv architecture with built-in bidirectional encoder and representation extraction.

    Sequence format: [prompt][examples]text<<SEP>>label1<<LABEL>>label2<<LABEL>>...<<SEP>>

    Flow:
    1. Takes hidden states from decoder backbone
    2. Applies bidirectional scorer_encoder (DebertaV2Encoder without embeddings)
    3. Extracts text repr from last <<SEP>> before labels
    4. Extracts label repr from each <<LABEL>> token
    5. Computes scores via MLP (concat text + label)
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        from transformers import DebertaV2Config
        from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder

        num_layers = getattr(config, "scorer_encoder_num_layers", 2)
        num_heads = max(1, config.hidden_size // 64)

        encoder_config = DebertaV2Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=config.hidden_size * 4,
            relative_attention=True,
            pos_att_type=["p2c", "c2p"],
            max_relative_positions=512,
        )
        self.scorer_encoder = DebertaV2Encoder(encoder_config)

        self.text_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.label_projector = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        mlp_hidden_size = getattr(config, "scorer_mlp_hidden_size", 1024)
        total_input_size = config.hidden_size * 2

        self.mlp = nn.Sequential(
            nn.Linear(total_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size // 2, 1),
        )

        self.epsilon = 1e-8

        # GRU-style recurrent reasoning over scorer_encoder. Gates are step-agnostic (no step
        # embeddings), so inference can run more steps than were seen during training.
        self.recurrent = getattr(config, "recurrent_steps", 1) > 1
        if self.recurrent:
            self.reset_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.update_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def _extract_representations(self, hidden_states, input_ids, attention_mask):
        """
        Extract text and label representations from hidden states.

        Sequence: text<<SEP>>label1<<LABEL>>label2<<LABEL>><<SEP>>
        Text repr: LAST <<SEP>> token (after all labels) - sees both text and labels
        Label repr: each <<LABEL>> token
        """
        batch_size, seq_length, hidden_size = hidden_states.shape

        sep_token_id = self.config.sep_token_index
        label_token_id = self.config.class_token_index

        valid_mask = attention_mask.bool()
        sep_mask = input_ids.eq(sep_token_id) & valid_mask
        label_mask = input_ids.eq(label_token_id) & valid_mask

        positions = torch.arange(seq_length, device=input_ids.device).expand(batch_size, -1)
        last_sep_positions = positions.masked_fill(~sep_mask, -1).amax(dim=1)
        has_sep = last_sep_positions.ge(0)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        text_repr = hidden_states[batch_indices, last_sep_positions.clamp_min(0)]
        text_repr = text_repr * has_sep.unsqueeze(-1)

        num_labels_per_batch = label_mask.sum(dim=1)
        max_labels = num_labels_per_batch.max().item()
        label_repr = hidden_states.new_zeros(
            batch_size,
            max_labels,
            hidden_size,
        )

        label_batch_indices, label_token_indices = label_mask.nonzero(as_tuple=True)
        label_ranks = label_mask.long().cumsum(dim=1) - 1
        label_indices = label_ranks[label_batch_indices, label_token_indices]
        label_repr[label_batch_indices, label_indices] = hidden_states[label_batch_indices, label_token_indices]

        return text_repr, label_repr

    def forward(self, hidden_states, input_ids, attention_mask, **kwargs):
        """Forward pass.

        Args:
            hidden_states: (batch_size, seq_length, hidden_size) from decoder backbone
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            **kwargs: recurrence options, see recurrent_forward

        Returns:
            logits: (batch_size, num_labels)
        """
        step_logits, _ = self.recurrent_forward(hidden_states, input_ids, attention_mask, **kwargs)
        return step_logits[-1]

    def _encode(self, hidden_states, attention_mask):
        return self.scorer_encoder(hidden_states, attention_mask=attention_mask, return_dict=True).last_hidden_state

    def _recurrent_step(self, state, inputs, attention_mask):
        """h_t = (1 - z) * h_{t-1} + z * Enc(r * h_{t-1} + (1 - r) * x), x = decoder hidden states."""
        reset = torch.sigmoid(self.reset_gate(torch.cat([state, inputs], dim=-1)))
        candidate = self._encode(reset * state + (1 - reset) * inputs, attention_mask)
        update = torch.sigmoid(self.update_gate(torch.cat([state, candidate], dim=-1)))
        return (1 - update) * state + update * candidate

    def _valid_label_mask(self, input_ids, attention_mask, num_labels):
        label_counts = (input_ids.eq(self.config.class_token_index) & attention_mask.bool()).sum(dim=1)
        return torch.arange(num_labels, device=input_ids.device).unsqueeze(0) < label_counts.unsqueeze(1)

    def _probabilities(self, logits, valid_labels):
        if self.config.problem_type == "single_label_classification":
            return torch.softmax(logits.masked_fill(~valid_labels, float("-inf")), dim=-1).nan_to_num(0.0)
        return torch.sigmoid(logits)

    def recurrent_forward(
        self,
        hidden_states,
        input_ids,
        attention_mask,
        num_steps=1,
        halt_threshold=None,
        return_all_steps=False,
        **kwargs,
    ):
        """Run step 1 (one scorer_encoder pass) and up to num_steps - 1 gated recurrent steps.

        With halt_threshold, an example stops once no valid label probability moves by
        halt_threshold or more between consecutive steps; its state and logits are frozen.

        Returns:
            step_logits: list of (batch_size, num_labels) logits for every step if
                return_all_steps, otherwise a single-element list with the final logits
            steps_taken: (batch_size,) number of steps each example ran
        """
        if num_steps > 1 and not self.recurrent:
            raise ValueError("num_steps > 1 requires a scorer built with config.recurrent_steps > 1.")

        bptt_steps = getattr(self.config, "recurrent_bptt_steps", None)

        state = self._encode(hidden_states, attention_mask)
        logits = self._score(state, input_ids, attention_mask)
        step_logits = [logits]

        batch_size, num_labels = logits.shape
        steps_taken = torch.ones(batch_size, dtype=torch.long, device=logits.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
        if halt_threshold:
            valid_labels = self._valid_label_mask(input_ids, attention_mask, num_labels)

        for step in range(1, num_steps):
            if bptt_steps and step % bptt_steps == 0:
                # truncated BPTT: later losses stop shaping earlier iterations
                state = state.detach()

            new_state = self._recurrent_step(state, hidden_states, attention_mask)
            new_logits = self._score(new_state, input_ids, attention_mask)

            if halt_threshold:
                change = (
                    (self._probabilities(new_logits, valid_labels) - self._probabilities(logits, valid_labels))
                    .abs()
                    .masked_fill(~valid_labels, 0.0)
                    .amax(dim=-1)
                )
                new_state = torch.where(active[:, None, None], new_state, state)
                new_logits = torch.where(active[:, None], new_logits, logits)
                steps_taken = steps_taken + active.long()
                active = active & change.ge(halt_threshold)
            else:
                steps_taken = steps_taken + 1

            state, logits = new_state, new_logits
            if return_all_steps:
                step_logits.append(logits)
            else:
                step_logits[-1] = logits

            if halt_threshold and not active.any():
                break

        if self.recurrent and self.training:
            # keep the gates in the graph for DDP when the sampled depth is 1
            unused = sum(param.sum() for param in (*self.reset_gate.parameters(), *self.update_gate.parameters()))
            step_logits[-1] = step_logits[-1] + 0.0 * unused

        return step_logits, steps_taken

    def _score(self, contextualized_hidden_states, input_ids, attention_mask):
        text_repr, label_repr = self._extract_representations(contextualized_hidden_states, input_ids, attention_mask)

        text_repr = self.text_projector(text_repr)
        text_repr = self.dropout(text_repr)

        label_repr = self.label_projector(label_repr)

        if self.config.normalize_features:
            text_repr = text_repr / (text_repr.norm(p=2, dim=-1, keepdim=True) + self.epsilon)
            label_repr = label_repr / (label_repr.norm(p=2, dim=-1, keepdim=True) + self.epsilon)

        batch_size, num_labels, dim = label_repr.shape
        text_repr_expanded = text_repr.unsqueeze(1).expand(batch_size, num_labels, dim)
        combined_rep = torch.cat([text_repr_expanded, label_repr], dim=-1)

        logits = self.mlp(combined_rep).squeeze(-1)

        return logits


SCORER2OBJECT = {
    "weighted-dot": ScorerWeightedDot,
    "simple": ScorerDot,
    "mlp": MLPScorer,
    "hopfield": HopfieldScorer,
    "cross-attn": CrossAttnScorer,
    "decoder-kv": DecoderKVScorer,
}
