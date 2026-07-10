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

    def _extract_representations(self, hidden_states, input_ids, attention_mask):
        """
        Extract text and label representations from hidden states.

        Sequence: text<<SEP>>label1<<LABEL>>label2<<LABEL>><<SEP>>
        Text repr: LAST <<SEP>> token (after all labels) - sees both text and labels
        Label repr: each <<LABEL>> token
        """
        batch_size, _, hidden_size = hidden_states.shape

        sep_token_id = self.config.sep_token_index
        label_token_id = self.config.class_token_index

        sep_mask = input_ids == sep_token_id
        label_mask = input_ids == label_token_id

        sep_positions = torch.where(sep_mask)
        label_positions = torch.where(label_mask)

        text_repr = torch.zeros(batch_size, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)

        for batch_idx in range(batch_size):
            batch_sep_positions = sep_positions[1][sep_positions[0] == batch_idx]
            if len(batch_sep_positions) > 0:
                # Take LAST SEP (after all labels) so text repr sees both text and labels
                last_sep_pos = batch_sep_positions[-1]
                text_repr[batch_idx] = hidden_states[batch_idx, last_sep_pos]

        num_labels_per_batch = torch.zeros(batch_size, dtype=torch.long, device=hidden_states.device)
        for batch_idx in range(batch_size):
            num_labels_per_batch[batch_idx] = (label_positions[0] == batch_idx).sum()

        max_labels = num_labels_per_batch.max().item()
        label_repr = torch.zeros(
            batch_size,
            max_labels,
            hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for batch_idx in range(batch_size):
            batch_label_positions = label_positions[1][label_positions[0] == batch_idx]
            for label_idx, pos in enumerate(batch_label_positions):
                if label_idx < max_labels:
                    label_repr[batch_idx, label_idx] = hidden_states[batch_idx, pos]

        return text_repr, label_repr

    def forward(self, hidden_states, input_ids, attention_mask, **kwargs):
        """Forward pass.

        Args:
            hidden_states: (batch_size, seq_length, hidden_size) from decoder backbone
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)

        Returns:
            logits: (batch_size, num_labels)
        """
        encoder_outputs = self.scorer_encoder(hidden_states, attention_mask=attention_mask, return_dict=True)

        contextualized_hidden_states = encoder_outputs.last_hidden_state

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
