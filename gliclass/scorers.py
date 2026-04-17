import torch
from torch import nn
from .ops import attn_varlen, attn_padded, unpad_input, pad_input

class ScorerWeightedDot(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, **kwargs):
        super().__init__()

        self.proj_text = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_label = nn.Linear(hidden_size, hidden_size * 2)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, 1)  # start, end, score
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
        scores = torch.einsum('BD,BCD->BC', text_rep, label_rep)
        return scores

class MLPScorer(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size=256, **kwargs):
        super().__init__()

        # Calculate the input size for the MLP
        total_input_size = hidden_size*2

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(total_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size // 2, 1)
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
            nn.Linear(mlp_hidden_size // 2, 1)
        )

        self.beta = beta
        self.num_iteration = num_iteration

    def forward(self, text_rep, label_rep, **kwargs):
        """
        text_rep: [batch_size, hidden_size]
        label_rep: [batch_size, num_labels, hidden_size]
        """
        for i in range(self.num_iteration):
            # Expand text_rep to match label_rep's batch shape
            text_rep_expanded = text_rep.unsqueeze(1)  # [batch_size, 1, dim]

            # Compute Q, K, V
            query = self.q_proj(label_rep)           # [batch_size, num_labels, dim]
            key   = self.k_proj(text_rep_expanded)   # [batch_size, 1, dim]
            value = self.v_proj(text_rep_expanded)   # [batch_size, 1, dim]


            attn = torch.bmm(query, key.transpose(1, 2))  # [b, num_labels, 1]
            attn = attn * self.beta                       # optional beta scaling
            attn = torch.nn.functional.softmax(attn, dim=1)                 # softmax over labels

            context = attn * value  # [b, num_labels, dim]

            label_rep = label_rep + context

        scores = self.mlp(label_rep).squeeze(-1)  # [b, num_labels]

        return scores

class CrossAttnScorer(nn.Module):
    def __init__(self, hidden_size, num_heads=16, attn_dropout=0.1, scorer_mlp_hidden_size=1024, use_sequence_packing=True, **kwargs):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        self.num_heads            = num_heads
        self.head_dim             = hidden_size // num_heads
        self.attn_dropout         = attn_dropout
        self.use_sequence_packing = use_sequence_packing

        self.q_norm  = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)

        self.q   = nn.Linear(hidden_size, hidden_size)
        self.k   = nn.Linear(hidden_size, hidden_size)
        self.v   = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, scorer_mlp_hidden_size),
            nn.GELU(),
            nn.Linear(scorer_mlp_hidden_size, scorer_mlp_hidden_size // 2),
            nn.GELU(),
            nn.Linear(scorer_mlp_hidden_size // 2, 1),
        )

    def _forward_packed(self, text_rep, label_rep, text_mask, batch_size, num_labels, hidden_size):
        """Unpadded path: flash_attn_varlen → PyTorch loop fallback."""
        label_mask = label_rep.abs().sum(dim=-1) > 0  # [B, N]

        text_unpad,  _, cu_seqlens_k, _            = unpad_input(text_rep,  text_mask)
        label_unpad, label_indices, cu_seqlens_q, _ = unpad_input(label_rep, label_mask)

        q = self.q(self.q_norm(label_unpad)).view(-1, self.num_heads, self.head_dim)
        k = self.k(self.kv_norm(text_unpad)).view(-1, self.num_heads, self.head_dim)
        v = self.v(text_unpad).view(-1, self.num_heads, self.head_dim)

        dropout_p = self.attn_dropout if self.training else 0.0
        context = attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p=dropout_p)
        context = self.norm(self.out(context.reshape(-1, hidden_size)))  # [Ql, H]

        scores_unpad = self.score_mlp(
            torch.cat([context, label_unpad], dim=-1)
        ).squeeze(-1)  # [Ql]

        return pad_input(
            scores_unpad.unsqueeze(-1), label_indices, batch_size, num_labels
        ).squeeze(-1)  # [B, N]

    def _forward_padded(self, text_rep, label_rep, text_mask, batch_size, num_labels, hidden_size):
        """Padded path: F.scaled_dot_product_attention (flash backend on CUDA)."""
        q = self.q(self.q_norm(label_rep)).view(batch_size, num_labels, self.num_heads, self.head_dim)
        k = self.k(self.kv_norm(text_rep)).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v(text_rep).view(batch_size, -1, self.num_heads, self.head_dim)

        dropout_p = self.attn_dropout if self.training else 0.0
        context = attn_padded(q, k, v, key_padding_mask=text_mask, dropout_p=dropout_p)
        # [B, N, nheads, head_dim] → [B, N, H]
        context = self.norm(self.out(context.reshape(batch_size, num_labels, hidden_size)))

        scores = self.score_mlp(
            torch.cat([context, label_rep], dim=-1)
        ).squeeze(-1)  # [B, N]

        return scores

    def forward(self, text_rep, label_rep, text_mask=None, **kwargs):
        batch_size, _, hidden_size = text_rep.shape
        num_labels = label_rep.shape[1]

        if text_mask is None:
            text_mask = torch.ones(batch_size, text_rep.shape[1],
                                   dtype=torch.bool, device=text_rep.device)

        if self.use_sequence_packing:
            return self._forward_packed(text_rep, label_rep, text_mask, batch_size, num_labels, hidden_size)
        else:
            return self._forward_padded(text_rep, label_rep, text_mask, batch_size, num_labels, hidden_size)


SCORER2OBJECT = {
    "weighted-dot": ScorerWeightedDot,
    "simple": ScorerDot,
    "mlp": MLPScorer,
    "hopfield": HopfieldScorer,
    "cross-attn": CrossAttnScorer
}
