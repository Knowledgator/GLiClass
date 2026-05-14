import torch
import torch.nn.functional as F

# ─── Attention (padded) ───────────────────────────────────────────────────────


def attn_padded(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Padded attention via F.scaled_dot_product_attention.
    Uses FlashAttention backend automatically on CUDA when available.

    Args:
        q:                [batch, nq, nheads, head_dim]
        k:                [batch, nk, nheads, head_dim]
        v:                [batch, nk, nheads, head_dim]
        key_padding_mask: [batch, nk] bool, True = real token
    Returns:
        [batch, nq, nheads, head_dim]
    """
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn_mask = None
    if key_padding_mask is not None:
        attn_mask = key_padding_mask[:, None, None, :].bool()

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
    )
    return out.transpose(1, 2)  # [batch, nq, nheads, head_dim]
