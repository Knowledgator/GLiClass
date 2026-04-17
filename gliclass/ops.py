import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


# ─── Padding helpers ──────────────────────────────────────────────────────────

@torch.compiler.disable
def _get_unpad_data(attention_mask: torch.Tensor):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
    """
    Remove padding from a batch of sequences.

    Args:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen)  1 = real token, 0 = pad
    Returns:
        hidden_states_unpadded: (total_nnz, ...)
        indices:                (total_nnz,)
        cu_seqlens:             (batch + 1,) int32
        max_seqlen:             int
    """
    indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return flat[indices], indices, cu_seqlens, max_seqlen


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int):
    """
    Restore padding after unpadded processing.

    Args:
        hidden_states: (total_nnz, ...)
        indices:       (total_nnz,)
        batch:         int
        seqlen:        int (max seqlen)
    Returns:
        (batch, seqlen, ...)
    """
    output = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
    output = output.index_put((indices,), hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


# ─── Attention (varlen) ───────────────────────────────────────────────────────

def _attn_varlen_torch(q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale):
    """Pure PyTorch fallback for unpadded variable-length attention."""
    B = cu_seqlens_q.shape[0] - 1
    out = torch.zeros_like(q)

    for b in range(B):
        qs, qe = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        ks, ke = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        if qe == qs or ke == ks:
            continue
        attn = torch.einsum('qhd,khd->hqk', q[qs:qe], k[ks:ke]) * sm_scale
        attn = F.softmax(attn, dim=-1)
        out[qs:qe] = torch.einsum('hqk,khd->qhd', attn, v[ks:ke])

    return out


def attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sm_scale: float | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Unpadded attention with variable-length sequences.
    Priority: flash_attn → PyTorch loop.

    Args:
        q:            [total_q, nheads, head_dim]
        k:            [total_k, nheads, head_dim]
        v:            [total_k, nheads, head_dim]
        cu_seqlens_q: [B+1] int32
        cu_seqlens_k: [B+1] int32
        sm_scale:     defaults to 1/sqrt(head_dim)
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    if FLASH_ATTN_AVAILABLE and q.is_cuda:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
        max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max())
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=sm_scale,
            causal=False,
        )

    return _attn_varlen_torch(q, k, v, cu_seqlens_q, cu_seqlens_k, sm_scale)


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
        attn_mask = key_padding_mask[:, None, None, :]

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
    )
    return out.transpose(1, 2)  # [batch, nq, nheads, head_dim]
