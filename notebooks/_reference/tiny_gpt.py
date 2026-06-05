"""Reference implementation of the TinyGPT architecture.

Mirrors the deep build of session1-attention-tinygpt. Imported by
session2-pretraining-lab so the training loop runs even if the student
did not complete Session 1.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TokenEmbedder(nn.Module):
    """Token embedding plus sinusoidal positional encoding."""

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "pos", _sinusoidal_positional_encoding(max_seq_len, d_model)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, t = ids.shape
        return self.token(ids) + self.pos[:t]


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard scaled dot-product attention.

    Shapes:
        q, k, v: (..., T, d_k)
        mask:    broadcastable to (..., T_q, T_k). 0 means masked.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, c = x.shape
        qkv = (
            self.qkv(x)
            .view(b, t, 3, self.n_heads, self.d_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedder = TokenEmbedder(vocab_size, d_model, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie input and output embedding weights.
        self.head.weight = self.embedder.token.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        mask = torch.tril(torch.ones(t, t, device=idx.device)).view(1, 1, t, t)
        x = self.embedder(idx)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len :]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
