"""Smoke tests for the TinyGPT reference module."""

import torch

from tiny_gpt import (
    MultiHeadAttention,
    TinyGPT,
    TokenEmbedder,
    TransformerBlock,
    scaled_dot_product_attention,
)


def test_token_embedder_shape():
    emb = TokenEmbedder(vocab_size=100, d_model=32, max_seq_len=16)
    ids = torch.randint(0, 100, (2, 8))
    out = emb(ids)
    assert out.shape == (2, 8, 32)


def test_scaled_dot_product_attention_shape():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == v.shape


def test_scaled_dot_product_attention_mask():
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    mask = torch.tril(torch.ones(3, 3)).view(1, 1, 3, 3)
    out = scaled_dot_product_attention(q, k, v, mask=mask)
    assert out.shape == v.shape


def test_multihead_attention_shape():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 8, 32)
    out = mha(x)
    assert out.shape == x.shape


def test_transformer_block_shape():
    block = TransformerBlock(d_model=32, n_heads=4)
    x = torch.randn(2, 8, 32)
    out = block(x)
    assert out.shape == x.shape


def test_tinygpt_forward_shape():
    model = TinyGPT(vocab_size=100, d_model=32, n_heads=4, n_layers=2, max_seq_len=16)
    idx = torch.randint(0, 100, (2, 8))
    logits = model(idx)
    assert logits.shape == (2, 8, 100)


def test_tinygpt_generate_extends_sequence():
    model = TinyGPT(vocab_size=100, d_model=32, n_heads=4, n_layers=2, max_seq_len=16)
    idx = torch.randint(0, 100, (1, 4))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_tinygpt_tied_weights():
    model = TinyGPT(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    assert model.head.weight.data_ptr() == model.embedder.token.weight.data_ptr()


if __name__ == "__main__":
    test_token_embedder_shape()
    test_scaled_dot_product_attention_shape()
    test_scaled_dot_product_attention_mask()
    test_multihead_attention_shape()
    test_transformer_block_shape()
    test_tinygpt_forward_shape()
    test_tinygpt_generate_extends_sequence()
    test_tinygpt_tied_weights()
    print("All tests passed.")
