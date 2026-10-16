"""Patch correctness on a tiny random Llama (no download)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM

from src.rope.patch import RopePatcher, band_transform, identity, zero_all


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def logits(m):
    ids = torch.arange(100).unsqueeze(0) % 128
    return m(input_ids=ids).logits


def test_identity_exact(model):
    p = RopePatcher(model)
    ref = logits(model)
    p.apply(identity)
    assert torch.equal(logits(model), ref)
    p.restore()


def test_band_patch_changes_and_restores(model):
    p = RopePatcher(model)
    ref = logits(model)
    p.apply(band_transform(16, 4, 0, "compress", 8.0))
    assert (logits(model) - ref).abs().max() > 1e-4
    p.restore()
    assert torch.equal(logits(model), ref) and p.check_pristine()


def test_zero_all_changes(model):
    p = RopePatcher(model)
    ref = logits(model)
    p.apply(zero_all)
    assert (logits(model) - ref).abs().max() > 1e-4
    p.restore()
