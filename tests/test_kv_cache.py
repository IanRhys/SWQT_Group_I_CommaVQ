"""
FILE: test file for `KVCache` in `gpt.py`.

HOW-TO-RUN (in root):
    python3 -m pytest tests/test_kv_cache.py
"""

import pytest
import torch

from utils.gpt import KVCache


# verifies that KVCache creates key and value caches with the expected shape
def test_kv_cache_initializes_caches_with_expected_shape():
    cache = KVCache(max_batch_size=2, max_seq_length=5, n_heads=3, head_dim=4)

    assert cache.k_cache.shape == (2, 3, 5, 4)
    assert cache.v_cache.shape == (2, 3, 5, 4)


# verifies that both caches start filled with zeros
def test_kv_cache_initializes_caches_with_zeros():
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=2, head_dim=3)

    assert torch.count_nonzero(cache.k_cache) == 0
    assert torch.count_nonzero(cache.v_cache) == 0


# verifies that the default cache dtype matches the implementation default
def test_kv_cache_uses_bfloat16_by_default():
    cache = KVCache(max_batch_size=1, max_seq_length=2, n_heads=1, head_dim=2)

    assert cache.k_cache.dtype == torch.bfloat16
    assert cache.v_cache.dtype == torch.bfloat16


# verifies that a custom dtype is applied to both registered cache tensors
def test_kv_cache_accepts_custom_dtype():
    cache = KVCache(max_batch_size=1, max_seq_length=2, n_heads=1, head_dim=2, dtype=torch.float32)

    assert cache.k_cache.dtype == torch.float32
    assert cache.v_cache.dtype == torch.float32


# verifies that update writes key and value tensors into the requested cache positions
def test_kv_cache_update_writes_values_at_requested_positions():
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=2, head_dim=3, dtype=torch.float32)
    input_pos = torch.tensor([1, 3], dtype=torch.int64)
    k_val = torch.arange(12, dtype=torch.float32).view(1, 2, 2, 3)
    v_val = (torch.arange(12, dtype=torch.float32) + 100).view(1, 2, 2, 3)

    k_out, v_out = cache.update(input_pos, k_val, v_val)

    torch.testing.assert_close(k_out[:, :, input_pos], k_val)
    torch.testing.assert_close(v_out[:, :, input_pos], v_val)


# verifies that update leaves untouched cache positions unchanged
def test_kv_cache_update_preserves_unwritten_positions():
    cache = KVCache(max_batch_size=1, max_seq_length=5, n_heads=1, head_dim=2, dtype=torch.float32)
    input_pos = torch.tensor([2], dtype=torch.int64)
    k_val = torch.tensor([[[[1.5, 2.5]]]], dtype=torch.float32)
    v_val = torch.tensor([[[[3.5, 4.5]]]], dtype=torch.float32)

    cache.update(input_pos, k_val, v_val)

    expected_k = torch.zeros((1, 1, 5, 2), dtype=torch.float32)
    expected_v = torch.zeros((1, 1, 5, 2), dtype=torch.float32)
    expected_k[:, :, 2] = k_val
    expected_v[:, :, 2] = v_val

    torch.testing.assert_close(cache.k_cache, expected_k)
    torch.testing.assert_close(cache.v_cache, expected_v)


# verifies that update returns the same underlying cache tensors stored on the module
def test_kv_cache_update_returns_internal_cache_tensors():
    cache = KVCache(max_batch_size=1, max_seq_length=3, n_heads=1, head_dim=2, dtype=torch.float32)
    input_pos = torch.tensor([0], dtype=torch.int64)
    k_val = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)
    v_val = torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float32)

    k_out, v_out = cache.update(input_pos, k_val, v_val)

    assert k_out is cache.k_cache
    assert v_out is cache.v_cache


# verifies that update rejects inputs when the number of positions does not match the sequence length
def test_kv_cache_update_rejects_mismatched_input_positions():
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=1, head_dim=2, dtype=torch.float32)
    input_pos = torch.tensor([0, 1], dtype=torch.int64)
    k_val = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    v_val = torch.zeros((1, 1, 1, 2), dtype=torch.float32)

    with pytest.raises(AssertionError):
        cache.update(input_pos, k_val, v_val)
