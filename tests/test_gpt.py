import pytest
import torch
import torch.nn as nn

from utils.gpt import (
    Attention,
    FeedForward,
    GPT,
    GPTConfig,
    KVCache,
    TransformerBlock,
    find_multiple,
    multinomial_sample_one_no_sync,
    sample,
)


def _tiny_config(**overrides):
    defaults = dict(
        block_size=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        dim=16,
        intermediate_size=32,
        tokens_per_frame=4,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _patch_kv_cache_float32(monkeypatch):
    # gpt.py's default bfloat16 KV cache clashes with float32 Q tensors inside
    # scaled_dot_product_attention on CPU. Patch the default dtype to exercise
    # the full decode path in tests without modifying source.
    from utils import gpt as gpt_module

    original = gpt_module.KVCache.__init__

    def patched(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.float32):
        original(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=dtype)

    monkeypatch.setattr(gpt_module.KVCache, "__init__", patched)


@pytest.mark.unit
def test_find_multiple_rounds_up():
    """Rounds up to the next multiple of k."""
    assert find_multiple(10, 8) == 16


@pytest.mark.unit
def test_find_multiple_returns_same_if_multiple():
    """Returns n unchanged when n is already a multiple of k."""
    assert find_multiple(16, 8) == 16


@pytest.mark.unit
def test_find_multiple_small_values():
    """Handles n below k."""
    assert find_multiple(3, 4) == 4


@pytest.mark.unit
def test_find_multiple_k_one():
    """k=1 is the identity."""
    assert find_multiple(7, 1) == 7


@pytest.mark.unit
def test_find_multiple_zero():
    """Zero is a multiple of any k."""
    assert find_multiple(0, 8) == 0


@pytest.mark.unit
def test_find_multiple_large_number():
    """Rounds up for model-sized inputs."""
    assert find_multiple(1025, 8) == 1032


@pytest.mark.unit
def test_multinomial_sample_one_no_sync_returns_int_with_trailing_dim():
    """Returns an int tensor with a trailing singleton dimension."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

    out = multinomial_sample_one_no_sync(probs)

    assert out.dtype == torch.int
    assert out.shape == (1,)


@pytest.mark.unit
def test_multinomial_sample_one_no_sync_one_hot_is_deterministic():
    """One-hot input samples that exact index."""
    probs = torch.tensor([0.0, 0.0, 1.0, 0.0])

    out = multinomial_sample_one_no_sync(probs)

    assert out.item() == 2


@pytest.mark.unit
def test_multinomial_sample_one_no_sync_index_in_bounds():
    """Sampled indices stay in the distribution's support."""
    torch.manual_seed(42)
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

    for _ in range(20):
        out = multinomial_sample_one_no_sync(probs)
        assert 0 <= out.item() < probs.shape[-1]


@pytest.mark.unit
def test_multinomial_sample_one_no_sync_batched_shape():
    """Preserves the batch dimension."""
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    out = multinomial_sample_one_no_sync(probs)

    assert out.shape == (2, 1)
    assert out[0].item() == 0
    assert out[1].item() == 2


@pytest.mark.unit
def test_sample_returns_token_and_probabilities():
    """Returns a (next-token-index, probability-distribution) pair."""
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    idx, probs = sample(logits)

    assert idx.dtype == torch.int
    assert idx.shape == (1,)
    assert probs.shape == (3,)
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


@pytest.mark.unit
def test_sample_probabilities_use_last_position():
    """Softmax is applied to the last-position logits only."""
    logits = torch.tensor([[[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]]])

    _, probs = sample(logits)

    assert probs.argmax().item() == 1


@pytest.mark.unit
def test_sample_selects_dominating_index():
    """A dominating last-position logit forces that index."""
    logits = torch.zeros(1, 2, 4)
    logits[0, -1, 2] = 1e4

    idx, _ = sample(logits)

    assert idx.item() == 2


@pytest.mark.unit
def test_gpt_config_default_values():
    """Default fields match the GPT-2-medium-style values."""
    config = GPTConfig()

    assert config.block_size == 20 * 129
    assert config.vocab_size == 1025
    assert config.n_layer == 24
    assert config.n_head == 16
    assert config.dim == 1024
    assert config.intermediate_size == 4 * 1024
    assert config.tokens_per_frame == 129


@pytest.mark.unit
def test_gpt_config_bos_token_uses_vocab_size_minus_one():
    """bos_token is vocab_size - 1."""
    config = GPTConfig()

    assert config.bos_token == config.vocab_size - 1
    assert config.bos_token == 1024


@pytest.mark.unit
def test_gpt_config_bos_token_updates_when_vocab_size_changes():
    """bos_token tracks vocab_size overrides."""
    config = GPTConfig(vocab_size=2048)

    assert config.bos_token == 2047


@pytest.mark.unit
def test_gpt_config_head_dim_computes_dim_divided_by_n_head():
    """head_dim is dim // n_head."""
    config = GPTConfig()

    assert config.head_dim == config.dim // config.n_head
    assert config.head_dim == 64


@pytest.mark.unit
def test_gpt_config_head_dim_updates_with_custom_dim_and_n_head():
    """head_dim tracks dim / n_head overrides."""
    config = GPTConfig(dim=768, n_head=12)
    assert config.head_dim == 64

    config = GPTConfig(dim=512, n_head=8)
    assert config.head_dim == 64


@pytest.mark.unit
def test_gpt_config_allows_custom_field_overrides():
    """All fields are overridable via the constructor."""
    config = GPTConfig(
        block_size=256,
        vocab_size=5000,
        n_layer=6,
        n_head=8,
        dim=512,
        intermediate_size=2048,
        tokens_per_frame=64,
    )

    assert config.block_size == 256
    assert config.vocab_size == 5000
    assert config.n_layer == 6
    assert config.n_head == 8
    assert config.dim == 512
    assert config.intermediate_size == 2048
    assert config.tokens_per_frame == 64


@pytest.mark.unit
def test_gpt_config_head_dim_is_integer():
    """head_dim is an int, not a float."""
    config = GPTConfig()

    assert isinstance(config.head_dim, int)


@pytest.mark.unit
def test_kv_cache_initializes_caches_with_expected_shape():
    """Caches have shape (B, H, T, D)."""
    cache = KVCache(max_batch_size=2, max_seq_length=5, n_heads=3, head_dim=4)

    assert cache.k_cache.shape == (2, 3, 5, 4)
    assert cache.v_cache.shape == (2, 3, 5, 4)


@pytest.mark.unit
def test_kv_cache_initializes_caches_with_zeros():
    """Both caches start zero-filled."""
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=2, head_dim=3)

    assert torch.count_nonzero(cache.k_cache) == 0
    assert torch.count_nonzero(cache.v_cache) == 0


@pytest.mark.unit
def test_kv_cache_uses_bfloat16_by_default():
    """Default dtype is bfloat16."""
    cache = KVCache(max_batch_size=1, max_seq_length=2, n_heads=1, head_dim=2)

    assert cache.k_cache.dtype == torch.bfloat16
    assert cache.v_cache.dtype == torch.bfloat16


@pytest.mark.unit
def test_kv_cache_accepts_custom_dtype():
    """Caller dtype propagates to both buffers."""
    cache = KVCache(max_batch_size=1, max_seq_length=2, n_heads=1, head_dim=2, dtype=torch.float32)

    assert cache.k_cache.dtype == torch.float32
    assert cache.v_cache.dtype == torch.float32


@pytest.mark.unit
def test_kv_cache_update_writes_values_at_requested_positions():
    """update() writes at the requested input positions."""
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=2, head_dim=3, dtype=torch.float32)
    input_pos = torch.tensor([1, 3], dtype=torch.int64)
    k_val = torch.arange(12, dtype=torch.float32).view(1, 2, 2, 3)
    v_val = (torch.arange(12, dtype=torch.float32) + 100).view(1, 2, 2, 3)

    k_out, v_out = cache.update(input_pos, k_val, v_val)

    torch.testing.assert_close(k_out[:, :, input_pos], k_val)
    torch.testing.assert_close(v_out[:, :, input_pos], v_val)


@pytest.mark.unit
def test_kv_cache_update_preserves_unwritten_positions():
    """update() leaves other positions untouched."""
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


@pytest.mark.unit
def test_kv_cache_update_returns_internal_cache_tensors():
    """update() returns the module's buffers, not copies."""
    cache = KVCache(max_batch_size=1, max_seq_length=3, n_heads=1, head_dim=2, dtype=torch.float32)
    input_pos = torch.tensor([0], dtype=torch.int64)
    k_val = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)
    v_val = torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float32)

    k_out, v_out = cache.update(input_pos, k_val, v_val)

    assert k_out is cache.k_cache
    assert v_out is cache.v_cache


@pytest.mark.unit
def test_kv_cache_update_rejects_mismatched_input_positions():
    """update() asserts input_pos length matches k_val's sequence dim."""
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=1, head_dim=2, dtype=torch.float32)
    input_pos = torch.tensor([0, 1], dtype=torch.int64)
    k_val = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    v_val = torch.zeros((1, 1, 1, 2), dtype=torch.float32)

    with pytest.raises(AssertionError):
        cache.update(input_pos, k_val, v_val)


@pytest.mark.unit
def test_feedforward_preserves_input_shape():
    """Preserves (B, T, dim) shape."""
    torch.manual_seed(42)
    config = _tiny_config()
    ff = FeedForward(config)
    x = torch.randn(2, 3, config.dim)

    y = ff(x)

    assert y.shape == x.shape


@pytest.mark.unit
def test_feedforward_intermediate_projection_widths():
    """c_fc widens to intermediate_size; c_proj projects back to dim."""
    config = _tiny_config()
    ff = FeedForward(config)

    assert ff.c_fc.in_features == config.dim
    assert ff.c_fc.out_features == config.intermediate_size
    assert ff.c_proj.in_features == config.intermediate_size
    assert ff.c_proj.out_features == config.dim


@pytest.mark.unit
def test_attention_forward_without_cache_preserves_shape():
    """Preserves shape without a KV cache (fresh-mask path)."""
    torch.manual_seed(42)
    config = _tiny_config()
    attn = Attention(config)
    bsz, seqlen = 2, 4
    x = torch.randn(bsz, seqlen, config.dim)
    mask = torch.tril(torch.ones(1, 1, seqlen, seqlen, dtype=torch.bool))

    y = attn(x, mask=mask, input_pos=None)

    assert y.shape == x.shape


@pytest.mark.unit
def test_attention_rejects_dim_not_divisible_by_n_head():
    """Rejects dim not divisible by n_head."""
    bad = GPTConfig(block_size=8, vocab_size=8, n_layer=1, n_head=3, dim=16,
                    intermediate_size=8, tokens_per_frame=2)
    with pytest.raises(AssertionError):
        Attention(bad)


@pytest.mark.unit
def test_attention_c_attn_is_packed_qkv_projection():
    """c_attn packs Q, K, V into a 3*dim projection."""
    config = _tiny_config()
    attn = Attention(config)

    assert attn.c_attn.in_features == config.dim
    assert attn.c_attn.out_features == 3 * config.dim
    assert attn.c_proj.in_features == config.dim
    assert attn.c_proj.out_features == config.dim


@pytest.mark.unit
def test_attention_kv_cache_defaults_to_none():
    """kv_cache starts as None."""
    attn = Attention(_tiny_config())

    assert attn.kv_cache is None


@pytest.mark.unit
def test_transformer_block_forward_preserves_shape():
    """Preserves (B, T, dim) shape."""
    torch.manual_seed(42)
    config = _tiny_config()
    block = TransformerBlock(config)
    bsz, seqlen = 1, 5
    x = torch.randn(bsz, seqlen, config.dim)
    mask = torch.tril(torch.ones(1, 1, seqlen, seqlen, dtype=torch.bool))
    input_pos = torch.arange(seqlen)

    y = block(x, input_pos=input_pos, mask=mask)

    assert y.shape == x.shape


@pytest.mark.unit
def test_transformer_block_residual_connection_is_wired():
    """Block collapses to identity when both sublayer outputs are zeroed."""
    torch.manual_seed(42)
    config = _tiny_config()
    block = TransformerBlock(config)
    nn.init.zeros_(block.attn.c_proj.weight)
    nn.init.zeros_(block.attn.c_proj.bias)
    nn.init.zeros_(block.mlp.c_proj.weight)
    nn.init.zeros_(block.mlp.c_proj.bias)

    x = torch.randn(1, 4, config.dim)
    mask = torch.tril(torch.ones(1, 1, 4, 4, dtype=torch.bool))
    input_pos = torch.arange(4)

    y = block(x, input_pos=input_pos, mask=mask)

    torch.testing.assert_close(y, x)


@pytest.mark.unit
def test_gpt_init_builds_expected_module_tree():
    """__init__ wires wte / wpe / h / ln_f / lm_head with config-derived shapes."""
    config = _tiny_config()
    model = GPT(config)

    assert isinstance(model.transformer["wte"], nn.Embedding)
    assert model.transformer["wte"].num_embeddings == config.vocab_size
    assert model.transformer["wte"].embedding_dim == config.dim
    assert isinstance(model.transformer["wpe"], nn.Embedding)
    assert model.transformer["wpe"].num_embeddings == config.block_size
    assert len(model.transformer["h"]) == config.n_layer
    assert isinstance(model.transformer["ln_f"], nn.LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.out_features == config.vocab_size
    assert model.max_batch_size == -1
    assert model.max_seq_length == -1
    assert all(block.attn.kv_cache is None for block in model.transformer["h"])


@pytest.mark.unit
def test_gpt_init_causal_mask_is_lower_triangular():
    """causal_mask is a block_size × block_size lower-triangular bool tensor."""
    config = _tiny_config()
    model = GPT(config)

    assert model.causal_mask.shape == (1, 1, config.block_size, config.block_size)
    assert model.causal_mask.dtype == torch.bool
    expected = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
    torch.testing.assert_close(model.causal_mask[0, 0], expected)


@pytest.mark.unit
def test_gpt_setup_caches_allocates_kv_cache_per_block():
    """setup_caches allocates a KVCache per block, rounded to a multiple of 8."""
    config = _tiny_config()
    model = GPT(config)

    model.setup_caches(max_batch_size=2, max_seq_length=10)

    assert model.max_batch_size == 2
    assert model.max_seq_length == 16  # find_multiple(10, 8)
    for block in model.transformer["h"]:
        assert isinstance(block.attn.kv_cache, KVCache)
        assert block.attn.kv_cache.k_cache.shape == (2, config.n_head, 16, config.head_dim)
    assert model.causal_mask.shape == (1, 1, 16, 16)


@pytest.mark.unit
def test_gpt_setup_caches_is_noop_when_already_large_enough():
    """setup_caches is a no-op when existing caches are large enough."""
    config = _tiny_config()
    model = GPT(config)

    model.setup_caches(max_batch_size=2, max_seq_length=16)
    first_cache = model.transformer["h"][0].attn.kv_cache

    model.setup_caches(max_batch_size=1, max_seq_length=8)

    assert model.transformer["h"][0].attn.kv_cache is first_cache
    assert model.max_batch_size == 2
    assert model.max_seq_length == 16


@pytest.mark.integration
def test_gpt_forward_without_input_pos_returns_logits():
    """forward without input_pos returns (B, T, vocab_size) logits."""
    torch.manual_seed(42)
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 5))

    logits = model(idx)

    assert logits.shape == (1, 5, config.vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.integration
def test_gpt_forward_with_explicit_input_pos_returns_logits():
    """forward with explicit input_pos returns correctly shaped logits."""
    torch.manual_seed(42)
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))
    input_pos = torch.tensor([0, 1, 2])

    logits = model(idx, input_pos=input_pos)

    assert logits.shape == (1, 3, config.vocab_size)


@pytest.mark.integration
def test_gpt_prefill_returns_single_next_token():
    """prefill returns a single sampled next-token tensor."""
    torch.manual_seed(42)
    config = _tiny_config()
    model = GPT(config)
    prompt = torch.randint(0, config.vocab_size, (4,))
    input_pos = torch.arange(4)

    next_token = model.prefill(prompt.view(1, -1), input_pos)

    assert next_token.shape == (1,)
    assert next_token.dtype == torch.int
    assert 0 <= next_token.item() < config.vocab_size


@pytest.mark.integration
def test_gpt_decode_one_token_requires_single_position(monkeypatch):
    """decode_one_token asserts input_pos has exactly one position."""
    _patch_kv_cache_float32(monkeypatch)
    config = _tiny_config()
    model = GPT(config)
    model.setup_caches(max_batch_size=1, max_seq_length=config.block_size)
    cur_token = torch.tensor([[3]])
    bad_pos = torch.tensor([0, 1])

    with pytest.raises(AssertionError):
        model.decode_one_token(cur_token, bad_pos)


@pytest.mark.system
def test_gpt_generate_produces_expected_length_and_in_vocab_tokens(monkeypatch):
    """generate returns max_new_tokens in-vocab tokens."""
    _patch_kv_cache_float32(monkeypatch)
    torch.manual_seed(42)
    config = _tiny_config()
    model = GPT(config)
    prompt = torch.randint(0, config.vocab_size, (4,), dtype=torch.int)
    max_new_tokens = 3

    out = model.generate(prompt, max_new_tokens=max_new_tokens)

    assert out.shape == (max_new_tokens,)
    assert out.dtype == prompt.dtype
    assert torch.all(out >= 0)
    assert torch.all(out < config.vocab_size)


@pytest.mark.integration
def test_gpt_decode_n_tokens_returns_expected_list_lengths(monkeypatch):
    """decode_n_tokens returns num_new_tokens tokens/probs and advances input_pos in place."""
    _patch_kv_cache_float32(monkeypatch)
    torch.manual_seed(42)
    config = _tiny_config()
    model = GPT(config)
    model.setup_caches(max_batch_size=1, max_seq_length=config.block_size)

    prompt = torch.randint(0, config.vocab_size, (4,), dtype=torch.int)
    prefill_pos = torch.arange(4)
    next_token = model.prefill(prompt.view(1, -1), prefill_pos)
    input_pos = torch.tensor([4], dtype=torch.int)

    new_tokens, new_probs = model.decode_n_tokens(next_token.view(1, -1), input_pos, num_new_tokens=3)

    assert len(new_tokens) == 3
    assert len(new_probs) == 3
    for tok in new_tokens:
        assert 0 <= tok.item() < config.vocab_size
    assert input_pos.item() == 4 + 3


@pytest.mark.unit
def test_gpt_load_state_dict_from_url_loads_transposed_weights(monkeypatch):
    """load_state_dict_from_url filters legacy bias keys, transposes the four packed weights, and injects a fresh causal_mask."""
    torch.manual_seed(42)
    config = _tiny_config()

    reference = GPT(config)
    reference_state = {k: v.clone() for k, v in reference.state_dict().items()}

    # pre-transpose the four packed weights so the method's transpose restores
    # them; drop causal_mask (method re-creates); add dummy legacy keys so the
    # filter branch (.attn.bias / .attn.masked_bias) is exercised
    transposed_suffixes = ("attn.c_attn.weight", "attn.c_proj.weight",
                           "mlp.c_fc.weight", "mlp.c_proj.weight")
    fake_download = {}
    for k, v in reference_state.items():
        if k == "causal_mask":
            continue
        if any(k.endswith(s) for s in transposed_suffixes):
            fake_download[k] = torch.transpose(v, 1, 0).clone()
        else:
            fake_download[k] = v.clone()
    fake_download["transformer.h.0.attn.masked_bias"] = torch.tensor([999.0])
    fake_download["transformer.h.0.attn.bias"] = torch.ones(1, 1, config.block_size, config.block_size)

    def fake_hub_loader(*_args, **_kwargs):
        return {k: v.clone() for k, v in fake_download.items()}

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_hub_loader)

    target = GPT(config)
    target.load_state_dict_from_url(url="http://ignored.invalid/model.bin")

    for k, v in reference_state.items():
        torch.testing.assert_close(dict(target.state_dict())[k], v)
