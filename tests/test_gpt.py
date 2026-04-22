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
    """Verifies a non-multiple is rounded up to the next multiple of k."""
    assert find_multiple(10, 8) == 16


@pytest.mark.unit
def test_find_multiple_returns_same_if_multiple():
    """Verifies an input that's already a multiple of k is returned unchanged."""
    assert find_multiple(16, 8) == 16


@pytest.mark.unit
def test_find_multiple_small_values():
    """Verifies an input smaller than k still rounds up to k."""
    assert find_multiple(3, 4) == 4


@pytest.mark.unit
def test_find_multiple_k_one():
    """Verifies k=1 leaves the input unchanged, since every integer is a multiple of 1."""
    assert find_multiple(7, 1) == 7


@pytest.mark.unit
def test_find_multiple_zero():
    """Verifies zero is treated as a multiple of any k."""
    assert find_multiple(0, 8) == 0


@pytest.mark.unit
def test_find_multiple_large_number():
    """Verifies rounding behaves correctly at realistic model-sized inputs."""
    assert find_multiple(1025, 8) == 1032


@pytest.mark.unit
def test_multinomial_sample_one_no_sync_returns_int_with_trailing_dim():
    """Verifies the sampler returns an int tensor with a trailing singleton dimension."""
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])

    out = multinomial_sample_one_no_sync(probs)

    assert out.dtype == torch.int
    assert out.shape == (1,)


@pytest.mark.unit
def test_multinomial_sample_one_no_sync_one_hot_is_deterministic():
    """Verifies a one-hot probability vector always samples that exact index."""
    probs = torch.tensor([0.0, 0.0, 1.0, 0.0])

    out = multinomial_sample_one_no_sync(probs)

    assert out.item() == 2


@pytest.mark.unit
def test_multinomial_sample_one_no_sync_index_in_bounds():
    """Verifies sampled indices always fall within the distribution's support."""
    torch.manual_seed(42)
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

    for _ in range(20):
        out = multinomial_sample_one_no_sync(probs)
        assert 0 <= out.item() < probs.shape[-1]


@pytest.mark.unit
def test_multinomial_sample_one_no_sync_batched_shape():
    """Verifies batched one-hot inputs resolve to the correct index per row and preserve the batch dimension."""
    probs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    out = multinomial_sample_one_no_sync(probs)

    assert out.shape == (2, 1)
    assert out[0].item() == 0
    assert out[1].item() == 2


@pytest.mark.unit
def test_sample_returns_token_and_probabilities():
    """Verifies sample returns the next-token index together with the full probability distribution for the last position."""
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    idx, probs = sample(logits)

    assert idx.dtype == torch.int
    assert idx.shape == (1,)
    assert probs.shape == (3,)
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


@pytest.mark.unit
def test_sample_probabilities_use_last_position():
    """Verifies softmax is applied only to the last timestep's logits."""
    logits = torch.tensor([[[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]]])

    _, probs = sample(logits)

    assert probs.argmax().item() == 1


@pytest.mark.unit
def test_sample_selects_dominating_index():
    """Verifies a logit dominating the last position forces that index to be sampled."""
    logits = torch.zeros(1, 2, 4)
    logits[0, -1, 2] = 1e4

    idx, _ = sample(logits)

    assert idx.item() == 2


@pytest.mark.unit
def test_gpt_config_default_values():
    """Verifies the default GPTConfig fields match the GPT-2-medium-style values hard-coded in the source."""
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
    """Verifies bos_token is derived from vocab_size as the last vocabulary index."""
    config = GPTConfig()

    assert config.bos_token == config.vocab_size - 1
    assert config.bos_token == 1024


@pytest.mark.unit
def test_gpt_config_bos_token_updates_when_vocab_size_changes():
    """Verifies bos_token updates when vocab_size is overridden, proving it's not hard-coded."""
    config = GPTConfig(vocab_size=2048)

    assert config.bos_token == 2047


@pytest.mark.unit
def test_gpt_config_head_dim_computes_dim_divided_by_n_head():
    """Verifies head_dim is computed as the embedding dim divided by the number of heads."""
    config = GPTConfig()

    assert config.head_dim == config.dim // config.n_head
    assert config.head_dim == 64


@pytest.mark.unit
def test_gpt_config_head_dim_updates_with_custom_dim_and_n_head():
    """Verifies head_dim updates correctly when dim and n_head are customized."""
    config = GPTConfig(dim=768, n_head=12)
    assert config.head_dim == 64

    config = GPTConfig(dim=512, n_head=8)
    assert config.head_dim == 64


@pytest.mark.unit
def test_gpt_config_allows_custom_field_overrides():
    """Verifies every GPTConfig field can be overridden through the constructor."""
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
    """Verifies head_dim is returned as an integer, not a float."""
    config = GPTConfig()

    assert isinstance(config.head_dim, int)


@pytest.mark.unit
def test_kv_cache_initializes_caches_with_expected_shape():
    """Verifies the key and value caches are allocated with the expected (batch, heads, seq_len, head_dim) shape."""
    cache = KVCache(max_batch_size=2, max_seq_length=5, n_heads=3, head_dim=4)

    assert cache.k_cache.shape == (2, 3, 5, 4)
    assert cache.v_cache.shape == (2, 3, 5, 4)


@pytest.mark.unit
def test_kv_cache_initializes_caches_with_zeros():
    """Verifies both caches start filled with zeros."""
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=2, head_dim=3)

    assert torch.count_nonzero(cache.k_cache) == 0
    assert torch.count_nonzero(cache.v_cache) == 0


@pytest.mark.unit
def test_kv_cache_uses_bfloat16_by_default():
    """Verifies the caches default to bfloat16 when no dtype is specified."""
    cache = KVCache(max_batch_size=1, max_seq_length=2, n_heads=1, head_dim=2)

    assert cache.k_cache.dtype == torch.bfloat16
    assert cache.v_cache.dtype == torch.bfloat16


@pytest.mark.unit
def test_kv_cache_accepts_custom_dtype():
    """Verifies a caller-provided dtype propagates to both the key and value caches."""
    cache = KVCache(max_batch_size=1, max_seq_length=2, n_heads=1, head_dim=2, dtype=torch.float32)

    assert cache.k_cache.dtype == torch.float32
    assert cache.v_cache.dtype == torch.float32


@pytest.mark.unit
def test_kv_cache_update_writes_values_at_requested_positions():
    """Verifies update writes the given key and value tensors at the specified input positions."""
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=2, head_dim=3, dtype=torch.float32)
    input_pos = torch.tensor([1, 3], dtype=torch.int64)
    k_val = torch.arange(12, dtype=torch.float32).view(1, 2, 2, 3)
    v_val = (torch.arange(12, dtype=torch.float32) + 100).view(1, 2, 2, 3)

    k_out, v_out = cache.update(input_pos, k_val, v_val)

    torch.testing.assert_close(k_out[:, :, input_pos], k_val)
    torch.testing.assert_close(v_out[:, :, input_pos], v_val)


@pytest.mark.unit
def test_kv_cache_update_preserves_unwritten_positions():
    """Verifies update does not modify cache positions outside the given input positions."""
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
    """Verifies update returns the module's registered buffers directly, not copies."""
    cache = KVCache(max_batch_size=1, max_seq_length=3, n_heads=1, head_dim=2, dtype=torch.float32)
    input_pos = torch.tensor([0], dtype=torch.int64)
    k_val = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)
    v_val = torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float32)

    k_out, v_out = cache.update(input_pos, k_val, v_val)

    assert k_out is cache.k_cache
    assert v_out is cache.v_cache


@pytest.mark.unit
def test_kv_cache_update_rejects_mismatched_input_positions():
    """Verifies update raises when input_pos doesn't match the sequence length of the incoming tensors."""
    cache = KVCache(max_batch_size=1, max_seq_length=4, n_heads=1, head_dim=2, dtype=torch.float32)
    input_pos = torch.tensor([0, 1], dtype=torch.int64)
    k_val = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    v_val = torch.zeros((1, 1, 1, 2), dtype=torch.float32)

    with pytest.raises(AssertionError):
        cache.update(input_pos, k_val, v_val)


@pytest.mark.unit
def test_feedforward_preserves_input_shape():
    """Verifies the feed-forward block preserves the (batch, seq_len, dim) shape of its input."""
    torch.manual_seed(42)
    config = _tiny_config()
    ff = FeedForward(config)
    x = torch.randn(2, 3, config.dim)

    y = ff(x)

    assert y.shape == x.shape


@pytest.mark.unit
def test_feedforward_intermediate_projection_widths():
    """Verifies c_fc widens the input to intermediate_size and c_proj projects it back to dim."""
    config = _tiny_config()
    ff = FeedForward(config)

    assert ff.c_fc.in_features == config.dim
    assert ff.c_fc.out_features == config.intermediate_size
    assert ff.c_proj.in_features == config.intermediate_size
    assert ff.c_proj.out_features == config.dim


@pytest.mark.unit
def test_attention_forward_without_cache_preserves_shape():
    """Verifies attention preserves input shape when no KV cache is attached (the fresh-mask path)."""
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
    """Verifies the constructor raises when dim is not divisible by the number of heads."""
    bad = GPTConfig(block_size=8, vocab_size=8, n_layer=1, n_head=3, dim=16,
                    intermediate_size=8, tokens_per_frame=2)
    with pytest.raises(AssertionError):
        Attention(bad)


@pytest.mark.unit
def test_attention_c_attn_is_packed_qkv_projection():
    """Verifies c_attn is a single linear layer that packs the query, key, and value projections."""
    config = _tiny_config()
    attn = Attention(config)

    assert attn.c_attn.in_features == config.dim
    assert attn.c_attn.out_features == 3 * config.dim
    assert attn.c_proj.in_features == config.dim
    assert attn.c_proj.out_features == config.dim


@pytest.mark.unit
def test_attention_kv_cache_defaults_to_none():
    """Verifies kv_cache starts unset and is only attached once setup_caches runs."""
    attn = Attention(_tiny_config())

    assert attn.kv_cache is None


@pytest.mark.unit
def test_transformer_block_forward_preserves_shape():
    """Verifies a full transformer block preserves the (batch, seq_len, dim) shape across attention, feed-forward, and residual connections."""
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
    """Verifies the residual connections are wired: zeroing both sublayers' output projections collapses the block to the identity."""
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
    """Verifies the GPT constructor builds the expected submodule tree with shapes derived from the config."""
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
    """Verifies the initial causal_mask is a lower-triangular bool tensor sized to the full block_size."""
    config = _tiny_config()
    model = GPT(config)

    assert model.causal_mask.shape == (1, 1, config.block_size, config.block_size)
    assert model.causal_mask.dtype == torch.bool
    expected = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
    torch.testing.assert_close(model.causal_mask[0, 0], expected)


@pytest.mark.unit
def test_gpt_setup_caches_allocates_kv_cache_per_block():
    """Verifies setup_caches attaches a KVCache to every transformer block, with the sequence length rounded up to a multiple of 8."""
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
    """Verifies setup_caches does nothing when the existing caches already cover the requested size."""
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
    """Verifies forward derives positions from the input when input_pos is omitted and returns correctly shaped logits."""
    torch.manual_seed(42)
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 5))

    logits = model(idx)

    assert logits.shape == (1, 5, config.vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.integration
def test_gpt_forward_with_explicit_input_pos_returns_logits():
    """Verifies forward returns correctly shaped logits when called with explicit input positions."""
    torch.manual_seed(42)
    config = _tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))
    input_pos = torch.tensor([0, 1, 2])

    logits = model(idx, input_pos=input_pos)

    assert logits.shape == (1, 3, config.vocab_size)


@pytest.mark.integration
def test_gpt_prefill_returns_single_next_token():
    """Verifies prefill returns a single next-token tensor sampled from the last-position logits."""
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
    """Verifies decode_one_token raises when input_pos contains more than one position."""
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
    """Verifies generate produces the requested number of tokens and keeps them within the model's vocabulary."""
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
    """Verifies decode_n_tokens returns the requested number of tokens and probabilities, and advances input_pos in place."""
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
    """Verifies load_state_dict_from_url filters legacy bias keys, transposes the four packed weight tensors, and injects a fresh causal_mask."""
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
