"""
FILE: test file for `GPTConfig` dataclass in `gpt.py`.

HOW-TO-RUN (in root):  
    python3 -m pytest tests/test_gpt_config.py  
"""

from utils.gpt import GPTConfig


# verify that default GPTConfig() object has the expected field values expected values as defined in gpt.py
def test_gpt_config_default_values():
    config = GPTConfig()

    assert config.block_size == 20 * 129
    assert config.vocab_size == 1025
    assert config.n_layer == 24
    assert config.n_head == 16
    assert config.dim == 1024
    assert config.intermediate_size == 4 * 1024
    assert config.tokens_per_frame == 129

# verifies that the bos_token property is derived correctly from vocab_size
def test_gpt_config_bos_token_uses_vocab_size_minus_one():
    config = GPTConfig()

    assert config.bos_token == config.vocab_size - 1
    assert config.bos_token == 1024

# verifies that bos_token is not hardcoded and changes correctly when a custom vocab_size is given
def test_gpt_config_bos_token_updates_when_vocab_size_changes():
    config = GPTConfig(vocab_size=2048)

    assert config.bos_token == 2047

# verifies that the head_dim attributes returns dim // n_head for the default configuration
def test_gpt_config_head_dim_computes_dim_divided_by_n_head():
    config = GPTConfig()

    assert config.head_dim == config.dim // config.n_head
    assert config.head_dim == 64

# verifies that the head_dim changes correctly when dim and n_head are customized
def test_gpt_config_head_dim_updates_with_custom_dim_and_n_head():
    config = GPTConfig(dim=768, n_head=12)
    assert config.head_dim == 64

    config = GPTConfig(dim=512, n_head=8)
    assert config.head_dim == 64

# verifies that user provided constructor overrides  are stored correctly
def test_gpt_config_allows_custom_field_overrides():
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

# verifies that head-dim is returned as an integer
def test_gpt_config_head_dim_is_integer():
    config = GPTConfig()

    assert isinstance(config.head_dim, int)