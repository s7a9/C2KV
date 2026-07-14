import torch

from models.pic_utils import build_pic_cache
from models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from pic_args import PICModelArgs


def tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        pic_enabled=True,
        pic_param="qkv",
        layer_types=["full_attention", "full_attention"],
        # CPU-only unit tests cannot load the FlashAttention CUDA extension.
        _attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config)


def test_training_defaults_to_flash_attention_2():
    assert PICModelArgs().attn_impl == "flash_attention_2"


def test_pic_has_no_anchor_modules_and_zero_initialized_residuals():
    model = tiny_model()
    module_names = dict(model.named_modules())
    assert not any("gist" in name or "anchor" in name for name in module_names)
    residual_parameters = [
        parameter for name, parameter in model.named_parameters() if "residual_" in name
    ]
    assert residual_parameters
    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in residual_parameters)
    hidden_states = torch.randn(2, 3, model.config.hidden_size)
    attention = model.model.layers[0].self_attn
    base_qkv = attention._project_qkv(hidden_states, use_pic=False)
    pic_qkv = attention._project_qkv(hidden_states, use_pic=True)
    assert all(torch.equal(base, pic) for base, pic in zip(base_qkv, pic_qkv))


def test_pic_cache_retains_every_nonpadding_token():
    model = tiny_model().eval()
    documents = torch.tensor(
        [[
            [1, 2, 3, -100],
            [4, 5, -100, -100],
            [6, 7, 8, 9],
            [-100, -100, -100, -100],
        ]]
    )
    cache, cache_mask, lengths = build_pic_cache(model.model, documents)
    assert lengths.tolist() == [9]
    assert cache.get_seq_length() == 9
    assert cache_mask.sum().item() == 9


def test_only_residual_qkv_are_trainable_and_receive_gradients():
    model = tiny_model().train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("residual_" in name)
    documents = torch.tensor([[[1, 2, 3, -100], [4, 5, 6, -100]]])
    query = torch.tensor([[7, 8, 9]])
    outputs = model(
        input_ids=query,
        attention_mask=torch.ones_like(query, dtype=torch.bool),
        position_ids=torch.tensor([[6, 7, 8]]),
        context_input_ids=documents,
        labels=query,
        use_cache=False,
    )
    outputs.loss.backward()
    residual_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "residual_" in name
    ]
    assert any(gradient is not None and torch.count_nonzero(gradient).item() for gradient in residual_grads)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if "residual_" not in name
    )
