"""Utilities for full-length position-independent caches (PIC).

PIC keeps every source token.  Documents are encoded independently with residual
QKV projections, their keys are kept before RoPE, and RoPE is applied only after
the documents have been assigned positions in the concatenated context.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache
from transformers.integrations import is_deepspeed_zero3_enabled


PIC_GRADIENT_CHECKPOINTING = False


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def make_residual_projection(
    hidden_size: int,
    output_size: int,
    bias: bool,
) -> torch.nn.Linear:
    """Create a strict residual branch whose initial contribution is zero."""
    projection = torch.nn.Linear(hidden_size, output_size, bias=bias)
    projection.weight.data.zero_()
    if projection.bias is not None:
        projection.bias.data.zero_()
    # Prevent post_init from replacing the deliberate zero initialization.
    projection._is_hf_initialized = True
    return projection


def init_missing_residual_projections(attention, missing_keys: list[str]) -> None:
    """Zero only newly introduced residual projections when loading a base model."""
    names = ("residual_q_proj", "residual_k_proj", "residual_v_proj")
    for name in names:
        if not hasattr(attention, name):
            continue
        qualified_suffix = f"self_attn.{name}.weight"
        if not any(key.endswith(qualified_suffix) for key in missing_keys):
            continue
        projection = getattr(attention, name)
        params = [projection.weight]
        if projection.bias is not None:
            params.append(projection.bias)
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                projection.weight.data.zero_()
                if projection.bias is not None:
                    projection.bias.data.zero_()
        else:
            projection.weight.data.zero_()
            if projection.bias is not None:
                projection.bias.data.zero_()


def _pack_cache_tensor(
    tensor: torch.Tensor,
    token_mask: torch.Tensor,
    packed_length: int,
) -> torch.Tensor:
    """Remove per-document padding and pad only at the batch level."""
    packed = []
    for batch_idx in range(tensor.shape[0]):
        sample = tensor[batch_idx][:, token_mask[batch_idx], :]
        pad_length = packed_length - sample.shape[-2]
        if pad_length:
            sample = torch.cat(
                [sample, sample.new_zeros(sample.shape[0], pad_length, sample.shape[-1])],
                dim=-2,
            )
        packed.append(sample)
    return torch.stack(packed, dim=0)


def build_pic_cache(
    model,
    context_input_ids: torch.LongTensor,
    past_key_values: Optional[DynamicCache] = None,
    past_attention_mask: Optional[torch.Tensor] = None,
) -> tuple[DynamicCache, torch.Tensor, torch.Tensor]:
    """Build and concatenate full-length caches for independently encoded documents.

    Args:
        model: ``Qwen3Model`` with PIC enabled.
        context_input_ids: ``(batch, documents, document_length)`` with ``-100`` padding.
        past_key_values: Optional prefix cache, typically the system prompt.
        past_attention_mask: Valid-token mask for the prefix cache.

    Returns:
        The combined cache, its 2-D validity mask, and per-sample context lengths.
    """
    if context_input_ids.ndim != 3:
        raise ValueError(
            "context_input_ids must have shape (batch, documents, document_length), "
            f"got {tuple(context_input_ids.shape)}"
        )
    batch_size, document_count, document_length = context_input_ids.shape
    flat_ids = context_input_ids.reshape(batch_size * document_count, document_length)
    flat_mask = flat_ids.ne(-100)
    model_mask = flat_mask.clone()
    # The fixed-width training dataset contains empty document slots.  Give those
    # rows one masked-out dummy token during attention to avoid all-masked softmax.
    empty_documents = ~model_mask.any(dim=1)
    model_mask[empty_documents, 0] = True
    pad_token_id = model.config.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    flat_ids = flat_ids.masked_fill(~flat_mask, pad_token_id)

    outputs, local_position_ids = model.generate_pic(flat_ids, model_mask)

    document_mask = flat_mask.reshape(batch_size, document_count, document_length)
    context_lengths = document_mask.sum(dim=(1, 2))
    packed_length = int(context_lengths.max().item())
    if packed_length == 0:
        raise ValueError("Each batch must contain at least one non-empty context document")

    past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    if past_attention_mask is None:
        prefix_mask = document_mask.new_ones((batch_size, past_length))
    else:
        prefix_mask = past_attention_mask.to(device=document_mask.device, dtype=torch.bool)

    # Assign positions according to original (uncompressed) document lengths.
    global_positions = local_position_ids.reshape(batch_size, document_count, document_length).clone()
    for batch_idx in range(batch_size):
        prefix_position = past_length
        for document_idx in range(document_count):
            valid_tokens = document_mask[batch_idx, document_idx]
            length = int(valid_tokens.sum().item())
            if length:
                global_positions[batch_idx, document_idx, valid_tokens] += prefix_position
                prefix_position += length
    flat_global_positions = global_positions.reshape(batch_size * document_count, document_length)
    cos, sin = model.rotary_emb(outputs.last_hidden_state, flat_global_positions)

    flattened_mask = document_mask.reshape(batch_size, document_count * document_length)
    context_mask = torch.arange(packed_length, device=document_mask.device).unsqueeze(0)
    context_mask = context_mask < context_lengths.unsqueeze(1)
    combined_layers = []
    for layer_idx, (key, value) in enumerate(outputs.past_key_values):
        key = apply_rotary_pos_emb(key, cos, sin)
        head_count, head_dim = key.shape[1], key.shape[-1]
        key = key.reshape(batch_size, document_count, head_count, document_length, head_dim)
        key = key.transpose(1, 2).reshape(batch_size, head_count, -1, head_dim)
        value = value.reshape(batch_size, document_count, head_count, document_length, head_dim)
        value = value.transpose(1, 2).reshape(batch_size, head_count, -1, head_dim)
        key = _pack_cache_tensor(key, flattened_mask, packed_length)
        value = _pack_cache_tensor(value, flattened_mask, packed_length)
        if past_key_values is not None:
            past_layer = past_key_values.layers[layer_idx]
            key = torch.cat([past_layer.keys, key], dim=-2)
            value = torch.cat([past_layer.values, value], dim=-2)
        combined_layers.append((key, value))

    combined_mask = torch.cat([prefix_mask, context_mask], dim=1)
    return DynamicCache(combined_layers, config=model.config), combined_mask, context_lengths


def process_context_input_ids(
    model,
    context_input_ids: torch.LongTensor,
    past_key_values: Optional[DynamicCache],
    attention_mask: Optional[torch.Tensor],
    past_attention_mask: Optional[torch.Tensor] = None,
) -> tuple[DynamicCache, torch.Tensor]:
    """Prepare a PIC prefix cache and extend the query attention mask."""
    cache, cache_mask, _ = build_pic_cache(
        model,
        context_input_ids,
        past_key_values=past_key_values,
        past_attention_mask=past_attention_mask,
    )
    if attention_mask is None:
        raise ValueError("attention_mask is required when context_input_ids is provided")
    attention_mask = torch.cat(
        [cache_mask.to(device=attention_mask.device, dtype=attention_mask.dtype), attention_mask],
        dim=1,
    )
    return cache, attention_mask
