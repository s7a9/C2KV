"""Trainer for Qwen3 full-length residual-QKV PIC."""

from typing import Any, Optional, Union

import torch
from transformers.cache_utils import DynamicCache
from transformers.trainer import Trainer

from pic_args import PICModelArgs


class PICMultiDocTrainer(Trainer):
    """Reuse MultiDocDataset while independently encoding every context document."""

    def __init__(self, *args, max_doc_length: int, model_args: PICModelArgs, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.max_doc_length = max_doc_length

    @torch.no_grad()
    def _build_system_kv(
        self, model, system_input_ids: torch.Tensor
    ) -> tuple[DynamicCache, torch.Tensor, int]:
        device = model.device
        system_input_ids = system_input_ids.to(device)
        real_mask = system_input_ids.ne(-100)
        real_lengths = real_mask.sum(dim=1)
        batch_size = system_input_ids.shape[0]
        system_width = int(real_lengths.max().item())
        pad_id = model.model.config.pad_token_id
        if pad_id is None:
            pad_id = 0
        left_ids = system_input_ids.new_full((batch_size, system_width), pad_id)
        system_mask = system_input_ids.new_zeros((batch_size, system_width))
        for batch_idx in range(batch_size):
            length = int(real_lengths[batch_idx].item())
            if length:
                left_ids[batch_idx, system_width - length:] = system_input_ids[batch_idx][real_mask[batch_idx]]
                system_mask[batch_idx, system_width - length:] = 1

        model.model.config._attn_implementation = "flash_attention_2"
        was_training = model.training
        model.eval()
        outputs = model(left_ids, attention_mask=system_mask, use_cache=True, logits_to_keep=1)
        if was_training:
            model.train()
        return outputs.past_key_values, system_mask, system_width

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        inputs.pop('dynamic', None)
        batch_size = inputs['context_input_ids'].shape[0]
        context_mask = inputs['context_input_ids'].ne(-100)
        system_input_ids = inputs.pop('system_input_ids')
        system_kv, system_mask, system_width = self._build_system_kv(model, system_input_ids)

        inputs['context_input_ids'] = inputs['context_input_ids'].reshape(
            batch_size, -1, self.max_doc_length
        )
        assert context_mask.any(dim=1).all(), f"context_input_ids must contain at least one non-empty document, {inputs}"
        inputs['past_key_values'] = system_kv
        inputs['past_attention_mask'] = system_mask

        query_length = inputs['input_ids'].shape[1]
        position_ids = torch.arange(
            query_length, dtype=torch.long, device=inputs['input_ids'].device
        ).unsqueeze(0).repeat(batch_size, 1)
        position_ids += system_width + context_mask.sum(dim=1, keepdim=True)
        inputs['position_ids'] = position_ids

        # PIC uses ordinary causal attention within each independent document;
        # unlike anchor-token C2KV it needs no custom FlexAttention mask.
        model.model.config._attn_implementation = "flash_attention_2"
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.model.config._attn_implementation = "flash_attention_2"
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
