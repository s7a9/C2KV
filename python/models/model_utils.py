import json
import torch
from packaging import version
from logging import getLogger
from typing import Tuple, Type, Optional
from dataclasses import asdict
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.integrations import is_deepspeed_zero3_enabled

from gist_args import ModelArgs
from pic_args import PICModelArgs

logger = getLogger(__name__)


class GistLossFunctionWithRegularization:
    def __init__(self, model: PreTrainedModel, param: str, regularization_strength: float):
        super().__init__()
        self.model = model
        self.param = param
        self.label_loss_function = model.loss_function
        self.regularization_strength = regularization_strength
    
    @staticmethod
    def _weight_loss(gist_module: torch.nn.Module, module: torch.nn.Module) -> torch.Tensor:
        return (gist_module.weight - module.weight).pow(2).mean()
    
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        vocab_size: int,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        label_loss = self.label_loss_function(logits, labels, vocab_size, num_items_in_batch, ignore_index, shift_labels, **kwargs)
        losses = []
        if 'q' in self.param:
            losses.extend(self._weight_loss(layer.self_attn.q_proj, layer.self_attn.gist_q_proj) for layer in self.model.model.layers)
        if 'k' in self.param:
            losses.extend(self._weight_loss(layer.self_attn.k_proj, layer.self_attn.gist_k_proj) for layer in self.model.model.layers)
        if 'v' in self.param:
            losses.extend(self._weight_loss(layer.self_attn.v_proj, layer.self_attn.gist_v_proj) for layer in self.model.model.layers)
        loss = label_loss + torch.stack(losses).mean() * self.regularization_strength
        return loss


def get_model_class(
    model_name_or_path: str,
    gist_param_type: str,
) -> Tuple[Type[PretrainedConfig], Type[PreTrainedModel]]:
    if gist_param_type == "pic":
        from .qwen3 import Qwen3ForCausalLM, Qwen3Config
        architecture_to_class = {
            'Qwen3ForCausalLM': (Qwen3Config, Qwen3ForCausalLM),
        }
    elif gist_param_type == "qkv":
        from .llama import LlamaForCausalLM, LlamaConfig
        from .qwen2_5 import Qwen2ForCausalLM, Qwen2Config
        architecture_to_class = {
            'LlamaForCausalLM': (LlamaConfig, LlamaForCausalLM),
            'Qwen2ForCausalLM': (Qwen2Config, Qwen2ForCausalLM),
        }
    else:
        raise ValueError(f"Unsupported gist_param_type: {gist_param_type}")
    probe_config = AutoConfig.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True,
        local_files_only=True,
    )
    architecture = probe_config.architectures[0]
    if architecture not in architecture_to_class:
        supported = ", ".join(architecture_to_class)
        raise ValueError(f"Unsupported architecture {architecture}; expected one of: {supported}")
    config_class, model_class = architecture_to_class[architecture]
    return config_class, model_class


def get_model_and_tokenizer(
    model_args: ModelArgs | PICModelArgs,
    device: str="cuda", 
    evaluation_mode: bool=True, 
    return_tokenizer_only: bool=False, 
    **kwargs
):    
    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)
    
    model_name_or_path = model_args_dict["model_name_or_path"]
    cache_dir = model_args_dict["model_cache_dir"]
    # access_token = model_args_dict["access_token"]

    logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        cache_dir=cache_dir, 
        padding_side=model_args_dict["padding_side"], 
        # token=access_token, 
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in model_name_or_path.lower():
        # there is a bug in llama tokenizer
        tokenizer.chat_template = r"{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    
    if return_tokenizer_only:
        return tokenizer

    dtype = model_args_dict["dtype"]
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    device_map = model_args_dict["device_map"]
    if device_map is None and not is_deepspeed_zero3_enabled():
        device_map = {"": device}
    
    rope_kwargs = {}
    rope_theta = model_args_dict["rope_theta"]
    if rope_theta is not None:
        rope_kwargs["rope_theta"] = rope_theta
    rope_method = model_args_dict["rope_method"]
    if rope_method is not None:
        rope_factor = model_args_dict["rope_factor"]
        rope_scaling = {
            "type": rope_method,
            "factor": rope_factor
        }
        # NOTE: do not destroy the default rope_scaling of the model
        rope_kwargs["rope_scaling"] = rope_scaling

    attn_kwargs = {}
    attn_impl = model_args_dict["attn_impl"]
    if attn_impl is not None:
        if version.parse(transformers.__version__) <= version.parse("4.36"):
            if attn_impl == "flash_attention_2":
                attn_kwargs["use_flash_attention_2"] = True
        else:
            attn_kwargs["attn_implementation"] = attn_impl

    gist_kwargs = {}
    for k, v in model_args_dict.items():
        if k.startswith("gist") and v is not None:
            gist_kwargs[k] = v

    enable_pic = model_args_dict.get("enable_pic", False)
    enable_gist = model_args_dict.get("enable_gist", False)
    if enable_pic and enable_gist:
        raise ValueError("enable_pic and enable_gist are mutually exclusive")

    if enable_pic:
        config_class, model_class = get_model_class(model_name_or_path, "pic")
        config = config_class.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            local_files_only=True,
            pic_enabled=True,
            pic_param=model_args_dict["pic_param"],
            **rope_kwargs,
            **attn_kwargs,
        )
        model = model_class.from_pretrained(
            model_name_or_path,
            config=config,
            dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        )
    elif enable_gist:
        gist_lora = model_args_dict["gist_param"] == 'lora'
        config_class, model_class = get_model_class(model_name_or_path, "lora" if gist_lora else "qkv")

        config = config_class.from_pretrained(
            model_name_or_path, 
            dtype=dtype,
            local_files_only=True,
            **gist_kwargs,
            **rope_kwargs,
            **attn_kwargs,
        )

        if gist_lora and (gist_lora_config_str := model_args_dict["gist_lora_config"]):
            if isinstance(config.gist_lora_config, str):
                config.gist_lora_config = json.loads(config.gist_lora_config)
            gist_lora_config = json.loads(gist_lora_config_str)
            if config.gist_lora_config is None:
                config.gist_lora_config = gist_lora_config
            elif config.gist_lora_config != gist_lora_config:
                raise ValueError(f"gist_lora_config in config is {config.gist_lora_config}, "
                    f"but {gist_lora_config} is specified in model_args")

        if gist_type := model_args_dict["gist_type"]:
            if config.gist_type is None:
                config.gist_type = gist_type
            elif config.gist_type != gist_type:
                raise ValueError(f"gist_type in config is {config.gist_type}, but {gist_type} is specified in model_args")
        if gist_param := model_args_dict["gist_param"]:
            if config.gist_param is None:
                config.gist_param = gist_param
            elif config.gist_param != gist_param:
                raise ValueError(f"gist_param in config is {config.gist_param}, but {gist_param} is specified in model_args")
        if gist_residual_type := model_args_dict["gist_residual_type"]:
            if config.gist_residual_type is None:
                config.gist_residual_type = gist_residual_type
            elif config.gist_residual_type != gist_residual_type:
                raise ValueError(f"gist_residual_type in config is {config.gist_residual_type}, but {gist_residual_type} is specified in model_args")
        if config.gist_token_id is None:
            config.gist_token_id = tokenizer.eos_token_id
        elif config.gist_token_id != tokenizer.eos_token_id:
            raise ValueError(f"gist_token_id in config is {config.gist_token_id}, but {tokenizer.eos_token_id} is specified in tokenizer")
        if model_args_dict["gist_reconstruct_loss_coef"] is not None:
            # this configuration takes effect only when the gist model is trained from base model
            # when loading gist model from gist checkpoint, this configuration may cause error
            config.gist_extra_embed_num = 2

        model = model_class.from_pretrained(
            model_name_or_path, 
            config=config,
            # cache_dir=cache_dir, 
            dtype=dtype,
            device_map=device_map, 
            # token=access_token,
            local_files_only=True,
        )

        if model_args_dict["gist_regularization"] is not None:
            assert not gist_lora, "lora and gist regularization are mutually exclusive"
            model.loss_function = GistLossFunctionWithRegularization(
                model, model_args_dict["gist_regularization"], model_args_dict["gist_regularization_factor"]
            )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir, 
            dtype=dtype,
            device_map=device_map,
            # token=access_token,
            trust_remote_code=True,
            # NOTE: do not destroy the default rope_scaling of the model
            **rope_kwargs,
            **attn_kwargs,
        )

    # load lora
    if model_args_dict["lora"] is not None:
        logger.info(f"loading lora from {model_args_dict['lora']}...")

        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, 
            model_args_dict["lora"],
            torch_dtype=dtype,
            device_map=device_map,
        )
        if model_args_dict["lora_unload"]:
            model = model.merge_and_unload()

    if model_args_dict["enable_tp"]:
        import tensor_parallel as tp
        logger.info("enabling tensor parallelism...")
        
        # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
        model = tp.tensor_parallel(model, sharded=True)

        if model.generation_config.eos_token_id == 128001:
            model.generation_config.eos_token_id = [128001, 128009]

    if isinstance(model, transformers.modeling_utils.PreTrainedModel):
        model = model.eval()
        if evaluation_mode:
            # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
            model.requires_grad_(False)
        # logger.info(model.config)

    # override the default generation config
    generation_config = model_args.get_generation_config()
    if len(generation_config):
        model.generation_config.update(**generation_config)
    logger.info(f"Specified generation config: {generation_config}")

    return model, tokenizer

def format_numel_str(numel: int) -> str:
    T = 1e12
    B = 1e9
    M = 1e6
    K = 1e3
    if numel >= T:
        return f"{numel / T:.2f} T"
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"
