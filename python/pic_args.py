"""Command-line arguments for full-length residual-QKV PIC."""

from dataclasses import asdict, dataclass, field
from typing import Optional

from transformers.training_args import TrainingArguments


@dataclass
class PICModelArgs:
    model_cache_dir: str | None = None
    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Qwen3 base model or residual-QKV PIC checkpoint."},
    )
    padding_side: str = "right"
    attn_impl: Optional[str] = "flash_attention_2"
    max_position_embeddings: Optional[int] = None
    rope_theta: Optional[float] = None
    rope_method: Optional[str] = None
    rope_factor: float = 1.0
    dtype: str = "bf16"
    device_map: Optional[str] = None

    enable_pic: bool = field(
        default=True,
        metadata={"help": "Enable full-length position-independent residual QKV."},
    )
    pic_param: str = field(
        default="qkv",
        metadata={"help": "Non-empty combination of q, k, and v residual projections."},
    )
    pic_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Checkpoint independent-document forwards."},
    )

    enable_tp: bool = False
    lora: Optional[str] = None
    lora_unload: bool = True

    # Kept for the shared model loader; training does not generate responses.
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    def get_generation_config(self) -> dict:
        return {
            key: value
            for key, value in {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }.items()
            if value is not None
        }

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PICTrainingArgs(TrainingArguments):
    only_train_pic: bool = field(
        default=True,
        metadata={"help": "Freeze Qwen3 and train only residual QKV projections."},
    )
    train_data: str | None = field(
        default=None,
        metadata={"help": "Root of the existing mixed Multi-Document datasets."},
    )
    dataset_shuffle_seed: int = 42
