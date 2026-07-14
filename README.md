# Residual-QKV Position-Independent Cache for Qwen3

This branch implements a full-length Position-Independent Cache (PIC) for Qwen3.
It deliberately does **not** use anchor/gist tokens and performs **no cache
compression**. Every valid document token produces one cache entry.

The base Qwen3 weights are frozen. Each attention layer receives three optional,
zero-initialized residual projections:

```text
Q_pic = Q_base(hidden) + residual_q(hidden)
K_pic = K_base(hidden) + residual_k(hidden)
V_pic = V_base(hidden) + residual_v(hidden)
```

Documents are encoded independently. Their K states are saved before RoPE, the
full document caches are concatenated, and RoPE is applied once using the tokens'
final global positions. The query then attends to the system prompt plus the
concatenated full-length document caches.

## Scope

- Model: Qwen3 only.
- Cache ratio: exactly 1:1 for non-padding context tokens.
- Trainable parameters: `residual_q_proj`, `residual_k_proj`, and
  `residual_v_proj` in every attention layer.
- Initialization: residual weights and biases are zero, so enabling PIC starts
  from the base QKV projections without a discontinuity.
- Data: the existing mixed Multi-Document preprocessing is reused unchanged.

The former Llama/Qwen2.5 and compressed C2KV files remain as legacy code, but the
Qwen3 model and `train.train_mdoc` entry point no longer instantiate or consume
anchor tokens.

## Important files

```text
python/models/pic_utils.py              # Build, position, pack, and concatenate PIC caches
python/models/qwen3/modeling_qwen3.py   # Qwen3 residual-QKV forward and independent document encoding
python/pic_args.py                       # PIC-only model and training arguments
python/train/train_data.py              # Reused MultiDocDataset preprocessing
python/train/pic_trainer.py             # PICMultiDocTrainer
python/train/train_mdoc.py              # Qwen3 PIC training entry point
python/inference/mdocdataset.py         # Reused MDQA evaluation datasets
python/inference/expr_pic.py             # PIC evaluation entry point
tests/test_pic.py                        # Tiny-Qwen3 cache/initialization/gradient checks
```

## Data layout

The training entry point retains the repository's mixed dataset layout:

```text
<train_data>/
├── longmagpie_1024/
└── microsoft--NextCoderDataset/

<train_data>_cleaned/
├── hotpotqa_train_cleaned/
└── wikimqa_train_cleaned/
```

It also loads `allenai/tulu-3-sft-mixture` through the existing
`MultiDocDataset` implementation. Documents are right-padded with `-100` in the
dataset and empty document slots are excluded when PIC caches are packed.

## Training

Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Run a provided Qwen3 script, for example:

```bash
bash scripts/train_qwen3-4b-mixed_mdoc.sh
```

The essential arguments are:

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

torchrun --nproc_per_node 8 -m train.train_mdoc \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --attn_impl flash_attention_2 \
    --enable_pic True \
    --pic_param qkv \
    --only_train_pic True \
    --train_data ./datasets \
    --output_dir ./checkpoints/qwen3-4b-residual-qkv-pic \
    --per_device_train_batch_size 1 \
    --bf16 True \
    --do_train True \
    --deepspeed ./configs/ds_config.json
```

Use `--pic_gradient_checkpointing True` to checkpoint the independent-document
forward pass. This is useful for larger Qwen3 variants because PIC intentionally
keeps all token states.

## Evaluation

The new evaluator reuses `python/inference/mdocdataset.py` and supports the same
dataset names as the existing MDQA experiments:

```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH

python python/inference/expr_pic.py \
    --model ./checkpoints/qwen3-4b-residual-qkv-pic \
    --dataset hotpotqa \
    --max_examples 500 \
    --output_file ./results/hotpotqa/residual_qkv_pic.jsonl
```

Each output row records both `context_tokens` and `cached_context_tokens`; they
must be equal.

## Checks

```bash
PYTHONPATH=python python -m pytest -q tests/test_pic.py
```

The checks cover:

1. no anchor/gist modules in Qwen3;
2. zero initialization of all residual QKV parameters;
3. one cache entry per non-padding document token;
4. gradients reaching residual QKV while the base model remains frozen.
