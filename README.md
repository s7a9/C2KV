# KVConcat

KVConcat is a research project focused on efficient Key-Value (KV) cache compression and management for Large Language Models (LLMs). This project implements novel techniques for optimizing KV cache usage during long-context inference, enabling more efficient processing of extended sequences.

## What is KVConcat?

KVConcat addresses the memory bottleneck in transformer-based language models by implementing an intelligent KV cache compression mechanism called "Gist". Instead of storing full KV caches for all tokens, KVConcat compresses context information into compact "gist tokens" while maintaining model performance. This approach significantly reduces memory requirements and enables processing of longer contexts.

## Key Features

### Core Capabilities

- **KV Cache Compression**: Efficiently compresses key-value caches using learnable gist tokens
- **Multiple Compression Strategies**: Supports different gist types including interleave-based compression
- **Flexible Chunk Management**: Configurable chunk sizes and maximum chunk numbers for different use cases
- **Multi-Stage Training Pipeline**: Progressive training approach with pretrain, SFT, and DPO stages
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation
- **Model Support**: Compatible with popular LLM architectures including:
  - Llama models
  - Qwen3 models
  - Variants with LoRA support

### Training Features

- **Stage 1 (Pretrain)**: Initial gist token training with reconstruction loss
- **Stage 2 (Supervised Fine-Tuning)**: Task-specific adaptation
- **Stage 3 (DPO)**: Direct Preference Optimization for alignment
- **Regularization Options**: QKV regularization for stable training
- **DeepSpeed Integration**: Distributed training support with ZeRO optimization
- **Flexible Data Loading**: Support for various datasets with configurable length constraints

### Inference Features

- **Multiple Evaluation Modes**:
  - Full computation baseline
  - Gist model inference
  - Block attention
  - Cache reuse
  - CacheBlend comparison
  - MemoRAG evaluation
- **Benchmark Support**: Integration with LongBench and other long-context benchmarks
- **Generation Configuration**: Customizable sampling, temperature, and top-p parameters
- **Tensor Parallelism**: Multi-GPU inference support

## Project Structure

```
KVConcat/
├── python/
│   ├── models/              # Model implementations
│   │   ├── llama/          # Llama model with gist support
│   │   ├── qwen3/          # Qwen3 model with gist support
│   │   ├── llama_lora/     # Llama with LoRA
│   │   ├── qwen3_lora/     # Qwen3 with LoRA
│   │   ├── gist_utils.py   # Gist token utilities
│   │   └── model_utils.py  # Model loading and utilities
│   ├── train/              # Training scripts
│   │   ├── stage1.py       # Pretrain stage
│   │   ├── stage2.py       # SFT stage
│   │   ├── stage3.py       # DPO stage
│   │   └── trainer.py      # Custom trainers
│   ├── inference/          # Inference and evaluation
│   │   ├── expr_gistmodel.py    # Gist model evaluation
│   │   ├── expr_fullcompute.py  # Baseline evaluation
│   │   ├── reuse_pipeline.py    # KV cache reuse
│   │   └── longbench_metrics.py # Benchmark metrics
│   └── gist_args.py        # Configuration and arguments
├── scripts/                # Training and evaluation scripts
│   ├── stage1_*.sh        # Stage 1 training scripts
│   ├── stage2_*.sh        # Stage 2 training scripts
│   ├── stage3_*.sh        # Stage 3 training scripts
│   └── llm_judge.py       # LLM-based evaluation
├── configs/               # Configuration files
│   └── ds_config.json    # DeepSpeed configuration
└── requirements.txt      # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s)
- PyTorch 2.9.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SJTU-RTEAS/KVConcat.git
cd KVConcat
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- transformers==4.57.1
- torch==2.9.0
- deepspeed==0.18.1
- accelerate>=0.26.0

## Usage

### Training

KVConcat uses a multi-stage training approach:

#### Stage 1: Pretrain
Train gist tokens on large-scale data:
```bash
bash scripts/stage1_qwen3-4b_pretrain.sh
```

#### Stage 2: Supervised Fine-Tuning
Adapt to specific tasks:
```bash
bash scripts/stage2_qwen3-4b.sh
```

#### Stage 3: DPO
Align with human preferences:
```bash
bash scripts/stage3_qwen3-4b.sh
```

### Inference

Evaluate trained models on benchmarks:
```bash
python -m inference.expr_gistmodel \
    --model_name_or_path <path_to_checkpoint> \
    --enable_gist True \
    --gist_type interleave-16 \
    --gist_mode 256,512,768,1024-30
```

### Configuration

Key configuration parameters in `gist_args.py`:

- `enable_gist`: Enable/disable gist compression
- `gist_type`: Compression strategy (e.g., "interleave-4", "interleave-16")
- `gist_mode`: Chunk size and max chunks (e.g., "1024-16")
- `gist_param`: Parameters to compress (e.g., "qkv")
- `gist_regularization`: Regularization type
- `gist_reconstruct_loss_coef`: Reconstruction loss weight

## Research Background

KVConcat is developed by the Real-Time Embedded and Autonomous Systems (RTEAS) group at Shanghai Jiao Tong University. The project explores efficient inference techniques for long-context language models, addressing the quadratic memory growth problem of traditional KV caching.

## Citation

If you use KVConcat in your research, please cite our work:

```bibtex
@misc{kvconcat2024,
  title={KVConcat: Efficient KV Cache Compression for Long-Context Language Models},
  author={SJTU-RTEAS},
  year={2024},
  howpublished={\url{https://github.com/SJTU-RTEAS/KVConcat}}
}
```

## License

Please refer to the repository for licensing information.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or collaborations, please contact the SJTU-RTEAS group.
