# Agent evaluation

This directory is independent of the multi-document evaluation stack.
`AgentDataset` examples contain:

- `system_prompt` (falls back to `You are a helpful assistant.`);
- `tools` in Hugging Face function-tool format;
- `messages` without the leading system message;
- expected tool calls for evaluation.

The full-compute path calls `apply_chat_template(messages, tools=tools)` once,
so Qwen3 renders the system prompt and tool definitions in one system message.

The C2KV path renders three independent pieces:

1. the system prompt as a normal system message;
2. tool definitions as another system message, produced by passing only
   `tools=` to the chat template;
3. the remaining conversation and assistant generation prompt.

The rendered system/tool segments have a small LRU cache. Identical definitions
can therefore reuse their prefilled/extracted KV across examples.

`AgentLLMTracesDataset` uses HuggingFace `datasets` cache by default. The first
run builds an Arrow cache for the filtered simple tool-call subset; later runs
with the same parquet files and preprocessing parameters load it directly via
the default `datasets` cache configuration.

`OpenSWETracesDataset` supports NVIDIA Open-SWE-Traces local exports such as
`/home/nas/dch/datasets/nvidia--Open-SWE-Traces/`. Use
`--dataset open_swe_traces`; `--benchmark` may be used to keep only a scaffold
or split, for example `openhands`, `sweagent`, `qwen35`, `minimax`, or a full
directory name like `qwen35_openhands_trajectories`. In history-compression
experiments, `--history-message-range` is used to skip shallow tool-call
prediction samples before applying `--max-samples-per-trace`, which caps how
many samples each single trajectory can contribute.

```bash
export PYTHONPATH=$PWD/python

python python/agent/expr_agent_full.py \
  --model <base-or-checkpoint-path> \
  --dataset-path /path/to/agent-llm-traces \
  --output-file results/agent_full.jsonl

python python/agent/expr_agent_c2kv.py \
  --model <c2kv-checkpoint-path> \
  --dataset-path /path/to/agent-llm-traces \
  --override-ratio 4 \
  --reuse-cache-size 1 \
  --output-file results/agent_c2kv.jsonl

python python/agent/expr_agent_c2kv_api.py \
  --base-url http://localhost:30000 \
  --model default \
  --tokenizer <c2kv-checkpoint-path> \
  --dataset-path /path/to/agent-llm-traces \
  --benchmark browsecompplus \
  --compression-ratio 2 \
  --tool-token-range 512,4096 \
  --request-token-range 1024,8192 \
  --preprocess-workers 4 \
  --max-examples 5 \
  --save-inputs \
  --output-file results/agent_c2kv_api.jsonl

python python/agent/expr_agent_c2kv_api.py \
  --base-url http://localhost:30000 \
  --model default \
  --tokenizer <c2kv-checkpoint-path> \
  --dataset open_swe_traces \
  --dataset-path /home/nas/dch/datasets/nvidia--Open-SWE-Traces \
  --benchmark openhands \
  --max-samples-per-trace 3 \
  --max-samples 1000 \
  --output-file results/open_swe_c2kv_api.jsonl
```
