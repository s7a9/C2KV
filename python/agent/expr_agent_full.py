from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.dataset import AgentDataset, load_agent_dataset
    from agent.eval_utils import (
        example_result,
        print_example,
        save_results,
        select_example_indices,
    )
    from agent.render import render_full_prompt
else:
    from .dataset import AgentDataset, load_agent_dataset
    from .eval_utils import (
        example_result,
        print_example,
        save_results,
        select_example_indices,
    )
    from .render import render_full_prompt


@torch.inference_mode()
def evaluate(args: argparse.Namespace, dataset: AgentDataset) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    device = model.device

    indices = select_example_indices(
        dataset,
        tokenizer,
        max_examples=args.max_examples,
    )
    total = len(indices)
    results = []
    for output_index, dataset_index in enumerate(
        tqdm(indices, desc="Agent full compute")
    ):
        example = dataset[dataset_index]
        prompt = render_full_prompt(tokenizer, example)
        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        max_new_tokens = example.max_new_tokens or dataset.max_new_tokens
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        generated = outputs[0, inputs.input_ids.shape[1] :]
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        print_example(output_index, total, example, prediction)
        results.append(example_result(dataset, example, prediction))
    return save_results(args.output_file, args.model, dataset, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-compute agent evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="agent_llm_traces")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-samples-per-trace",
        type=int,
        help="For open_swe_traces, cap the number of tool-call samples produced from each trace",
    )
    parser.add_argument("--max-tools", type=int)
    parser.add_argument(
        "--benchmark",
        help="Only keep examples from one benchmark, or comma-separated benchmarks",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-file")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_agent_dataset(
        args.dataset,
        args.dataset_path,
        max_samples=args.max_samples,
        max_tools=args.max_tools,
        max_samples_per_trace=args.max_samples_per_trace,
        max_new_tokens=args.max_new_tokens,
        benchmark=args.benchmark,
    )
    summary = evaluate(args, dataset)
    print(f"\nTool-call accuracy: {summary['tool_call_accuracy']:.4f}")


if __name__ == "__main__":
    main()
