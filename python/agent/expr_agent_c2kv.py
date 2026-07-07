from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, MutableMapping, TypeVar

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.dataset import AgentDataset, load_agent_dataset
    from agent.eval_utils import (
        example_result,
        print_example,
        save_results,
        select_example_indices,
    )
    from agent.render import render_reuse_prompt
    from inference.expr_timer import ExprTimer
    from models import blend_gist_key_values, get_model_class
else:
    from .dataset import AgentDataset, load_agent_dataset
    from .eval_utils import (
        example_result,
        print_example,
        save_results,
        select_example_indices,
    )
    from .render import render_reuse_prompt
    from inference.expr_timer import ExprTimer
    from models import blend_gist_key_values, get_model_class


def _tokenize_rendered(tokenizer: Any, text: str, device: torch.device) -> Any:
    return tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)


def _prefill_kv_cache(model: Any, inputs: Any) -> Any:
    return model(**inputs, use_cache=True, return_dict=True).past_key_values


T = TypeVar("T")


def _cache_put(
    cache: MutableMapping[str, T],
    key: str,
    value: T,
    capacity: int,
) -> None:
    if capacity <= 0:
        return
    cache[key] = value
    if len(cache) > capacity:
        cache.pop(next(iter(cache)))


@torch.inference_mode()
def evaluate(args: argparse.Namespace, dataset: AgentDataset) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    _, model_class = get_model_class(args.model, "qkv")
    model = model_class.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    device = model.device
    timer = ExprTimer("agent_c2kv", enable=args.profile)
    system_cache_store: OrderedDict[str, Any] = OrderedDict()
    tool_cache_store: OrderedDict[str, tuple[Any, Any, Any, int]] = OrderedDict()

    indices = select_example_indices(
        dataset,
        tokenizer,
        max_examples=args.max_examples,
    )
    total = len(indices)
    results = []
    for output_index, dataset_index in enumerate(tqdm(indices, desc="Agent C2KV")):
        example = dataset[dataset_index]
        record = timer.record(example.qid)
        prompt = render_reuse_prompt(tokenizer, example)

        cached_system = system_cache_store.get(prompt.system)
        if cached_system is None:
            system_inputs = _tokenize_rendered(tokenizer, prompt.system, device)
            with record.record("system_prefill"):
                system_cache = _prefill_kv_cache(model, system_inputs)
            system_length = system_inputs.attention_mask.sum().item()
            _cache_put(
                system_cache_store,
                prompt.system,
                (system_cache, system_length),
                args.reuse_cache_size,
            )
        else:
            system_cache_store.move_to_end(prompt.system)
            system_cache, system_length = cached_system

        cached_tools = tool_cache_store.get(prompt.tools)
        if cached_tools is None:
            tool_inputs = _tokenize_rendered(tokenizer, prompt.tools, device)
            gist_kwargs = {}
            if getattr(model.config, "gist_type", None) == "dynamic-interleave":
                gist_kwargs["ratio"] = args.override_ratio
            model.model.config._attn_implementation = "flex_attention"
            print(f"Tool inputs shape: {tool_inputs.input_ids.shape}, attention mask shape: {tool_inputs.attention_mask.shape}")
            outputs, gist_mask, tool_position_ids = model.model.generate_gist(
                **tool_inputs,
                **gist_kwargs,
            )
            tool_position_ids = tool_position_ids[:, -gist_mask.shape[1] :]
            tool_length = tool_inputs.attention_mask.sum().item()
            raw_tool_cache = (
                outputs.past_key_values,
                gist_mask,
                tool_position_ids,
                tool_length,
            )
            _cache_put(
                tool_cache_store,
                prompt.tools,
                raw_tool_cache,
                args.reuse_cache_size,
            )
        else:
            tool_cache_store.move_to_end(prompt.tools)
            raw_tool_cache = cached_tools
        gist_key_values, gist_mask, tool_position_ids, tool_length = raw_tool_cache

        with record.record("blend"):
            tool_cache, _ = blend_gist_key_values(
                model.config,
                [gist_key_values],
                [gist_mask],
                [tool_position_ids.clone()],
                model.model.rotary_emb,
                system_length,
            )
        for system_layer, tool_layer in zip(system_cache.layers, tool_cache.layers):
            tool_layer.keys = torch.cat([system_layer.keys, tool_layer.keys], dim=-2)
            tool_layer.values = torch.cat([system_layer.values, tool_layer.values], dim=-2)

        original_prefix_length = system_length + tool_length
        conversation_inputs = _tokenize_rendered(
            tokenizer,
            prompt.conversation,
            device,
        )
        conversation_length = conversation_inputs.input_ids.shape[1]
        cache_length = tool_cache.get_seq_length()
        mock_cache_ids = conversation_inputs.input_ids.new_zeros((1, cache_length))
        input_ids = torch.cat([mock_cache_ids, conversation_inputs.input_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            original_prefix_length,
            original_prefix_length + conversation_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        model.model.config._attn_implementation = "flash_attention_2"
        max_new_tokens = example.max_new_tokens or dataset.max_new_tokens
        with record.record("generate"):
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=tool_cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                use_gist=True,
            )
        prediction = tokenizer.decode(
            generated[0, input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        print_example(output_index, total, example, prediction)
        results.append(
            example_result(
                dataset,
                example,
                prediction,
                record.summary() if args.profile else None,
            )
        )
    return save_results(
        args.output_file,
        args.model,
        dataset,
        results,
        timer.statistics() if args.profile else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C2KV agent evaluation")
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
    parser.add_argument("--override-ratio", type=int, default=4)
    parser.add_argument(
        "--reuse-cache-size",
        type=int,
        default=1,
        help="Number of distinct rendered system/tool segments to keep on GPU",
    )
    parser.add_argument("--profile", action="store_true")
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
