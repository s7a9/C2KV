from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.dataset import AgentDataset, AgentExample, load_agent_dataset
    from agent.eval_utils import (
        example_result,
        print_example,
        save_results,
    )
    from agent.render import render_reuse_prompt
else:
    from .dataset import AgentDataset, AgentExample, load_agent_dataset
    from .eval_utils import (
        example_result,
        print_example,
        save_results,
    )
    from .render import render_reuse_prompt


def extract_segment(
    base_url: str,
    text: str,
    compression_ratio: int,
    role: str = "system",
) -> Dict[str, Any]:
    url = f"{base_url}/v1/c2kv/extract"
    payload = {
        "text": text,
        "compression_ratio": compression_ratio,
        "role": role,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        warnings.warn(f"extract error: {exc}")
        return {}


def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_new_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        warnings.warn(f"chat completion error: {exc}")
        return {"choices": [{"message": {"content": ""}}]}


def response_to_prediction(response_json: Dict[str, Any]) -> str:
    try:
        message = response_json["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        warnings.warn(f"malformed chat completion response: {exc}")
        return ""
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        return json.dumps(tool_calls, ensure_ascii=False)
    content = message.get("content")
    if isinstance(content, str) and content:
        return content
    if content is not None and not isinstance(content, str):
        warnings.warn(
            f"chat completion returned non-string content ({type(content).__name__}); using empty prediction"
        )
        return ""
    return content if isinstance(content, str) else ""


def parse_token_range(value: Optional[str]) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Token range must be formatted as a,b; got {value!r}"
        )
    low = int(parts[0]) if parts[0].strip() else None
    high = int(parts[1]) if parts[1].strip() else None
    if low is None and high is None:
        raise argparse.ArgumentTypeError(
            f"Token range must specify at least one bound; got {value!r}"
        )
    if low is not None and low < 0:
        raise argparse.ArgumentTypeError(f"Lower bound must be >= 0; got {low}")
    if high is not None and high < 0:
        raise argparse.ArgumentTypeError(f"Upper bound must be >= 0; got {high}")
    if low is not None and high is not None and low > high:
        raise argparse.ArgumentTypeError(
            f"Lower bound must be <= upper bound; got {value!r}"
        )
    return low, high


def token_count_in_range(
    token_count: int,
    token_range: Optional[Tuple[Optional[int], Optional[int]]],
) -> bool:
    if token_range is None:
        return True
    low, high = token_range
    if low is not None and token_count < low:
        return False
    if high is not None and token_count > high:
        return False
    return True


def build_chat_messages(
    example: AgentExample,
    tool_message_content: str,
    tool_key_hash: Optional[str],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": example.system_prompt},
    ]
    tool_message: Dict[str, Any] = {
        "role": "system",
        "content": tool_message_content,
    }
    if tool_key_hash:
        tool_message["c2kv_key_hash"] = tool_key_hash
    messages.append(tool_message)
    messages.extend(example.messages)
    return messages


def render_chat_request_for_count(tokenizer: Any, messages: List[Dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def system_message_content(rendered_system_message: str) -> str:
    """Strip one Qwen-style rendered system message to the content sent via API.

    The HTTP API templates role/content messages internally.  For C2KV API
    parity, /extract and /chat must therefore receive the same raw content
    instead of an already wrapped <|im_start|>...<|im_end|> message.
    """

    start = "<|im_start|>system\n"
    end = "<|im_end|>"
    if rendered_system_message.startswith(start):
        body = rendered_system_message[len(start) :]
        end_index = body.rfind(end)
        if end_index >= 0:
            return body[:end_index]
    warnings.warn(
        "Could not strip rendered system message wrappers; using rendered text as-is"
    )
    return rendered_system_message


def _cache_put(cache: Dict[str, str], key: str, value: str, capacity: int) -> None:
    if capacity <= 0:
        return
    cache[key] = value
    if len(cache) > capacity:
        cache.pop(next(iter(cache)))


def _process_example(
    output_index: int,
    total: int,
    example: AgentExample,
    tool_content: str,
    tool_tokens: int,
    request_tokens: int,
    args: argparse.Namespace,
    tool_cache: Dict[str, str],
    cache_lock: Lock,
) -> Tuple[int, Dict[str, Any], float, float]:
    base_url = args.base_url.rstrip("/")
    key_hash = None
    extract_json: Optional[Dict[str, Any]] = None

    t0 = time.perf_counter()
    with cache_lock:
        cached_key_hash = tool_cache.get(tool_content)
    if cached_key_hash is not None:
        key_hash = cached_key_hash
    else:
        try:
            extract_json = extract_segment(
                base_url,
                tool_content,
                args.compression_ratio,
                role=args.extract_role,
            )
            if extract_json.get("success") and extract_json.get("key_hash"):
                key_hash = extract_json["key_hash"]
                with cache_lock:
                    _cache_put(
                        tool_cache,
                        tool_content,
                        key_hash,
                        args.reuse_cache_size,
                    )
            else:
                warnings.warn(
                    f"[{example.qid}] extract failed: "
                    f"{extract_json.get('error') if extract_json else None}"
                )
        except requests.RequestException as exc:
            warnings.warn(f"[{example.qid}] extract error: {exc}")
    t_extract = time.perf_counter() - t0

    messages = build_chat_messages(example, tool_content, key_hash)
    max_new_tokens = example.max_new_tokens or args.max_new_tokens

    t1 = time.perf_counter()
    response_json = chat_completion(
        base_url,
        args.model,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
    )
    t_chat = time.perf_counter() - t1
    prediction = response_to_prediction(response_json)

    if args.print_examples:
        print_example(output_index, total, example, prediction)

    record = example_result(
        args.dataset_obj,
        example,
        prediction,
        {"extract": round(t_extract, 4), "chat": round(t_chat, 4)}
        if args.profile
        else None,
    )
    record["tool_key_hash"] = key_hash
    record["tool_tokens"] = tool_tokens
    record["request_tokens"] = request_tokens
    if args.save_inputs:
        record["inputs"] = {
            "extract": {
                "text": tool_content,
                "compression_ratio": args.compression_ratio,
                "role": args.extract_role,
            },
            "chat": {
                "model": args.model,
                "messages": messages,
                "max_completion_tokens": max_new_tokens,
                "temperature": args.temperature,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }
        if extract_json is not None:
            record["extract_response"] = extract_json
        record["chat_response"] = response_json

    return output_index, record, t_extract, t_chat


def _preprocess_example_for_selection(
    dataset_index: int,
    example: AgentExample,
    tokenizer: Any,
    tool_token_range: Optional[Tuple[Optional[int], Optional[int]]],
    request_token_range: Optional[Tuple[Optional[int], Optional[int]]],
) -> Optional[Tuple[int, AgentExample, str, int, int]]:
    tool_content = system_message_content(render_reuse_prompt(tokenizer, example).tools)
    tool_tokens = len(tokenizer.encode(tool_content, add_special_tokens=False))
    if not token_count_in_range(tool_tokens, tool_token_range):
        return None

    chat_messages = build_chat_messages(example, tool_content, tool_key_hash=None)
    request_text = render_chat_request_for_count(tokenizer, chat_messages)
    request_tokens = len(tokenizer.encode(request_text, add_special_tokens=False))
    if not token_count_in_range(request_tokens, request_token_range):
        return None

    return dataset_index, example, tool_content, tool_tokens, request_tokens


def select_examples_with_token_preprocessing(
    args: argparse.Namespace,
    dataset: AgentDataset,
    tokenizer: Any,
) -> Tuple[List[Tuple[AgentExample, str, int, int]], int]:
    selected: List[Tuple[AgentExample, str, int, int]] = []
    scanned = 0
    preprocess_workers = max(1, args.preprocess_workers)
    chunk_size = max(1, preprocess_workers * args.preprocess_chunk_multiplier)

    for chunk_start in range(0, len(dataset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        scanned = chunk_end
        chunk_examples = [
            (dataset_index, dataset[dataset_index])
            for dataset_index in range(chunk_start, chunk_end)
        ]

        if preprocess_workers == 1:
            chunk_results = [
                _preprocess_example_for_selection(
                    dataset_index,
                    example,
                    tokenizer,
                    args.tool_token_range,
                    args.request_token_range,
                )
                for dataset_index, example in chunk_examples
            ]
        else:
            with ThreadPoolExecutor(max_workers=preprocess_workers) as executor:
                futures = [
                    executor.submit(
                        _preprocess_example_for_selection,
                        dataset_index,
                        example,
                        tokenizer,
                        args.tool_token_range,
                        args.request_token_range,
                    )
                    for dataset_index, example in chunk_examples
                ]
                chunk_results = [future.result() for future in futures]

        for result in chunk_results:
            if result is None:
                continue
            _, example, tool_content, tool_tokens, request_tokens = result
            selected.append((example, tool_content, tool_tokens, request_tokens))
            if args.max_examples is not None and len(selected) >= args.max_examples:
                return selected, scanned

    return selected, scanned


def evaluate(args: argparse.Namespace, dataset: AgentDataset) -> Dict[str, Any]:
    tokenizer_name = args.tokenizer or args.model
    if tokenizer_name == "default":
        raise ValueError(
            "Please pass --tokenizer <local-checkpoint-or-tokenizer-path> when "
            '--model is the API placeholder name "default".'
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    selected, scanned = select_examples_with_token_preprocessing(
        args,
        dataset,
        tokenizer,
    )

    total = len(selected)
    print(
        f"Selected {total} examples from {scanned} scanned examples "
        f"(tool_token_range={args.tool_token_range}, "
        f"request_token_range={args.request_token_range}, "
        f"preprocess_workers={args.preprocess_workers})"
    )

    # Keep the dataset available to worker calls without changing public APIs.
    args.dataset_obj = dataset

    tool_cache: Dict[str, str] = {}
    cache_lock = Lock()
    results: List[Optional[Dict[str, Any]]] = [None] * total
    extract_times: List[float] = []
    chat_times: List[float] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_example,
                output_index,
                total,
                example,
                tool_content,
                tool_tokens,
                request_tokens,
                args,
                tool_cache,
                cache_lock,
            ): output_index
            for output_index, (
                example,
                tool_content,
                tool_tokens,
                request_tokens,
            ) in enumerate(selected)
        }
        with tqdm(total=total, desc="Agent C2KV API") as progress:
            for future in as_completed(futures):
                output_index, record, t_extract, t_chat = future.result()
                results[output_index] = record
                if args.profile:
                    extract_times.append(t_extract)
                    chat_times.append(t_chat)
                progress.update(1)

    final_results = [record for record in results if record is not None]
    statistics: Dict[str, Any] = {
        "base_url": args.base_url.rstrip("/"),
        "compression_ratio": args.compression_ratio,
        "reuse_cache_size": args.reuse_cache_size,
        "tool_token_range": args.tool_token_range,
        "request_token_range": args.request_token_range,
    }
    if args.profile and extract_times:
        statistics.update(
            {
                "extract_mean": round(float(np.mean(extract_times)), 4),
                "extract_std": round(float(np.std(extract_times)), 4),
                "chat_mean": round(float(np.mean(chat_times)), 4),
                "chat_std": round(float(np.std(chat_times)), 4),
            }
        )
    return save_results(args.output_file, args.model, dataset, final_results, statistics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C2KV agent evaluation via HTTP API")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="SGLang/OpenAI-compatible C2KV server base URL",
    )
    parser.add_argument(
        "--model",
        default="default",
        help="Model name sent to /v1/chat/completions",
    )
    parser.add_argument(
        "--tokenizer",
        help="Local tokenizer/checkpoint path used to render the reusable tool segment",
    )
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
    parser.add_argument(
        "--tool-token-range",
        type=parse_token_range,
        help=(
            "Keep examples whose reusable tool-definition content token count is "
            "within a,b. Bounds are inclusive; either side may be empty, e.g. 1000, or ,8000."
        ),
    )
    parser.add_argument(
        "--request-token-range",
        type=parse_token_range,
        help=(
            "Keep examples whose full chat request token count is within a,b. "
            "The count includes system prompt, tool-definition message, conversation, "
            "and assistant generation prompt before C2KV replacement."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-file")
    parser.add_argument(
        "--compression-ratio",
        type=int,
        default=2,
        help="Compression ratio passed to /v1/c2kv/extract",
    )
    parser.add_argument(
        "--extract-role",
        default="system",
        choices=["system", "user", "assistant"],
        help="Role used by /v1/c2kv/extract for the rendered tool segment",
    )
    parser.add_argument(
        "--reuse-cache-size",
        type=int,
        default=1,
        help="Number of distinct rendered tool segments to memoize client-side",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=4,
        help="Number of worker threads for tokenizer-based preprocessing",
    )
    parser.add_argument(
        "--preprocess-chunk-multiplier",
        type=int,
        default=8,
        help=(
            "Preprocess at most preprocess_workers * this many examples per "
            "selection chunk before checking --max-examples"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--save-inputs",
        action="store_true",
        help="Save actual extract/chat payloads and raw API responses into JSONL records",
    )
    parser.add_argument("--print-examples", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load tokenizer from local files only",
    )
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
    print(f"Loaded {len(dataset)} examples from {args.dataset}")
    summary = evaluate(args, dataset)
    print(f"\nTool-call accuracy: {summary['tool_call_accuracy']:.4f}")
    if args.output_file:
        print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main()
