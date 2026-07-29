import argparse
import json
import math
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm

from expr_c2kv_api import chat_completion, cut_documents, extract_document, safe_metric
from mdocdataset import AbstractMDQADataset, load_mdoc_dataset


REUSE_PATTERNS = ("forward", "backward", "middle", "random")
REUSE_RATIOS = (1.0, 0.75, 0.5, 0.25)
REUSE_PATTERN_ALIASES = {
    "front": "forward",
    "from-front": "forward",
    "从前往后": "forward",
    "back": "backward",
    "from-back": "backward",
    "从后往前": "backward",
    "center": "middle",
    "中间": "middle",
    "随机": "random",
}


def _reuse_count(num_documents: int, ratio: float) -> int:
    if num_documents <= 0 or ratio <= 0:
        return 0
    return min(num_documents, max(1, math.ceil(num_documents * ratio)))


def select_reuse_indices(
    num_documents: int,
    ratio: float,
    pattern: str,
    rng: random.Random,
) -> List[int]:
    count = _reuse_count(num_documents, ratio)
    if count == 0:
        return []
    if pattern == "forward":
        return list(range(count))
    if pattern == "backward":
        return list(range(num_documents - count, num_documents))
    if pattern == "middle":
        start = max(0, (num_documents - count) // 2)
        return list(range(start, start + count))
    if pattern == "random":
        return sorted(rng.sample(range(num_documents), count))
    raise ValueError(f"Unsupported reuse pattern: {pattern}")


def extract_documents_once(
    example: dict,
    base_url: str,
    documents: List[str],
    compression_ratio: int,
) -> List[Dict[str, Any]]:
    doc_messages: List[Dict[str, Any]] = []
    for doc in documents:
        doc_message: Dict[str, Any] = {"role": "user", "content": doc}
        try:
            result = extract_document(base_url, doc, compression_ratio, role="user")
            if result.get("success"):
                doc_message["c2kv_extracted"] = True
            else:
                warnings.warn(f"[{example['qid']}] extract failed: {result.get('error')}")
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            warnings.warn(f"[{example['qid']}] extract error: {exc}")
        doc_messages.append(doc_message)
    return doc_messages


def build_mixed_messages(
    system_prompt: Optional[str],
    doc_messages: Sequence[Dict[str, Any]],
    question: str,
    reuse_indices: Sequence[int],
    c2kv_reuse: str,
) -> List[Dict[str, Any]]:
    reuse_index_set = set(reuse_indices)
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for index, doc_message in enumerate(doc_messages):
        mixed_message = {
            "role": doc_message["role"],
            "content": doc_message["content"],
        }
        if index in reuse_index_set and doc_message.get("c2kv_extracted"):
            mixed_message["c2kv_reuse"] = c2kv_reuse
        messages.append(mixed_message)
    messages.append({"role": "user", "content": question})
    return messages


def _process_example(
    example: dict,
    base_url: str,
    model: str,
    cut_length: Optional[int],
    compression_ratio: int,
    default_max_new_tokens: int,
    default_system_prompt: Optional[str],
    metric_fn,
    reuse_patterns: Sequence[str],
    reuse_ratios: Sequence[float],
    random_trials: int,
    seed: int,
    c2kv_reuse: str,
) -> List[dict]:
    system_prompt = example.get("system_prompt", default_system_prompt)
    max_new_tokens = example.get("max_new_tokens", default_max_new_tokens)
    documents = cut_documents(example["documents"], max_length=cut_length)
    doc_messages = extract_documents_once(example, base_url, documents, compression_ratio)

    records: List[dict] = []
    for pattern in reuse_patterns:
        trials = random_trials if pattern == "random" else 1
        for ratio in reuse_ratios:
            for trial in range(trials):
                rng = random.Random(f"{seed}:{example['qid']}:{pattern}:{ratio}:{trial}")
                reuse_indices = select_reuse_indices(len(documents), ratio, pattern, rng)
                messages = build_mixed_messages(
                    system_prompt,
                    doc_messages,
                    example["question"],
                    reuse_indices,
                    c2kv_reuse,
                )
                pred = chat_completion(base_url, model, messages, max_new_tokens)
                em_score = safe_metric(metric_fn, pred, example["answer"], example["qid"])
                reused_indices = [
                    index
                    for index in reuse_indices
                    if doc_messages[index].get("c2kv_extracted")
                ]
                records.append(
                    {
                        "qid": example["qid"],
                        "reuse_pattern": pattern,
                        "reuse_ratio": ratio,
                        "random_trial": trial if pattern == "random" else None,
                        "num_documents": len(documents),
                        "requested_reuse_indices": reuse_indices,
                        "reused_indices": reused_indices,
                        "reused_documents": len(reused_indices),
                        "prediction": pred,
                        "ground_truth": example["answer"],
                        "em_score": em_score,
                    }
                )
    return records


def _parse_reuse_patterns(value: str) -> List[str]:
    if value == "all":
        return list(REUSE_PATTERNS)
    patterns = [
        REUSE_PATTERN_ALIASES.get(item.strip(), item.strip())
        for item in value.split(",")
        if item.strip()
    ]
    unknown = sorted(set(patterns) - set(REUSE_PATTERNS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown reuse pattern(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(REUSE_PATTERNS)} or all."
        )
    return patterns


def _parse_reuse_ratios(value: str) -> List[float]:
    ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not ratios:
        raise argparse.ArgumentTypeError("At least one reuse ratio is required")
    for ratio in ratios:
        if ratio <= 0 or ratio > 1:
            raise argparse.ArgumentTypeError(f"Reuse ratio must be in (0, 1], got {ratio}")
    return ratios


def evaluate_via_api(args: argparse.Namespace, dataset: AbstractMDQADataset) -> Dict[str, float]:
    base_url = args.base_url.rstrip("/")
    num_examples = len(dataset) if args.max_examples is None else min(args.max_examples, len(dataset))

    dataset.system_prompt = "You are a helpful assistant."
    default_system_prompt = dataset.system_prompt
    default_max_new_tokens = dataset.max_new_tokens
    examples = [dataset[i] for i in range(num_examples)]

    records_by_example: List[Optional[List[dict]]] = [None] * num_examples
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_example,
                example,
                base_url,
                args.model,
                args.cut_length,
                args.compression_ratio,
                default_max_new_tokens,
                default_system_prompt,
                dataset.metric,
                args.reuse_patterns,
                args.reuse_ratios,
                args.random_trials,
                args.seed,
                args.c2kv_reuse,
            ): i
            for i, example in enumerate(examples)
        }
        total_runs = sum(
            len(args.reuse_ratios) * (args.random_trials if pattern == "random" else 1)
            for pattern in args.reuse_patterns
        )
        with tqdm(total=num_examples * total_runs) as pbar:
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                records = future.result()
                records_by_example[i] = records
                pbar.update(len(records))

    records = [record for example_records in records_by_example for record in (example_records or [])]
    exact_match = sum(record["em_score"] for record in records) / len(records) if records else 0.0

    summary_by_setting: Dict[str, Dict[str, float]] = {}
    for pattern in args.reuse_patterns:
        for ratio in args.reuse_ratios:
            setting_records = [
                record
                for record in records
                if record["reuse_pattern"] == pattern and record["reuse_ratio"] == ratio
            ]
            if not setting_records:
                continue
            key = f"{pattern}:{ratio:g}"
            summary_by_setting[key] = {
                "num_runs": len(setting_records),
                "exact_match": sum(record["em_score"] for record in setting_records) / len(setting_records),
            }

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary: Dict[str, Any] = {
            "base_url": base_url,
            "model": args.model,
            "dataset": dataset.__class__.__name__,
            "num_examples": num_examples,
            "num_runs": len(records),
            "exact_match": exact_match,
            "compression_ratio": args.compression_ratio,
            "c2kv_reuse": args.c2kv_reuse,
            "reuse_patterns": args.reuse_patterns,
            "reuse_ratios": args.reuse_ratios,
            "random_trials": args.random_trials,
            "by_setting": summary_by_setting,
        }
        summary_file = args.output_file.replace(".jsonl", "_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return {"exact_match": exact_match, "num_examples": num_examples, "num_runs": len(records)}


def main():
    parser = argparse.ArgumentParser(
        description="Test mixed C2KV reuse and non-reuse documents via HTTP API"
    )
    parser.add_argument("--base-url", type=str, default="http://localhost:30000")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--only_supporting", action="store_true")
    parser.add_argument("--cot", action="store_true", default=False)
    parser.add_argument("--cut_length", type=int, default=None)
    parser.add_argument("--compression-ratio", type=int, default=4, dest="compression_ratio")
    parser.add_argument(
        "--c2kv-reuse",
        choices=("required", "best_effort"),
        default="required",
        help="Reuse policy placed on successfully extracted text messages (default: required)",
    )
    parser.add_argument(
        "--reuse-pattern",
        type=_parse_reuse_patterns,
        default=list(REUSE_PATTERNS),
        dest="reuse_patterns",
        help="Reuse document selection pattern: all, or comma-separated forward,backward,middle,random",
    )
    parser.add_argument(
        "--reuse-ratios",
        type=_parse_reuse_ratios,
        default=list(REUSE_RATIOS),
        help="Comma-separated reuse ratios, default: 1,0.75,0.5,0.25",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=1,
        help="Number of random selections per ratio when reuse-pattern includes random",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()
    if args.random_trials < 1:
        raise ValueError("--random-trials must be >= 1")
    if args.output_file and os.path.exists(args.output_file):
        print(f"Output file {args.output_file} already exists, skipping")
        return

    dataset = load_mdoc_dataset(
        args.dataset,
        args.dataset_path,
        only_supporting=args.only_supporting,
        enable_cot=args.cot,
    )
    print(f"Loaded {len(dataset)} examples from {args.dataset} dataset")

    results = evaluate_via_api(args, dataset)
    print("\nMixed Reuse Evaluation Results:")
    print(f"Base URL: {args.base_url}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Number of runs: {results['num_runs']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
