import argparse
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

from mdocdataset import AbstractMDQADataset, load_mdoc_dataset


def cut_documents(documents: List[str], max_length: int | None) -> List[str]:
    if max_length is None:
        return documents
    docs = []
    for document in documents:
        last_document = ''
        for passage in document.split('\n\n'):
            if not passage.strip():
                continue
            if len(last_document) + len(passage) > max_length:
                docs.append(last_document)
                last_document = passage
            else:
                last_document += passage + '\n\n'
        if last_document:
            docs.append(last_document)
    return docs


def extract_document(base_url: str, text: str, compression_ratio: int = 4, role: str = "user") -> dict:
    url = f"{base_url}/v1/c2kv/extract"
    payload = {
        "text": text,
        "compression_ratio": compression_ratio,
        "role": role,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def chat_completion(base_url: str, model: str, messages: list, max_new_tokens: int, temperature: float = 0.0) -> str:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_new_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        warnings.warn(f"chat completion error: {exc}")
        return ""
    if content is None:
        warnings.warn("chat completion returned null content; using empty prediction")
        return ""
    if not isinstance(content, str):
        warnings.warn(f"chat completion returned non-string content ({type(content).__name__}); using empty prediction")
        return ""
    return content


def safe_metric(metric_fn, pred: str, ground_truth: Any, qid: Any) -> float:
    try:
        return metric_fn(pred or "", ground_truth)
    except Exception as exc:
        warnings.warn(f"[{qid}] metric error: {exc}")
        return 0.0


def _process_example(
    example: dict,
    base_url: str,
    model: str,
    cut_length: int | None,
    compression_ratio: int,
    c2kv_reuse: str,
    default_max_new_tokens: int,
    profile: bool,
    default_system_prompt: str | None,
    metric_fn,
) -> Tuple[dict, float, float]:
    system_prompt = example.get('system_prompt', default_system_prompt)
    max_new_tokens = example.get('max_new_tokens', default_max_new_tokens)
    documents = cut_documents(example['documents'], max_length=cut_length)

    # Extract documents sequentially
    t0 = time.perf_counter()
    doc_messages = []
    for doc in documents:
        try:
            result = extract_document(base_url, doc, compression_ratio, role="user")
            if result.get("success"):
                doc_messages.append(
                    {
                        "role": "user",
                        "content": doc,
                        "c2kv_reuse": c2kv_reuse,
                    }
                )
            else:
                warnings.warn(f"[{example['qid']}] extract failed: {result.get('error')}")
                doc_messages.append({"role": "user", "content": doc})
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            warnings.warn(f"[{example['qid']}] extract error: {e}")
            doc_messages.append({"role": "user", "content": doc})
    t_extract = time.perf_counter() - t0

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(doc_messages)
    messages.append({"role": "user", "content": example['question']})

    t1 = time.perf_counter()
    pred = chat_completion(base_url, model, messages, max_new_tokens)
    t_chat = time.perf_counter() - t1

    em_score = safe_metric(metric_fn, pred, example['answer'], example['qid'])

    record: Dict[str, Any] = {
        'qid': example['qid'],
        'prediction': pred,
        'ground_truth': example['answer'],
        'em_score': em_score,
    }
    if profile:
        record['timer'] = {'extract': round(t_extract, 4), 'chat': round(t_chat, 4)}

    return record, t_extract, t_chat


def evaluate_via_api(args: argparse.Namespace, dataset: AbstractMDQADataset) -> Dict[str, float]:
    base_url = args.base_url.rstrip("/")
    model = args.model
    max_examples = args.max_examples
    output_file = args.output_file
    cut_length = args.cut_length
    profile = args.profile
    compression_ratio = args.compression_ratio
    workers = args.workers

    num_examples = len(dataset) if max_examples is None else min(max_examples, len(dataset))

    dataset.system_prompt = "You are a helpful assistant."
    default_system_prompt = dataset.system_prompt
    default_max_new_tokens = dataset.max_new_tokens

    # Pre-fetch all examples (datasets may not be thread-safe for concurrent __getitem__)
    examples = [dataset[i] for i in range(num_examples)]

    records: List[Optional[dict]] = [None] * num_examples
    extract_times: List[float] = []
    chat_times: List[float] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                _process_example,
                example,
                base_url, model, cut_length, compression_ratio, args.c2kv_reuse,
                default_max_new_tokens, profile, default_system_prompt,
                dataset.metric,
            ): i
            for i, example in enumerate(examples)
        }
        with tqdm(total=num_examples) as pbar:
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                record, t_extract, t_chat = future.result()
                records[i] = record
                if profile:
                    extract_times.append(t_extract)
                    chat_times.append(t_chat)
                pbar.update(1)

    em_scores = [r['em_score'] for r in records]
    exact_match = sum(em_scores) / len(em_scores) if em_scores else 0.0

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        summary: Dict[str, Any] = {
            'base_url': base_url,
            'model': model,
            'dataset': dataset.__class__.__name__,
            'num_examples': len(records),
            'exact_match': exact_match,
            'compression_ratio': compression_ratio,
            'c2kv_reuse': args.c2kv_reuse,
        }
        if profile and extract_times:
            summary['extract_mean'] = round(float(np.mean(extract_times)), 4)
            summary['extract_std'] = round(float(np.std(extract_times)), 4)
            summary['chat_mean'] = round(float(np.mean(chat_times)), 4)
            summary['chat_std'] = round(float(np.std(chat_times)), 4)

        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

    return {'exact_match': exact_match, 'num_examples': len(records)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate C2KV inference engine on MDQA datasets via HTTP API")

    parser.add_argument("--base-url", type=str, default="http://localhost:30000",
                        help="SGLang server base URL (e.g. http://localhost:30000)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Type of dataset (musique, wikimqa, hotpotqa, …)")
    parser.add_argument("--model", type=str, default="default",
                        help="Model name passed in chat completions payload")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Dataset file path")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to evaluate")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSONL file path for results")
    parser.add_argument("--only_supporting", action="store_true",
                        help="For Musique dataset, use only supporting paragraphs")
    parser.add_argument("--cot", action="store_true", default=False,
                        help="Use chain-of-thought prompt")
    parser.add_argument("--cut_length", type=int, default=None,
                        help="Cut documents to at most this many characters")
    parser.add_argument("--compression-ratio", type=int, default=4, dest="compression_ratio",
                        help="Gist compression ratio passed to /v1/c2kv/extract (default: 4)")
    parser.add_argument(
        "--c2kv-reuse",
        choices=("required", "best_effort"),
        default="required",
        help="Reuse policy placed on successfully extracted text messages (default: required)",
    )
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Record extract and chat latencies per example")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent examples to process (default: 1)")

    args = parser.parse_args()

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

    print(f"\nEvaluation Results:")
    print(f"Base URL: {args.base_url}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of examples evaluated: {results['num_examples']}")
    print(f"Score: {results['exact_match']:.4f}")


if __name__ == "__main__":
    main()
