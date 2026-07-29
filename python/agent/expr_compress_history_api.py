from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.dataset import (
        AgentDataset,
        AgentExample,
        extract_tool_calls,
        load_agent_dataset,
    )
    from agent.eval_utils import example_result, print_example
    from agent.expr_agent_c2kv_api import (
        extract_segment,
        parse_token_range,
        response_to_prediction,
        token_count_in_range,
    )
else:
    from .dataset import AgentDataset, AgentExample, extract_tool_calls, load_agent_dataset
    from .eval_utils import example_result, print_example
    from .expr_agent_c2kv_api import (
        extract_segment,
        parse_token_range,
        response_to_prediction,
        token_count_in_range,
    )


REUSE_PATTERNS = ("forward", "random")
REUSE_RATIOS = (1.0, 0.75, 0.5, 0.25)
DROP_RATIOS = (0.75, 0.5, 0.25)


ResultKey = Tuple[Any, ...]


def _json_compatible(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    return value


def evaluation_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """Return the parameters that determine selection or generated records."""
    dataset_path = str(Path(args.dataset_path).expanduser().resolve())
    tokenizer = args.tokenizer
    if tokenizer is not None:
        tokenizer = str(Path(tokenizer).expanduser().resolve())
    return _json_compatible(
        {
            "base_url": args.base_url.rstrip("/"),
            "model": args.model,
            "tokenizer": tokenizer,
            "dataset": args.dataset,
            "dataset_path": dataset_path,
            "max_examples": args.max_examples,
            "max_samples": args.max_samples,
            "max_samples_per_trace": args.max_samples_per_trace,
            "max_tools": args.max_tools,
            "benchmark": args.benchmark,
            "history_message_range": args.history_message_range,
            "request_token_range": args.request_token_range,
            "max_new_tokens": args.max_new_tokens,
            "compression_ratio": args.compression_ratio,
            "c2kv_reuse": args.c2kv_reuse,
            "reuse_patterns": args.reuse_patterns,
            "reuse_ratios": args.reuse_ratios,
            "random_trials": args.random_trials,
            "include_no_reuse": args.include_no_reuse,
            "include_drop_prefix": args.include_drop_prefix,
            "drop_ratios": args.drop_ratios,
            "seed": args.seed,
            "temperature": args.temperature,
            "profile": args.profile,
            "save_inputs": args.save_inputs,
        }
    )


def _legacy_summary_parameters(summary: Dict[str, Any]) -> Dict[str, Any]:
    legacy_keys = (
        "base_url",
        "compression_ratio",
        "reuse_cache_size",
        "history_message_range",
        "request_token_range",
        "reuse_patterns",
        "reuse_ratios",
        "random_trials",
        "include_no_reuse",
        "include_drop_prefix",
        "drop_ratios",
    )
    parameters = {
        key: summary[key]
        for key in legacy_keys
        if key in summary
    }
    if "model" in summary:
        parameters["model"] = summary["model"]
    return parameters


def _parameter_differences(
    summary: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> List[str]:
    if summary is None:
        return ["summary 文件不存在或无法读取"]
    saved = summary.get("parameters")
    if isinstance(saved, dict):
        keys = sorted(set(saved) | set(current))
    else:
        saved = _legacy_summary_parameters(summary)
        if not saved:
            return ["summary 中没有可比较的参数"]
        keys = sorted(saved)
    return [
        f"{key}: 已保存={saved.get(key)!r}, 当前={current.get(key)!r}"
        for key in keys
        if saved.get(key) != current.get(key)
    ]


def _prompt_existing_output(output_path: Path, parameters_match: bool) -> str:
    if parameters_match:
        prompt = (
            f"输出文件 {output_path} 已存在且参数一致，"
            "请选择 [o]覆盖 / [c]继续 / [e]退出: "
        )
        choices = {
            "o": "overwrite",
            "overwrite": "overwrite",
            "覆盖": "overwrite",
            "c": "continue",
            "continue": "continue",
            "继续": "continue",
            "e": "exit",
            "exit": "exit",
            "退出": "exit",
        }
    else:
        prompt = (
            f"输出文件 {output_path} 已存在但参数不一致，"
            "请选择 [o]覆盖 / [e]退出: "
        )
        choices = {
            "o": "overwrite",
            "overwrite": "overwrite",
            "覆盖": "overwrite",
            "e": "exit",
            "exit": "exit",
            "退出": "exit",
        }

    while True:
        try:
            choice = input(prompt).strip().lower()
        except EOFError:
            print("未收到选择，退出以避免覆盖已有结果。")
            return "exit"
        if choice in choices:
            return choices[choice]
        print("无效选择，请重新输入。")


def _load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"无法读取 summary {path}: {exc}")
        return None
    if not isinstance(value, dict):
        warnings.warn(f"summary {path} 的顶层不是 JSON 对象")
        return None
    return value


def _load_result_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"无法解析 {path} 第 {line_number} 行: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
            records.append(record)
    return records


def prepare_existing_output(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.output_file:
        return []
    output_path = Path(args.output_file)
    if not output_path.exists():
        return []

    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    differences = _parameter_differences(
        _load_json_file(summary_path) if summary_path.exists() else None,
        evaluation_parameters(args),
    )
    if differences:
        print("检测到以下参数差异:")
        for difference in differences:
            print(f"  - {difference}")
    action = _prompt_existing_output(output_path, not differences)
    if action == "exit":
        raise SystemExit(0)
    if action == "overwrite":
        return []

    records = _load_result_records(output_path)
    completed = sum(bool(record.get("prediction")) for record in records)
    empty = len(records) - completed
    print(
        f"已读取 {len(records)} 条已有结果，其中 {completed} 条可跳过，"
        f"{empty} 条 prediction 为空、需要重测。"
    )
    return records


def result_key(record: Dict[str, Any]) -> Optional[ResultKey]:
    qid = record.get("qid")
    experiment_type = record.get("experiment_type")
    if qid is None or experiment_type is None:
        return None
    if experiment_type == "no_reuse":
        return (qid, experiment_type)
    if experiment_type == "drop_prefix":
        return (qid, experiment_type, record.get("drop_ratio"))
    if experiment_type == "mixed_reuse":
        return (
            qid,
            experiment_type,
            record.get("reuse_pattern"),
            record.get("reuse_ratio"),
            record.get("random_trial"),
        )
    return None


def completed_result_keys(records: Sequence[Dict[str, Any]]) -> Set[ResultKey]:
    nonempty_keys = {
        key
        for record in records
        if record.get("prediction") and (key := result_key(record)) is not None
    }
    empty_keys = {
        key
        for record in records
        if not record.get("prediction") and (key := result_key(record)) is not None
    }
    return nonempty_keys - empty_keys


def merge_results(
    existing_records: Sequence[Dict[str, Any]],
    new_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    replaced_keys = {
        key
        for record in new_records
        if (key := result_key(record)) is not None
    }
    return [
        record
        for record in existing_records
        if result_key(record) not in replaced_keys
    ] + list(new_records)


def _run_key(
    qid: str,
    experiment_type: str,
    *,
    pattern: Optional[str] = None,
    ratio: Optional[float] = None,
    trial: Optional[int] = None,
) -> ResultKey:
    if experiment_type == "no_reuse":
        return (qid, experiment_type)
    if experiment_type == "drop_prefix":
        return (qid, experiment_type, ratio)
    return (qid, experiment_type, pattern, ratio, trial)


def pending_run_count(
    example: AgentExample,
    args: argparse.Namespace,
    completed_keys: Set[ResultKey],
) -> int:
    count = 0
    if args.include_no_reuse:
        count += _run_key(example.qid, "no_reuse") not in completed_keys
    if args.include_drop_prefix:
        count += sum(
            _run_key(example.qid, "drop_prefix", ratio=ratio) not in completed_keys
            for ratio in args.drop_ratios
        )
    for pattern in args.reuse_patterns:
        trials = args.random_trials if pattern == "random" else 1
        count += sum(
            _run_key(
                example.qid,
                "mixed_reuse",
                pattern=pattern,
                ratio=ratio,
                trial=trial if pattern == "random" else None,
            )
            not in completed_keys
            for ratio in args.reuse_ratios
            for trial in range(trials)
        )
    return count


def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    import requests

    url = f"{base_url}/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_new_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        warnings.warn(f"chat completion error: {exc}")
        return {"choices": [{"message": {"content": ""}}]}


def _reuse_count(num_messages: int, ratio: float) -> int:
    if num_messages <= 0 or ratio <= 0:
        return 0
    return min(num_messages, max(1, math.ceil(num_messages * ratio)))


def select_reuse_indices(
    num_messages: int,
    ratio: float,
    pattern: str,
    rng: random.Random,
) -> List[int]:
    count = _reuse_count(num_messages, ratio)
    if count == 0:
        return []
    if pattern == "forward":
        return list(range(count))
    if pattern == "random":
        return sorted(rng.sample(range(num_messages), count))
    raise ValueError(f"Unsupported reuse pattern: {pattern}")


def extract_role(role: str) -> str:
    return role if role in {"system", "user", "assistant"} else "user"


def split_protected_messages(
    example: AgentExample,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    user_indices = [
        index
        for index, message in enumerate(example.messages)
        if message.get("role") == "user"
    ]
    if not user_indices:
        return [], list(example.messages), []
    first_user_index = user_indices[0]
    last_user_index = user_indices[-1]
    if first_user_index == last_user_index:
        return (
            list(example.messages[: first_user_index + 1]),
            list(example.messages[first_user_index + 1 :]),
            [],
        )
    return (
        list(example.messages[: first_user_index + 1]),
        list(example.messages[first_user_index + 1 : last_user_index]),
        list(example.messages[last_user_index:]),
    )


def protected_chat_prefix(example: AgentExample) -> List[Dict[str, Any]]:
    first_user_messages, _, _ = split_protected_messages(example)
    return [
        {"role": "system", "content": example.system_prompt},
        *first_user_messages,
    ]


def protected_chat_suffix(example: AgentExample) -> List[Dict[str, Any]]:
    _, _, final_user_messages = split_protected_messages(example)
    return final_user_messages


def protected_history_messages(example: AgentExample) -> List[Dict[str, Any]]:
    _, history, _ = split_protected_messages(example)
    return history


def protected_flags() -> Dict[str, bool]:
    return {
        "preserve_first_user_message": True,
        "preserve_final_user_message": True,
    }


def _history_chat_message(message: Dict[str, Any]) -> Dict[str, Any]:
    chat_message = {
        "role": message["role"],
        "content": message.get("content", ""),
    }
    for key in ("tool_calls", "tool_call_id", "name"):
        if key in message:
            chat_message[key] = message[key]
    return chat_message


def build_chat_messages(
    example: AgentExample,
    extracted_history: Sequence[Dict[str, Any]],
    reuse_indices: Sequence[int],
    c2kv_reuse: str = "required",
) -> List[Dict[str, Any]]:
    messages = protected_chat_prefix(example)
    reuse_index_set = set(reuse_indices)
    for index, message in enumerate(extracted_history):
        chat_message = _history_chat_message(message)
        if index in reuse_index_set and message.get("c2kv_extracted"):
            chat_message["c2kv_reuse"] = c2kv_reuse
        messages.append(chat_message)
    messages.extend(protected_chat_suffix(example))
    return messages


def _drop_count(num_messages: int, ratio: float) -> int:
    if num_messages <= 0 or ratio <= 0:
        return 0
    return min(num_messages, math.ceil(num_messages * ratio))


def build_drop_prefix_messages(
    example: AgentExample,
    extracted_history: Sequence[Dict[str, Any]],
    drop_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    drop_count = _drop_count(len(extracted_history), drop_ratio)
    dropped_indices = list(range(drop_count))
    kept_indices = list(range(drop_count, len(extracted_history)))
    messages = protected_chat_prefix(example)
    for message in extracted_history[drop_count:]:
        messages.append(_history_chat_message(message))
    messages.extend(protected_chat_suffix(example))
    return messages, dropped_indices, kept_indices


def extract_history_once(
    example: AgentExample,
    base_url: str,
    compression_ratio: int,
) -> Tuple[List[Dict[str, Any]], float]:
    extracted: List[Dict[str, Any]] = []
    total_extract_time = 0.0
    for message in protected_history_messages(example):
        message_role = message.get("role", "user")
        role = extract_role(message_role)
        content = message.get("content", "")
        extracted_message: Dict[str, Any] = {
            "role": message.get("role", role),
            "content": content,
        }
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in message:
                extracted_message[key] = message[key]
        is_plain_text_message = (
            message_role in {"system", "user", "assistant"}
            and isinstance(content, str)
            and bool(content)
            and not any(
                key in message for key in ("tool_calls", "tool_call_id", "name")
            )
        )
        if not is_plain_text_message:
            extracted.append(extracted_message)
            continue

        t0 = time.perf_counter()
        result = extract_segment(base_url, content, compression_ratio, role=role)
        total_extract_time += time.perf_counter() - t0
        if result.get("success"):
            extracted_message["c2kv_extracted"] = True
            extracted_message["c2kv_diagnostic_key_hash"] = result.get("key_hash")
        else:
            warnings.warn(f"[{example.qid}] history extract failed: {result.get('error')}")
        extracted.append(extracted_message)
    return extracted, total_extract_time


def render_request_for_count(
    tokenizer: Any,
    example: AgentExample,
) -> str:
    chat_messages = [
        {"role": "system", "content": example.system_prompt},
        *example.messages,
    ]
    return tokenizer.apply_chat_template(
        chat_messages,
        tools=example.tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def count_message_tokens(tokenizer: Any, message: Dict[str, Any]) -> int:
    rendered = tokenizer.apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def token_sum(indices: Sequence[int], token_lengths: Sequence[int]) -> int:
    return sum(token_lengths[index] for index in indices)


def token_ratio(tokens: int, request_tokens: int) -> float:
    if request_tokens <= 0:
        return 0.0
    return tokens / request_tokens


def _normalize_metric_call(call: Any) -> Optional[Dict[str, Any]]:
    if isinstance(call, list):
        call = call[0] if call else None
    if not isinstance(call, dict):
        return None
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "name": function.get("name", ""),
        "arguments": arguments,
    }


def _prediction_metric_call(prediction: str) -> Optional[Dict[str, Any]]:
    calls = extract_tool_calls(prediction or "")
    return _normalize_metric_call(calls)


def _range_overlap(left: Any, right: Any) -> Optional[float]:
    if not (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == 2
        and len(right) == 2
        and all(isinstance(value, int) for value in [*left, *right])
    ):
        return None
    left_start, left_end = left
    right_start, right_end = right
    if left_end < left_start or right_end < right_start:
        return 0.0
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    union = max(left_end, right_end) - min(left_start, right_start) + 1
    return intersection / union if union > 0 else 0.0


def _soft_tool_metrics(
    prediction_call: Optional[Dict[str, Any]],
    reference_call: Optional[Dict[str, Any]],
    prefix: str = "",
) -> Dict[str, Any]:
    parse_success = prediction_call is not None
    if prediction_call is None or reference_call is None:
        return {
            f"{prefix}parse_success": float(parse_success),
            f"{prefix}tool_name_match": 0.0,
            f"{prefix}tool_action_match": 0.0,
            f"{prefix}tool_target_match": 0.0,
            f"{prefix}soft_tool_score": 0.0,
        }

    predicted_args = prediction_call["arguments"]
    reference_args = reference_call["arguments"]
    name_match = prediction_call["name"] == reference_call["name"]
    action_match = (
        predicted_args.get("command") == reference_args.get("command")
        if ("command" in predicted_args or "command" in reference_args)
        else True
    )
    target_match = (
        predicted_args.get("path") == reference_args.get("path")
        if ("path" in predicted_args or "path" in reference_args)
        else True
    )
    range_overlap = _range_overlap(
        predicted_args.get("view_range"),
        reference_args.get("view_range"),
    )
    if range_overlap is None:
        residual_predicted = {
            key: value
            for key, value in predicted_args.items()
            if key not in {"command", "path"}
        }
        residual_reference = {
            key: value
            for key, value in reference_args.items()
            if key not in {"command", "path"}
        }
        detail_score = 1.0 if residual_predicted == residual_reference else 0.0
    else:
        detail_score = range_overlap

    soft_score = 0.0
    if name_match:
        soft_score = (
            0.4
            + 0.2 * float(action_match)
            + 0.25 * float(target_match)
            + 0.15 * detail_score
        )

    metrics: Dict[str, Any] = {
        f"{prefix}parse_success": 1.0,
        f"{prefix}tool_name_match": float(name_match),
        f"{prefix}tool_action_match": float(action_match),
        f"{prefix}tool_target_match": float(target_match),
        f"{prefix}soft_tool_score": round(min(soft_score, 1.0), 6),
    }
    if range_overlap is not None:
        metrics[f"{prefix}view_range_overlap"] = round(range_overlap, 6)
    return metrics


def add_soft_tool_metrics(record: Dict[str, Any]) -> None:
    prediction_call = _prediction_metric_call(record.get("prediction", ""))
    reference_call = _normalize_metric_call(record.get("expected_tool_calls"))
    record.update(_soft_tool_metrics(prediction_call, reference_call))


def add_behavior_metrics(results: List[Dict[str, Any]]) -> None:
    if any(record.get("expected_answer") is not None for record in results):
        baseline_predictions = {
            record["qid"]: record.get("prediction", "")
            for record in results
            if record.get("experiment_type") == "no_reuse"
        }
        for record in results:
            baseline_prediction = baseline_predictions.get(record["qid"])
            if record.get("experiment_type") == "no_reuse" or baseline_prediction is None:
                continue
            record["behavior_exact_match"] = float(
                (record.get("prediction") or "").strip().lower()
                == (baseline_prediction or "").strip().lower()
            )
        return

    baseline_calls = {
        record["qid"]: _prediction_metric_call(record.get("prediction", ""))
        for record in results
        if record.get("experiment_type") == "no_reuse"
    }
    for record in results:
        baseline_call = baseline_calls.get(record["qid"])
        if record.get("experiment_type") == "no_reuse" or baseline_call is None:
            continue
        prediction_call = _prediction_metric_call(record.get("prediction", ""))
        record["behavior_exact_match"] = float(prediction_call == baseline_call)
        record.update(
            _soft_tool_metrics(
                prediction_call,
                baseline_call,
                prefix="behavior_",
            )
        )


def _mean_metric(records: Sequence[Dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return sum(values) / len(values) if values else 0.0


def _metric_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    has_answer = any(record.get("expected_answer") is not None for record in records)
    if has_answer:
        return {
            "accuracy": _mean_metric(records, "score"),
            "behavior_exact_match": _mean_metric(records, "behavior_exact_match"),
        }
    return {
        "tool_call_accuracy": _mean_metric(records, "score"),
        "parse_rate": _mean_metric(records, "parse_success"),
        "tool_name_match": _mean_metric(records, "tool_name_match"),
        "tool_action_match": _mean_metric(records, "tool_action_match"),
        "tool_target_match": _mean_metric(records, "tool_target_match"),
        "soft_tool_score": _mean_metric(records, "soft_tool_score"),
        "behavior_exact_match": _mean_metric(records, "behavior_exact_match"),
        "behavior_tool_name_match": _mean_metric(records, "behavior_tool_name_match"),
        "behavior_soft_tool_score": _mean_metric(records, "behavior_soft_tool_score"),
    }


def add_record_metrics(record: Dict[str, Any]) -> None:
    if record.get("expected_answer") is None:
        add_soft_tool_metrics(record)


def save_compress_history_results(
    output_file: Optional[str],
    model_name: str,
    dataset: AgentDataset,
    results: List[Dict[str, Any]],
    statistics: Dict[str, Any],
) -> Dict[str, Any]:
    mixed_results = [
        record
        for record in results
        if record.get("experiment_type") == "mixed_reuse"
    ]
    no_reuse_results = [
        record
        for record in results
        if record.get("experiment_type") == "no_reuse"
    ]
    drop_prefix_results = [
        record
        for record in results
        if record.get("experiment_type") == "drop_prefix"
    ]
    drop_prefix_by_setting: Dict[str, Dict[str, Any]] = {}
    for ratio in statistics.get("drop_ratios", []):
        setting_records = [
            record
            for record in drop_prefix_results
            if record.get("drop_ratio") == ratio
        ]
        if setting_records:
            drop_prefix_by_setting[f"drop_prefix:{ratio:g}"] = {
                "num_runs": len(setting_records),
                **_metric_summary(setting_records),
                "dropped_token_ratio": sum(
                    record.get("dropped_token_ratio", 0.0)
                    for record in setting_records
                )
                / len(setting_records),
            }
    summary = {
        "model": model_name,
        "dataset": dataset.__class__.__name__,
        "num_examples": statistics["num_selected_examples"],
        "num_runs": len(results),
        **_metric_summary(results),
        "mixed_reuse": {
            "num_runs": len(mixed_results),
            **_metric_summary(mixed_results),
            "reused_token_ratio": (
                sum(record.get("reused_token_ratio", 0.0) for record in mixed_results)
                / len(mixed_results)
                if mixed_results
                else 0.0
            ),
            "requested_reuse_token_ratio": (
                sum(
                    record.get("requested_reuse_token_ratio", 0.0)
                    for record in mixed_results
                )
                / len(mixed_results)
                if mixed_results
                else 0.0
            ),
        },
        "no_reuse": {
            "num_runs": len(no_reuse_results),
            **_metric_summary(no_reuse_results),
            "reused_token_ratio": 0.0,
        },
        "drop_prefix": {
            "num_runs": len(drop_prefix_results),
            **_metric_summary(drop_prefix_results),
            "dropped_token_ratio": (
                sum(record.get("dropped_token_ratio", 0.0) for record in drop_prefix_results)
                / len(drop_prefix_results)
                if drop_prefix_results
                else 0.0
            ),
            "by_setting": drop_prefix_by_setting,
        },
        **statistics,
    }
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        summary_path = path.with_name(f"{path.stem}_summary.json")
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def _preprocess_example_for_selection(
    dataset_index: int,
    example: AgentExample,
    tokenizer: Any,
    history_message_range: Optional[Tuple[Optional[int], Optional[int]]],
    request_token_range: Optional[Tuple[Optional[int], Optional[int]]],
) -> Optional[Tuple[int, AgentExample, List[int], int]]:
    eligible_history = protected_history_messages(example)
    if not token_count_in_range(len(eligible_history), history_message_range):
        return None
    history_token_lengths = [
        count_message_tokens(tokenizer, message)
        for message in eligible_history
    ]

    request_text = render_request_for_count(tokenizer, example)
    request_tokens = len(tokenizer.encode(request_text, add_special_tokens=False))
    if not token_count_in_range(request_tokens, request_token_range):
        return None
    return dataset_index, example, history_token_lengths, request_tokens


def select_examples_with_preprocessing(
    args: argparse.Namespace,
    dataset: AgentDataset,
    tokenizer: Any,
) -> Tuple[List[Tuple[AgentExample, List[int], int]], int]:
    selected: List[Tuple[AgentExample, List[int], int]] = []
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
                    args.history_message_range,
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
                        args.history_message_range,
                        args.request_token_range,
                    )
                    for dataset_index, example in chunk_examples
                ]
                chunk_results = [future.result() for future in futures]

        for result in chunk_results:
            if result is None:
                continue
            _, example, history_token_lengths, request_tokens = result
            selected.append((example, history_token_lengths, request_tokens))
            if args.max_examples is not None and len(selected) >= args.max_examples:
                return selected, scanned
    return selected, scanned


def _process_example(
    output_index: int,
    total: int,
    example: AgentExample,
    history_token_lengths: List[int],
    request_tokens: int,
    args: argparse.Namespace,
    completed_keys: Set[ResultKey],
) -> Tuple[int, List[Dict[str, Any]], float, float]:
    if pending_run_count(example, args, completed_keys) == 0:
        return output_index, [], 0.0, 0.0

    base_url = args.base_url.rstrip("/")
    extracted_history, extract_time = extract_history_once(
        example,
        base_url,
        args.compression_ratio,
    )

    max_new_tokens = example.max_new_tokens or args.max_new_tokens
    records: List[Dict[str, Any]] = []
    chat_time = 0.0
    reusable_history_tokens = sum(history_token_lengths)
    no_reuse_key = _run_key(example.qid, "no_reuse")
    if args.include_no_reuse and no_reuse_key not in completed_keys:
        messages = build_chat_messages(
            example,
            extracted_history,
            reuse_indices=[],
            c2kv_reuse=args.c2kv_reuse,
        )
        t0 = time.perf_counter()
        response_json = chat_completion(
            base_url,
            args.model,
            messages,
            example.tools,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
        )
        chat_time += time.perf_counter() - t0
        prediction = response_to_prediction(response_json)
        if args.print_examples:
            print_example(output_index, total, example, prediction)

        timer = None
        if args.profile:
            timer = {
                "extract_total": round(extract_time, 4),
                "chat_total_so_far": round(chat_time, 4),
            }
        record = example_result(args.dataset_obj, example, prediction, timer)
        add_record_metrics(record)
        record.update(
            {
                "reuse_pattern": "none",
                "reuse_ratio": 0.0,
                "experiment_type": "no_reuse",
                "random_trial": None,
                **protected_flags(),
                "history_messages": len(extracted_history),
                "request_tokens": request_tokens,
                "reusable_history_tokens": reusable_history_tokens,
                "requested_reuse_indices": [],
                "requested_reuse_tokens": 0,
                "requested_reuse_token_ratio": 0.0,
                "reused_indices": [],
                "reused_messages": 0,
                "reused_tokens": 0,
                "reused_token_ratio": 0.0,
            }
        )
        if args.save_inputs:
            record["inputs"] = {
                "extract_history": [
                    {
                        "role": extract_role(message.get("role", "user")),
                        "content": message.get("content", ""),
                        "compression_ratio": args.compression_ratio,
                        "key_hash": message.get("c2kv_diagnostic_key_hash"),
                    }
                    for message in extracted_history
                ],
                "chat": {
                    "model": args.model,
                    "messages": messages,
                    "tools": example.tools,
                    "max_completion_tokens": max_new_tokens,
                    "temperature": args.temperature,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
            record["chat_response"] = response_json
        records.append(record)

    if args.include_drop_prefix:
        for drop_ratio in args.drop_ratios:
            drop_prefix_key = _run_key(
                example.qid,
                "drop_prefix",
                ratio=drop_ratio,
            )
            if drop_prefix_key in completed_keys:
                continue
            messages, dropped_indices, kept_indices = build_drop_prefix_messages(
                example,
                extracted_history,
                drop_ratio,
            )
            dropped_tokens = token_sum(dropped_indices, history_token_lengths)
            kept_tokens = token_sum(kept_indices, history_token_lengths)
            t0 = time.perf_counter()
            response_json = chat_completion(
                base_url,
                args.model,
                messages,
                example.tools,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
            )
            chat_time += time.perf_counter() - t0
            prediction = response_to_prediction(response_json)
            if args.print_examples:
                print_example(output_index, total, example, prediction)

            timer = None
            if args.profile:
                timer = {
                    "extract_total": round(extract_time, 4),
                    "chat_total_so_far": round(chat_time, 4),
                }
            record = example_result(args.dataset_obj, example, prediction, timer)
            add_record_metrics(record)
            record.update(
                {
                    "experiment_type": "drop_prefix",
                    "reuse_pattern": "drop_prefix",
                    "reuse_ratio": 0.0,
                    "drop_ratio": drop_ratio,
                    "random_trial": None,
                    **protected_flags(),
                    "history_messages": len(extracted_history),
                    "request_tokens": request_tokens,
                    "reusable_history_tokens": reusable_history_tokens,
                    "dropped_indices": dropped_indices,
                    "dropped_messages": len(dropped_indices),
                    "dropped_tokens": dropped_tokens,
                    "dropped_token_ratio": token_ratio(dropped_tokens, request_tokens),
                    "kept_indices": kept_indices,
                    "kept_messages": len(kept_indices),
                    "kept_tokens": kept_tokens,
                    "kept_token_ratio": token_ratio(kept_tokens, request_tokens),
                    "requested_reuse_indices": [],
                    "requested_reuse_tokens": 0,
                    "requested_reuse_token_ratio": 0.0,
                    "reused_indices": [],
                    "reused_messages": 0,
                    "reused_tokens": 0,
                    "reused_token_ratio": 0.0,
                }
            )
            if args.save_inputs:
                record["inputs"] = {
                    "chat": {
                        "model": args.model,
                        "messages": messages,
                        "tools": example.tools,
                        "max_completion_tokens": max_new_tokens,
                        "temperature": args.temperature,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
                record["chat_response"] = response_json
            records.append(record)

    for pattern in args.reuse_patterns:
        trials = args.random_trials if pattern == "random" else 1
        for ratio in args.reuse_ratios:
            for trial in range(trials):
                mixed_reuse_key = _run_key(
                    example.qid,
                    "mixed_reuse",
                    pattern=pattern,
                    ratio=ratio,
                    trial=trial if pattern == "random" else None,
                )
                if mixed_reuse_key in completed_keys:
                    continue
                rng = random.Random(f"{args.seed}:{example.qid}:{pattern}:{ratio}:{trial}")
                reuse_indices = select_reuse_indices(
                    len(extracted_history),
                    ratio,
                    pattern,
                    rng,
                )
                messages = build_chat_messages(
                    example,
                    extracted_history,
                    reuse_indices,
                    c2kv_reuse=args.c2kv_reuse,
                )
                t0 = time.perf_counter()
                response_json = chat_completion(
                    base_url,
                    args.model,
                    messages,
                    example.tools,
                    max_new_tokens=max_new_tokens,
                    temperature=args.temperature,
                )
                chat_time += time.perf_counter() - t0
                prediction = response_to_prediction(response_json)
                if args.print_examples:
                    print_example(output_index, total, example, prediction)

                reused_indices = [
                    index
                    for index in reuse_indices
                    if extracted_history[index].get("c2kv_extracted")
                ]
                requested_reuse_tokens = token_sum(reuse_indices, history_token_lengths)
                reused_tokens = token_sum(reused_indices, history_token_lengths)
                timer = None
                if args.profile:
                    timer = {
                        "extract_total": round(extract_time, 4),
                        "chat_total_so_far": round(chat_time, 4),
                    }
                record = example_result(args.dataset_obj, example, prediction, timer)
                add_record_metrics(record)
                record.update(
                    {
                        "reuse_pattern": pattern,
                        "reuse_ratio": ratio,
                        "experiment_type": "mixed_reuse",
                        "random_trial": trial if pattern == "random" else None,
                        **protected_flags(),
                        "history_messages": len(extracted_history),
                        "request_tokens": request_tokens,
                        "reusable_history_tokens": reusable_history_tokens,
                        "requested_reuse_indices": reuse_indices,
                        "requested_reuse_tokens": requested_reuse_tokens,
                        "requested_reuse_token_ratio": token_ratio(
                            requested_reuse_tokens,
                            request_tokens,
                        ),
                        "reused_indices": reused_indices,
                        "reused_messages": len(reused_indices),
                        "reused_tokens": reused_tokens,
                        "reused_token_ratio": token_ratio(reused_tokens, request_tokens),
                    }
                )
                if args.save_inputs:
                    record["inputs"] = {
                        "extract_history": [
                            {
                                "role": extract_role(message.get("role", "user")),
                                "content": message.get("content", ""),
                                "compression_ratio": args.compression_ratio,
                                "key_hash": message.get("c2kv_diagnostic_key_hash"),
                            }
                            for message in extracted_history
                        ],
                        "chat": {
                            "model": args.model,
                            "messages": messages,
                            "tools": example.tools,
                            "max_completion_tokens": max_new_tokens,
                            "temperature": args.temperature,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    }
                    record["chat_response"] = response_json
                records.append(record)
    return output_index, records, extract_time, chat_time


def _parse_reuse_patterns(value: str) -> List[str]:
    if value == "all":
        return list(REUSE_PATTERNS)
    patterns = [item.strip() for item in value.split(",") if item.strip()]
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


def _min_messages_from_history_range(
    history_message_range: Optional[Tuple[Optional[int], Optional[int]]],
) -> Optional[int]:
    if history_message_range is None:
        return None
    low, _ = history_message_range
    if low is None:
        return None
    # Open-SWE traces generally have one protected user message followed by
    # assistant/tool history, so this is an early prefilter. The exact
    # protected-history filter still runs later in preprocessing.
    return low + 1


def evaluate(
    args: argparse.Namespace,
    dataset: AgentDataset,
    existing_results: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
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
    selected, scanned = select_examples_with_preprocessing(args, dataset, tokenizer)
    total = len(selected)
    total_runs = sum(
        len(args.reuse_ratios) * (args.random_trials if pattern == "random" else 1)
        for pattern in args.reuse_patterns
    )
    if args.include_no_reuse:
        total_runs += 1
    if args.include_drop_prefix:
        total_runs += len(args.drop_ratios)
    print(
        f"Selected {total} examples from {scanned} scanned examples "
        f"(preserve_first_user_message=True, "
        f"preserve_final_user_message=True, "
        f"history_message_range={args.history_message_range}, "
        f"request_token_range={args.request_token_range}, "
        f"runs_per_example={total_runs})"
    )

    args.dataset_obj = dataset
    completed_keys = completed_result_keys(existing_results)
    pending_runs = [
        pending_run_count(example, args, completed_keys)
        for example, _, _ in selected
    ]
    scheduled_runs = sum(pending_runs)
    skipped_runs = total * total_runs - scheduled_runs
    if existing_results:
        print(f"续测将跳过 {skipped_runs} 个已有结果，执行 {scheduled_runs} 个测试。")
    results_by_example: List[Optional[List[Dict[str, Any]]]] = [None] * total
    extract_times: List[float] = []
    chat_times: List[float] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_example,
                output_index,
                total,
                example,
                history_token_lengths,
                request_tokens,
                args,
                completed_keys,
            ): output_index
            for output_index, (
                example,
                history_token_lengths,
                request_tokens,
            ) in enumerate(selected)
            if pending_runs[output_index] > 0
        }
        with tqdm(total=scheduled_runs, desc="Agent compress history API") as progress:
            for future in as_completed(futures):
                output_index, records, extract_time, chat_time = future.result()
                results_by_example[output_index] = records
                if args.profile:
                    extract_times.append(extract_time)
                    chat_times.append(chat_time)
                progress.update(len(records))

    new_results = [
        record
        for example_records in results_by_example
        for record in (example_records or [])
    ]
    final_results = merge_results(existing_results, new_results)
    add_behavior_metrics(final_results)
    by_setting: Dict[str, Dict[str, Any]] = {}
    for pattern in args.reuse_patterns:
        for ratio in args.reuse_ratios:
            setting_records = [
                record
                for record in final_results
                if (
                    record.get("experiment_type") == "mixed_reuse"
                    and record["reuse_pattern"] == pattern
                    and record["reuse_ratio"] == ratio
                )
            ]
            if setting_records:
                key = f"{pattern}:{ratio:g}"
                by_setting[key] = {
                    "num_runs": len(setting_records),
                    **_metric_summary(setting_records),
                    "reused_token_ratio": sum(
                        record.get("reused_token_ratio", 0.0)
                        for record in setting_records
                    )
                    / len(setting_records),
                    "requested_reuse_token_ratio": sum(
                        record.get("requested_reuse_token_ratio", 0.0)
                        for record in setting_records
                    )
                    / len(setting_records),
                }

    statistics: Dict[str, Any] = {
        "base_url": args.base_url.rstrip("/"),
        "compression_ratio": args.compression_ratio,
        "c2kv_reuse": args.c2kv_reuse,
        **protected_flags(),
        "history_message_range": args.history_message_range,
        "request_token_range": args.request_token_range,
        "reuse_patterns": args.reuse_patterns,
        "reuse_ratios": args.reuse_ratios,
        "random_trials": args.random_trials,
        "include_no_reuse": args.include_no_reuse,
        "include_drop_prefix": args.include_drop_prefix,
        "drop_ratios": args.drop_ratios,
        "num_selected_examples": total,
        "num_runs": len(final_results),
        "by_setting": by_setting,
        "parameters": evaluation_parameters(args),
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
    return save_compress_history_results(
        args.output_file,
        args.model,
        dataset,
        final_results,
        statistics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress agent conversation history via C2KV HTTP API"
    )
    parser.add_argument("--base-url", type=str, default="http://localhost:30000")
    parser.add_argument("--model", default="default")
    parser.add_argument(
        "--tokenizer",
        help="Local tokenizer/checkpoint path used to count and render requests",
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
        "--history-message-range",
        type=parse_token_range,
        help=(
            "Keep examples whose compressible middle-history message count is within a,b. "
            "The first and final user messages are always preserved and excluded."
        ),
    )
    parser.add_argument(
        "--request-token-range",
        type=parse_token_range,
        help="Keep examples whose full native chat request token count is within a,b.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-file")
    parser.add_argument("--compression-ratio", type=int, default=2)
    parser.add_argument(
        "--c2kv-reuse",
        choices=("required", "best_effort"),
        default="required",
        help="Reuse policy placed on successfully extracted text history messages",
    )
    parser.add_argument(
        "--reuse-pattern",
        type=_parse_reuse_patterns,
        default=list(REUSE_PATTERNS),
        dest="reuse_patterns",
        help="Reuse history selection pattern: all, or comma-separated forward,random",
    )
    parser.add_argument(
        "--reuse-ratios",
        type=_parse_reuse_ratios,
        default=list(REUSE_RATIOS),
        help="Comma-separated reuse ratios, default: 1,0.75,0.5,0.25",
    )
    parser.add_argument("--random-trials", type=int, default=4)
    parser.add_argument(
        "--include-no-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run one full-compute-history control per example and summarize it separately.",
    )
    parser.add_argument(
        "--include-drop-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run controls that drop the oldest eligible history messages without C2KV reuse.",
    )
    parser.add_argument(
        "--drop-ratios",
        type=_parse_reuse_ratios,
        default=list(DROP_RATIOS),
        help="Comma-separated oldest-history drop ratios, default: 0.75,0.5,0.25",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--preprocess-chunk-multiplier", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--print-examples", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_trials < 1:
        raise ValueError("--random-trials must be >= 1")
    existing_results = prepare_existing_output(args)
    dataset = load_agent_dataset(
        args.dataset,
        args.dataset_path,
        max_samples=args.max_samples,
        max_tools=args.max_tools,
        min_messages=_min_messages_from_history_range(args.history_message_range),
        max_samples_per_trace=args.max_samples_per_trace,
        max_new_tokens=args.max_new_tokens,
        benchmark=args.benchmark,
    )
    print(f"Loaded {len(dataset)} examples from {args.dataset}")
    summary = evaluate(args, dataset, existing_results)
    if "tool_call_accuracy" in summary:
        print(f"\nTool-call accuracy: {summary['tool_call_accuracy']:.4f}")
        print(f"Soft tool score: {summary['soft_tool_score']:.4f}")
        print(
            "Behavior preservation score: "
            f"{summary['behavior_soft_tool_score']:.4f}"
        )
    else:
        print(f"\nAccuracy: {summary['accuracy']:.4f}")
        print(
            "Behavior exact match: "
            f"{summary['behavior_exact_match']:.4f}"
        )
    if args.output_file:
        print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main()
