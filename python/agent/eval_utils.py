from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .dataset import AgentDataset, AgentExample
from .render import count_tool_tokens


def select_example_indices(
    dataset: AgentDataset,
    tokenizer: Any,
    max_examples: int | None = None,
    max_tool_tokens: int | None = None,
) -> List[int]:
    indices = []
    for index in range(len(dataset)):
        if (
            max_tool_tokens is not None
            and count_tool_tokens(tokenizer, dataset[index]) > max_tool_tokens
        ):
            continue
        indices.append(index)
        if max_examples is not None and len(indices) >= max_examples:
            break
    return indices


def example_result(
    dataset: AgentDataset,
    example: AgentExample,
    prediction: str,
    timer: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "qid": example.qid,
        "prediction": prediction,
        "expected_tool_calls": example.expected_tool_calls,
        "score": dataset.score(prediction, example),
    }
    if example.expected_answer is not None:
        result["expected_answer"] = example.expected_answer
    if timer is not None:
        result["timer"] = timer
    return result


def save_results(
    output_file: str | None,
    model_name: str,
    dataset: AgentDataset,
    results: List[Dict[str, Any]],
    statistics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    summary = {
        "model": model_name,
        "dataset": dataset.__class__.__name__,
        "num_examples": len(results),
        "tool_call_accuracy": (
            sum(result["score"] for result in results) / len(results)
            if results
            else 0.0
        ),
        **(statistics or {}),
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


def print_example(index: int, total: int, example: AgentExample, prediction: str) -> None:
    print(f"\n=========== Example {index + 1}/{total} ===========")
    print(f"QID: {example.qid}")
    print(f"System: \n{example.system_prompt}")
    print(f"Tools: \n{example.tools}")
    print(f"Messages: \n{example.messages}")
    print(f"Expected: \n{example.expected_tool_calls}")
    print(f"Prediction: \n{prediction}")
