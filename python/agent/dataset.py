from __future__ import annotations

import json
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class AgentExample:
    qid: str
    system_prompt: str
    tools: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    expected_tool_calls: List[Dict[str, Any]]
    max_new_tokens: Optional[int] = None


class AgentDataset(ABC):
    """Base interface for agent evaluation datasets.

    Every example exposes the three inputs needed by an agent chat template:
    a system prompt, tool definitions, and conversation messages. Tool
    definitions are deliberately not encoded as ordinary message content.
    """

    default_system_prompt = DEFAULT_SYSTEM_PROMPT
    max_new_tokens = 128

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> AgentExample:
        raise NotImplementedError

    def score(self, prediction: str, example: AgentExample) -> float:
        predicted = extract_tool_calls(prediction)
        return float(
            canonical_tool_calls(predicted)
            == canonical_tool_calls(example.expected_tool_calls)
        )


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _message_parts(messages: Sequence[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for message in messages:
        yield from message.get("parts") or []


def normalize_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for tool in tools:
        if isinstance(tool.get("function"), dict):
            normalized.append(dict(tool))
            continue
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return normalized


def normalize_messages(
    messages: Sequence[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        parts = message.get("parts")
        if parts is None:
            content = message.get("content", "")
        else:
            content = "\n".join(
                str(part.get("content", ""))
                for part in parts
                if part.get("type") == "text"
            )
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if content or role in {"assistant", "tool"}:
            normalized.append({"role": role, "content": content})
    system_prompt = "\n".join(system_parts).strip() or DEFAULT_SYSTEM_PROMPT
    return system_prompt, normalized


def _normalize_call(call: Dict[str, Any]) -> Dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return {
        "name": function.get("name", ""),
        "arguments": arguments,
    }


def canonical_tool_calls(calls: Sequence[Dict[str, Any]]) -> str:
    return json.dumps(
        [_normalize_call(call) for call in calls],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    payloads = _TOOL_CALL_RE.findall(text)
    if not payloads:
        payloads = [text.strip()]
    calls = []
    for payload in payloads:
        try:
            call = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(call, dict):
            calls.append(call)
        elif isinstance(call, list):
            calls.extend(item for item in call if isinstance(item, dict))
    return calls


def _agent_example_to_record(example: AgentExample) -> Dict[str, Any]:
    return {
        "qid": example.qid,
        "system_prompt": example.system_prompt,
        "tools": json.dumps(example.tools, ensure_ascii=False),
        "messages": json.dumps(example.messages, ensure_ascii=False),
        "expected_tool_calls": json.dumps(
            example.expected_tool_calls,
            ensure_ascii=False,
        ),
        "max_new_tokens": example.max_new_tokens,
    }


def _record_to_agent_example(record: Dict[str, Any]) -> AgentExample:
    return AgentExample(
        qid=record["qid"],
        system_prompt=record["system_prompt"],
        tools=_json_loads(record["tools"], []),
        messages=_json_loads(record["messages"], []),
        expected_tool_calls=_json_loads(record["expected_tool_calls"], []),
        max_new_tokens=record.get("max_new_tokens"),
    )


def _normalize_benchmarks(
    benchmark: Optional[str | Sequence[str]],
) -> Optional[List[str]]:
    if benchmark is None:
        return None
    if isinstance(benchmark, str):
        values = [item.strip() for item in benchmark.split(",")]
    else:
        values = [str(item).strip() for item in benchmark]
    values = [item for item in values if item]
    return values or None


def _dataset_fingerprint(
    files: Sequence[Path],
    max_tools: Optional[int],
    max_input_tokens: Optional[int],
    max_new_tokens: int,
    benchmarks: Optional[Sequence[str]],
) -> str:
    payload = {
        "version": 3,
        "max_tools": max_tools,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "benchmarks": sorted(benchmarks) if benchmarks is not None else None,
        "files": [
            {
                "path": str(file.resolve()),
                "size": file.stat().st_size,
                "mtime_ns": file.stat().st_mtime_ns,
            }
            for file in files
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _open_swe_dataset_fingerprint(
    files: Sequence[Path],
    tool_files: Sequence[Path],
    max_tools: Optional[int],
    max_input_tokens: Optional[int],
    max_new_tokens: int,
    min_messages: Optional[int],
    max_samples_per_trace: Optional[int],
    benchmarks: Optional[Sequence[str]],
) -> str:
    payload = {
        "version": 1,
        "dataset": "open_swe_traces",
        "max_tools": max_tools,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "min_messages": min_messages,
        "max_samples_per_trace": max_samples_per_trace,
        "benchmarks": sorted(benchmarks) if benchmarks is not None else None,
        "files": [
            {
                "path": str(file.resolve()),
                "size": file.stat().st_size,
                "mtime_ns": file.stat().st_mtime_ns,
            }
            for file in files
        ],
        "tool_files": [
            {
                "path": str(file.resolve()),
                "size": file.stat().st_size,
                "mtime_ns": file.stat().st_mtime_ns,
            }
            for file in tool_files
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _iter_agent_llm_trace_records(
    files: Sequence[str],
    max_tools: Optional[int],
    max_input_tokens: Optional[int],
    max_new_tokens: int,
    benchmarks: Optional[Sequence[str]],
) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    benchmark_set = set(benchmarks) if benchmarks is not None else None
    for file_name in files:
        file = Path(file_name)
        table = pq.read_table(file)
        for row_index, row in enumerate(table.to_pylist()):
            row_benchmark = row.get("benchmark") or ""
            for span_index, span in enumerate(row.get("spans") or []):
                benchmark = span.get("benchmark") or row_benchmark
                if benchmark_set is not None and benchmark not in benchmark_set:
                    continue
                example = AgentLLMTracesDataset._extract_example(
                    row,
                    span,
                    f"{file.name}:{row_index}:{span_index}",
                    max_tools=max_tools,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                )
                if example is not None:
                    yield _agent_example_to_record(example)


def _load_open_swe_tools(data_path: Path, scaffold: str) -> List[Dict[str, Any]]:
    tool_file = data_path / f"{scaffold}_tools.json"
    if not tool_file.is_file():
        raise FileNotFoundError(f"Missing Open-SWE tool definition file: {tool_file}")
    return normalize_tools(json.loads(tool_file.read_text(encoding="utf-8")))


def _open_swe_scaffold(file: Path) -> str:
    parent = file.parent.name.lower()
    if "openhands" in parent:
        return "openhands"
    if "sweagent" in parent:
        return "sweagent"
    raise ValueError(f"Cannot infer Open-SWE scaffold from {file}")


def _open_swe_file_matches(
    file: Path,
    benchmarks: Optional[Sequence[str]],
) -> bool:
    if benchmarks is None:
        return True
    parent = file.parent.name.lower()
    candidates = {
        parent,
        parent.removesuffix("_trajectories"),
        file.stem.lower(),
    }
    parts = parent.removesuffix("_trajectories").split("_")
    candidates.update(parts)
    if parent.startswith("minimax_m25"):
        candidates.add("minimax_m25")
    if parent.startswith("qwen35"):
        candidates.add("qwen35_122b")
    if "openhands" in parent:
        candidates.add("openhands")
    if "sweagent" in parent:
        candidates.add("sweagent")
    return bool(candidates.intersection({item.lower() for item in benchmarks}))


def _estimate_open_swe_input_tokens(
    system_prompt: str,
    messages: Sequence[Dict[str, Any]],
) -> int:
    # Open-SWE does not include token counts. This rough filter keeps the
    # max_input_tokens option useful without adding a tokenizer dependency here.
    characters = len(system_prompt) + sum(
        len(str(message.get("content", ""))) for message in messages
    )
    return characters // 4


def _normalize_open_swe_history_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    normalized: Dict[str, Any] = {
        "type": call.get("type") or "function",
        "function": {
            "name": function.get("name", ""),
            "arguments": function.get("arguments", "{}"),
        },
    }
    if call.get("id"):
        normalized["id"] = call["id"]
    return normalized


def normalize_open_swe_messages(
    messages: Sequence[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    normalized: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""
        if role == "system":
            if content:
                system_parts.append(content)
            continue

        normalized_message: Dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if role == "assistant" and message.get("tool_calls"):
            normalized_message["tool_calls"] = [
                _normalize_open_swe_history_tool_call(call)
                for call in message.get("tool_calls") or []
                if isinstance(call, dict)
            ]
        if message.get("tool_call_id"):
            normalized_message["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            normalized_message["name"] = message["name"]

        if content or role in {"assistant", "tool"}:
            normalized.append(normalized_message)

    system_prompt = "\n".join(system_parts).strip() or DEFAULT_SYSTEM_PROMPT
    return system_prompt, normalized


def _iter_open_swe_trace_records(
    files: Sequence[str],
    data_path: str,
    max_tools: Optional[int],
    max_input_tokens: Optional[int],
    max_new_tokens: int,
    min_messages: Optional[int],
    max_samples_per_trace: Optional[int],
) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    root = Path(data_path)
    tools_by_scaffold: Dict[str, List[Dict[str, Any]]] = {}
    for file_name in files:
        file = Path(file_name)
        scaffold = _open_swe_scaffold(file)
        if scaffold not in tools_by_scaffold:
            tools_by_scaffold[scaffold] = _load_open_swe_tools(root, scaffold)
        tools = tools_by_scaffold[scaffold]
        if max_tools is not None and len(tools) > max_tools:
            continue

        table = pq.read_table(file)
        for row_index, row in enumerate(table.to_pylist()):
            for example in OpenSWETracesDataset._extract_examples(
                row,
                tools,
                f"{file.parent.name}/{file.name}:{row_index}",
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                min_messages=min_messages,
                max_samples_per_trace=max_samples_per_trace,
            ):
                yield _agent_example_to_record(example)


class AgentLLMTracesDataset(AgentDataset):
    """Simple tool-call subset of Exgentic/agent-llm-traces."""

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        max_tools: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_new_tokens: int = 128,
        benchmark: Optional[str | Sequence[str]] = None,
        min_messages: Optional[int] = None,
        max_samples_per_trace: Optional[int] = None,
        use_hf_cache: bool = True,
        dataset_cache_dir: Optional[str] = None,
    ) -> None:
        self.data_path = Path(data_path).expanduser()
        data_dir = (
            self.data_path / "data"
            if (self.data_path / "data").is_dir()
            else self.data_path
        )
        files = sorted(data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {data_dir}")

        self.max_new_tokens = max_new_tokens
        benchmarks = _normalize_benchmarks(benchmark)
        if use_hf_cache:
            self.dataset = self._load_hf_dataset(
                files,
                max_samples=max_samples,
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                benchmarks=benchmarks,
                dataset_cache_dir=dataset_cache_dir,
            )
            self.examples = None
        else:
            self.dataset = None
            self.examples: Optional[List[AgentExample]] = []
            for record in _iter_agent_llm_trace_records(
                [str(file) for file in files],
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                benchmarks=benchmarks,
            ):
                self.examples.append(_record_to_agent_example(record))
                if max_samples is not None and len(self.examples) >= max_samples:
                    return

    @staticmethod
    def _load_hf_dataset(
        files: Sequence[Path],
        max_samples: Optional[int],
        max_tools: Optional[int],
        max_input_tokens: Optional[int],
        max_new_tokens: int,
        benchmarks: Optional[Sequence[str]],
        dataset_cache_dir: Optional[str],
    ) -> Any:
        from datasets import Dataset, Features, Value

        features = Features(
            {
                "qid": Value("string"),
                "system_prompt": Value("string"),
                "tools": Value("string"),
                "messages": Value("string"),
                "expected_tool_calls": Value("string"),
                "max_new_tokens": Value("int64"),
            }
        )
        dataset = Dataset.from_generator(
            _iter_agent_llm_trace_records,
            features=features,
            cache_dir=dataset_cache_dir,
            gen_kwargs={
                "files": [str(file) for file in files],
                "max_tools": max_tools,
                "max_input_tokens": max_input_tokens,
                "max_new_tokens": max_new_tokens,
                "benchmarks": tuple(benchmarks) if benchmarks is not None else None,
            },
            fingerprint=_dataset_fingerprint(
                files,
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                benchmarks=benchmarks,
            ),
        )
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        return dataset

    @staticmethod
    def _extract_example(
        row: Dict[str, Any],
        span: Dict[str, Any],
        qid: str,
        max_tools: Optional[int],
        max_input_tokens: Optional[int],
        max_new_tokens: int = 128,
    ) -> Optional[AgentExample]:
        attrs = span.get("attributes") or {}
        raw_messages = _json_loads(attrs.get("gen_ai.input.messages"), [])
        raw_outputs = _json_loads(attrs.get("gen_ai.output.messages"), [])
        raw_tools = _json_loads(attrs.get("gen_ai.tool.definitions"), [])
        expected_calls = [
            part
            for part in _message_parts(raw_outputs)
            if part.get("type") == "tool_call"
        ]
        prior_tool_use = any(
            part.get("type") in {"tool_call", "tool_call_response", "tool_result"}
            for part in _message_parts(raw_messages)
        )
        input_tokens = attrs.get("gen_ai.usage.input_tokens") or 0
        if not raw_messages or not raw_tools or prior_tool_use or not expected_calls:
            return None
        if max_tools is not None and len(raw_tools) > max_tools:
            return None
        if max_input_tokens is not None and input_tokens > max_input_tokens:
            return None

        system_prompt, messages = normalize_messages(raw_messages)
        if not messages:
            return None
        return AgentExample(
            qid=qid,
            system_prompt=system_prompt,
            tools=normalize_tools(raw_tools),
            messages=messages,
            expected_tool_calls=[_normalize_call(call) for call in expected_calls],
            max_new_tokens=max_new_tokens,
        )

    def __len__(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        assert self.examples is not None
        return len(self.examples)

    def __getitem__(self, index: int) -> AgentExample:
        if self.dataset is not None:
            return _record_to_agent_example(self.dataset[index])
        assert self.examples is not None
        return self.examples[index]


class OpenSWETracesDataset(AgentDataset):
    """Tool-call prediction samples from nvidia/Open-SWE-Traces trajectories."""

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        max_tools: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_new_tokens: int = 128,
        benchmark: Optional[str | Sequence[str]] = None,
        min_messages: Optional[int] = None,
        max_samples_per_trace: Optional[int] = None,
        use_hf_cache: Optional[bool] = None,
        dataset_cache_dir: Optional[str] = None,
    ) -> None:
        if min_messages is not None and min_messages < 1:
            raise ValueError("min_messages must be >= 1")
        if max_samples_per_trace is not None and max_samples_per_trace < 1:
            raise ValueError("max_samples_per_trace must be >= 1")
        self.data_path = Path(data_path).expanduser()
        data_dir = (
            self.data_path / "data"
            if (self.data_path / "data").is_dir()
            else self.data_path
        )
        benchmarks = _normalize_benchmarks(benchmark)
        files = sorted(
            file
            for file in data_dir.glob("*_trajectories/*.parquet")
            if _open_swe_file_matches(file, benchmarks)
        )
        if not files:
            raise FileNotFoundError(f"No Open-SWE parquet files found under {data_dir}")

        self.max_new_tokens = max_new_tokens
        if use_hf_cache is None:
            use_hf_cache = max_samples is None
        if use_hf_cache:
            self.dataset = self._load_hf_dataset(
                files,
                self.data_path,
                max_samples=max_samples,
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                min_messages=min_messages,
                max_samples_per_trace=max_samples_per_trace,
                benchmarks=benchmarks,
                dataset_cache_dir=dataset_cache_dir,
            )
            self.examples = None
        else:
            self.dataset = None
            self.examples: Optional[List[AgentExample]] = []
            for record in _iter_open_swe_trace_records(
                [str(file) for file in files],
                data_path=str(self.data_path),
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                min_messages=min_messages,
                max_samples_per_trace=max_samples_per_trace,
            ):
                self.examples.append(_record_to_agent_example(record))
                if max_samples is not None and len(self.examples) >= max_samples:
                    return

    @staticmethod
    def _load_hf_dataset(
        files: Sequence[Path],
        data_path: Path,
        max_samples: Optional[int],
        max_tools: Optional[int],
        max_input_tokens: Optional[int],
        max_new_tokens: int,
        min_messages: Optional[int],
        max_samples_per_trace: Optional[int],
        benchmarks: Optional[Sequence[str]],
        dataset_cache_dir: Optional[str],
    ) -> Any:
        from datasets import Dataset, Features, Value

        features = Features(
            {
                "qid": Value("string"),
                "system_prompt": Value("string"),
                "tools": Value("string"),
                "messages": Value("string"),
                "expected_tool_calls": Value("string"),
                "max_new_tokens": Value("int64"),
            }
        )
        tool_files = sorted(data_path.glob("*_tools.json"))
        dataset = Dataset.from_generator(
            _iter_open_swe_trace_records,
            features=features,
            cache_dir=dataset_cache_dir,
            gen_kwargs={
                "files": [str(file) for file in files],
                "data_path": str(data_path),
                "max_tools": max_tools,
                "max_input_tokens": max_input_tokens,
                "max_new_tokens": max_new_tokens,
                "min_messages": min_messages,
                "max_samples_per_trace": max_samples_per_trace,
            },
            fingerprint=_open_swe_dataset_fingerprint(
                files,
                tool_files,
                max_tools=max_tools,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                min_messages=min_messages,
                max_samples_per_trace=max_samples_per_trace,
                benchmarks=benchmarks,
            ),
        )
        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        return dataset

    @staticmethod
    def _extract_examples(
        row: Dict[str, Any],
        tools: List[Dict[str, Any]],
        qid: str,
        max_input_tokens: Optional[int],
        max_new_tokens: int = 128,
        min_messages: Optional[int] = None,
        max_samples_per_trace: Optional[int] = None,
    ) -> Iterator[AgentExample]:
        trajectory = row.get("trajectory") or []
        yielded = 0
        for message_index, message in enumerate(trajectory):
            if max_samples_per_trace is not None and yielded >= max_samples_per_trace:
                break
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            system_prompt, messages = normalize_open_swe_messages(
                trajectory[:message_index]
            )
            if not messages:
                continue
            if min_messages is not None and len(messages) < min_messages:
                continue
            if (
                max_input_tokens is not None
                and _estimate_open_swe_input_tokens(system_prompt, messages)
                > max_input_tokens
            ):
                continue
            yielded += 1
            yield AgentExample(
                qid=f"{qid}:{message_index}",
                system_prompt=system_prompt,
                tools=tools,
                messages=messages,
                expected_tool_calls=[
                    _normalize_call(call)
                    for call in message.get("tool_calls") or []
                    if isinstance(call, dict)
                ],
                max_new_tokens=max_new_tokens,
            )

    def __len__(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        assert self.examples is not None
        return len(self.examples)

    def __getitem__(self, index: int) -> AgentExample:
        if self.dataset is not None:
            return _record_to_agent_example(self.dataset[index])
        assert self.examples is not None
        return self.examples[index]


def load_agent_dataset(
    name: str,
    path: str,
    **kwargs: Any,
) -> AgentDataset:
    if name == "agent_llm_traces":
        return AgentLLMTracesDataset(path, **kwargs)
    if name == "open_swe_traces":
        return OpenSWETracesDataset(path, **kwargs)
    raise ValueError(f"Unsupported agent dataset: {name}")
