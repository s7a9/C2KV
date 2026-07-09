from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import OrderedDict
from copy import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import requests


JSONDict = Dict[str, Any]
CacheKey = Tuple[str, str, int]
LOGGER = logging.getLogger("c2kv_proxy")


class LRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self._lock = Lock()
        self._items: OrderedDict[CacheKey, str] = OrderedDict()

    def get(self, key: CacheKey) -> Optional[str]:
        if self.capacity <= 0:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: CacheKey, value: str) -> None:
        if self.capacity <= 0:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def clear(self) -> int:
        with self._lock:
            size = len(self._items)
            self._items.clear()
            return size

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def _parse_roles(value: str) -> set[str]:
    return {role.strip() for role in value.split(",") if role.strip()}


def _message_content(message: JSONDict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _extract_role(message: JSONDict, override: str) -> str:
    if override != "auto":
        return override
    role = str(message.get("role", "user"))
    return role if role in {"system", "user", "assistant"} else "user"


def _last_user_index(messages: Sequence[JSONDict]) -> Optional[int]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def protected_message_indices(
    messages: Sequence[JSONDict],
    args: argparse.Namespace,
) -> set[int]:
    count = len(messages)
    protected = set(range(min(args.protect_prefix_messages, count)))
    suffix_start = max(0, count - args.protect_suffix_messages)
    protected.update(range(suffix_start, count))
    if args.preserve_last_user:
        last_user = _last_user_index(messages)
        if last_user is not None:
            protected.add(last_user)
    return protected


def eligible_message_indices(messages: Sequence[JSONDict], args: argparse.Namespace) -> List[int]:
    eligible_roles = _parse_roles(args.eligible_roles)
    protected = protected_message_indices(messages, args)
    indices: List[int] = []
    for index, message in enumerate(messages):
        if index in protected:
            continue
        if message.get("c2kv_key_hash"):
            continue
        role = str(message.get("role", ""))
        if role not in eligible_roles:
            continue
        content = _message_content(message)
        if len(content) < args.min_content_chars:
            continue
        indices.append(index)
    return indices


def select_reuse_indices(
    candidates: Sequence[int],
    pattern: str,
    ratio: float,
    rng: random.Random,
) -> List[int]:
    if pattern == "none" or not candidates:
        return []
    if pattern == "all":
        return list(candidates)
    count = max(1, min(len(candidates), int(len(candidates) * ratio + 0.999999)))
    if pattern == "forward":
        return list(candidates[:count])
    if pattern == "random":
        return sorted(rng.sample(list(candidates), count))
    raise ValueError(f"unsupported reuse pattern: {pattern}")


def post_extract(
    upstream_base_url: str,
    text: str,
    role: str,
    compression_ratio: int,
    timeout: float,
    retries: int,
    retry_interval: float,
) -> Tuple[JSONDict, int]:
    attempts = max(1, retries + 1)
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                f"{upstream_base_url}/v1/c2kv/extract",
                json={
                    "text": text,
                    "role": role,
                    "compression_ratio": compression_ratio,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json(), attempt + 1
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            LOGGER.warning(
                "extract attempt failed attempt=%s/%s role=%s chars=%s error=%s",
                attempt + 1,
                attempts,
                role,
                len(text),
                exc,
            )
            if attempt + 1 < attempts and retry_interval > 0:
                time.sleep(retry_interval)
    if isinstance(last_exc, requests.RequestException):
        raise last_exc
    raise requests.RequestException(str(last_exc))


class C2KVProxy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.upstream_base_url = args.upstream_base_url.rstrip("/")
        self.cache = LRUCache(args.cache_size)

    def rewrite_chat_payload(self, payload: JSONDict) -> Tuple[JSONDict, JSONDict]:
        request_args = self._request_args(payload)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload, {"error": "payload.messages must be a list"}

        rewritten = dict(payload)
        rewritten_messages = [dict(message) if isinstance(message, dict) else message for message in messages]
        rewritten["messages"] = rewritten_messages

        rng = random.Random(f"{self.args.seed}:{time.time_ns()}")
        candidates = eligible_message_indices(rewritten_messages, request_args)
        selected = select_reuse_indices(
            candidates,
            request_args.reuse_pattern,
            request_args.reuse_ratio,
            rng,
        )

        stats: JSONDict = {
            "candidate_messages": len(candidates),
            "protected_messages": len(protected_message_indices(rewritten_messages, request_args)),
            "selected_messages": len(selected),
            "cache_hits": 0,
            "extract_requests": 0,
            "extract_attempts": 0,
            "extract_failures": 0,
            "attached_key_hashes": 0,
        }
        for index in selected:
            message = rewritten_messages[index]
            if not isinstance(message, dict):
                continue
            content = _message_content(message)
            role = _extract_role(message, request_args.extract_role)
            cache_key = (role, content, request_args.compression_ratio)
            key_hash = self.cache.get(cache_key)
            if key_hash is not None:
                stats["cache_hits"] += 1
            else:
                stats["extract_requests"] += 1
                try:
                    result, attempts = post_extract(
                        self.upstream_base_url,
                        content,
                        role,
                        request_args.compression_ratio,
                        request_args.request_timeout,
                        request_args.extract_retries,
                        request_args.extract_retry_interval,
                    )
                    stats["extract_attempts"] += attempts
                    key_hash = result.get("key_hash") if result.get("success") else None
                    if key_hash:
                        self.cache.put(cache_key, key_hash)
                    else:
                        stats["extract_failures"] += 1
                        continue
                except requests.RequestException:
                    stats["extract_attempts"] += request_args.extract_retries + 1
                    stats["extract_failures"] += 1
                    LOGGER.exception(
                        "extract failed index=%s role=%s chars=%s retries=%s",
                        index,
                        role,
                        len(content),
                        request_args.extract_retries,
                    )
                    continue
            message["c2kv_key_hash"] = key_hash
            stats["attached_key_hashes"] += 1

        if request_args.strip_proxy_fields:
            rewritten.pop("c2kv_proxy", None)
        stats["cache_size"] = len(self.cache)
        stats["reuse_pattern"] = request_args.reuse_pattern
        stats["reuse_ratio"] = request_args.reuse_ratio
        stats["compression_ratio"] = request_args.compression_ratio
        LOGGER.info(
            "rewrite chat model=%s messages=%s candidates=%s protected=%s "
            "selected=%s attached=%s cache_hits=%s extract_requests=%s "
            "extract_attempts=%s extract_failures=%s cache_size=%s "
            "pattern=%s ratio=%s compression=%s",
            rewritten.get("model"),
            len(rewritten_messages),
            stats["candidate_messages"],
            stats["protected_messages"],
            stats["selected_messages"],
            stats["attached_key_hashes"],
            stats["cache_hits"],
            stats["extract_requests"],
            stats["extract_attempts"],
            stats["extract_failures"],
            stats["cache_size"],
            stats["reuse_pattern"],
            stats["reuse_ratio"],
            stats["compression_ratio"],
        )
        return rewritten, stats

    def _request_args(self, payload: JSONDict) -> argparse.Namespace:
        request_args = copy(self.args)
        overrides = payload.get("c2kv_proxy")
        if not isinstance(overrides, dict):
            return request_args
        for key in (
            "reuse_pattern",
            "reuse_ratio",
            "compression_ratio",
            "eligible_roles",
            "extract_role",
            "preserve_last_user",
            "protect_prefix_messages",
            "protect_suffix_messages",
            "min_content_chars",
            "extract_retries",
            "extract_retry_interval",
        ):
            if key in overrides:
                setattr(request_args, key, overrides[key])
        if request_args.reuse_pattern not in {"none", "all", "forward", "random"}:
            raise ValueError("c2kv_proxy.reuse_pattern must be one of none/all/forward/random")
        if request_args.extract_role not in {"auto", "system", "user", "assistant"}:
            raise ValueError("c2kv_proxy.extract_role must be one of auto/system/user/assistant")
        if request_args.reuse_ratio <= 0 or request_args.reuse_ratio > 1:
            raise ValueError("c2kv_proxy.reuse_ratio must be in (0, 1]")
        if request_args.compression_ratio < 1:
            raise ValueError("c2kv_proxy.compression_ratio must be >= 1")
        if request_args.protect_prefix_messages < 0:
            raise ValueError("c2kv_proxy.protect_prefix_messages must be >= 0")
        if request_args.protect_suffix_messages < 0:
            raise ValueError("c2kv_proxy.protect_suffix_messages must be >= 0")
        if request_args.extract_retries < 0:
            raise ValueError("c2kv_proxy.extract_retries must be >= 0")
        if request_args.extract_retry_interval < 0:
            raise ValueError("c2kv_proxy.extract_retry_interval must be >= 0")
        return request_args

    def forward(self, path: str, payload: Optional[JSONDict]) -> requests.Response:
        url = f"{self.upstream_base_url}{path}"
        return requests.post(url, json=payload, timeout=self.args.request_timeout)

    def forward_get(self, path: str) -> requests.Response:
        url = f"{self.upstream_base_url}{path}"
        return requests.get(url, timeout=self.args.request_timeout)

    def clear_cache(self) -> JSONDict:
        cleared = self.cache.clear()
        LOGGER.info("cleared proxy cache entries=%s", cleared)
        return {"status": "ok", "cleared_entries": cleared, "cache_size": len(self.cache)}


def make_handler(proxy: C2KVProxy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "C2KVProxy/0.1"

        def _send_json(self, status: int, payload: JSONDict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> Optional[JSONDict]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON body"})
                return None
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "JSON body must be an object"})
                return None
            return payload

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path in {"/health", "/v1/c2kv/proxy/health"}:
                LOGGER.debug("health check path=%s client=%s", path, self.client_address)
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "upstream_base_url": proxy.upstream_base_url,
                        "cache_size": len(proxy.cache),
                    },
                )
                return
            try:
                response = proxy.forward_get(path)
            except requests.RequestException as exc:
                LOGGER.exception("GET upstream failed path=%s", path)
                self._send_json(502, {"error": str(exc)})
                return
            body = response.content
            self.send_response(response.status_code)
            self.send_header(
                "Content-Type",
                response.headers.get("Content-Type", "application/json"),
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            LOGGER.info("GET path=%s status=%s bytes=%s", path, response.status_code, len(body))

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            started = time.perf_counter()
            if path in {"/clear_cache", "/v1/c2kv/proxy/cache/clear"}:
                self._send_json(200, proxy.clear_cache())
                return

            payload = self._read_json()
            if payload is None:
                LOGGER.warning("POST path=%s invalid_json client=%s", path, self.client_address)
                return

            if path == "/v1/chat/completions":
                try:
                    payload, stats = proxy.rewrite_chat_payload(payload)
                except ValueError as exc:
                    LOGGER.warning("POST path=%s invalid_proxy_config error=%s", path, exc)
                    self._send_json(400, {"error": str(exc)})
                    return
            else:
                stats = {}

            try:
                response = proxy.forward(path, payload)
            except requests.RequestException as exc:
                LOGGER.exception("POST upstream failed path=%s stats=%s", path, stats)
                self._send_json(502, {"error": str(exc), "c2kv_proxy": stats})
                return

            body = response.content
            if path == "/v1/chat/completions" and proxy.args.include_proxy_stats:
                try:
                    response_json = response.json()
                    response_json["c2kv_proxy"] = stats
                    body = json.dumps(response_json, ensure_ascii=False).encode("utf-8")
                    content_type = "application/json"
                except ValueError:
                    content_type = response.headers.get("Content-Type", "application/octet-stream")
            else:
                content_type = response.headers.get("Content-Type", "application/json")

            self.send_response(response.status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "POST path=%s status=%s bytes=%s elapsed=%.3fs stats=%s",
                path,
                response.status_code,
                len(body),
                elapsed,
                stats,
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            if proxy.args.quiet:
                return
            LOGGER.debug("http %s", fmt % args)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible C2KV proxy that extracts reusable message KV before forwarding chat requests."
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=31000)
    parser.add_argument("--upstream-base-url", default="http://localhost:30000")
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument(
        "--reuse-pattern",
        choices=["none", "all", "forward", "random"],
        default="all",
        help="Which eligible messages to extract and attach as reusable KV.",
    )
    parser.add_argument(
        "--reuse-ratio",
        type=float,
        default=1.0,
        help="Fraction of eligible messages used by forward/random patterns.",
    )
    parser.add_argument(
        "--eligible-roles",
        default="user,assistant,tool",
        help="Comma-separated message roles eligible for C2KV extraction.",
    )
    parser.add_argument(
        "--extract-role",
        choices=["auto", "system", "user", "assistant"],
        default="auto",
        help="Role sent to /v1/c2kv/extract. auto maps tool/unknown roles to user.",
    )
    parser.add_argument(
        "--preserve-last-user",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not extract the last user message, usually the live query.",
    )
    parser.add_argument(
        "--protect-prefix-messages",
        type=int,
        default=0,
        help="Always leave the first N messages unextracted.",
    )
    parser.add_argument(
        "--protect-suffix-messages",
        type=int,
        default=0,
        help="Always leave the last N messages unextracted.",
    )
    parser.add_argument("--min-content-chars", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--extract-retries",
        type=int,
        default=0,
        help="Number of retries after a failed /v1/c2kv/extract request.",
    )
    parser.add_argument(
        "--extract-retry-interval",
        type=float,
        default=1.0,
        help="Seconds to sleep between extract retries.",
    )
    parser.add_argument(
        "--include-proxy-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach c2kv_proxy statistics to JSON chat responses.",
    )
    parser.add_argument(
        "--strip-proxy-fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove top-level c2kv_proxy config fields before forwarding.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-file",
        help="Optional file path for proxy logs. Defaults to stderr.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.reuse_ratio <= 0 or args.reuse_ratio > 1:
        raise ValueError("--reuse-ratio must be in (0, 1]")
    if args.compression_ratio < 1:
        raise ValueError("--compression-ratio must be >= 1")
    if args.protect_prefix_messages < 0:
        raise ValueError("--protect-prefix-messages must be >= 0")
    if args.protect_suffix_messages < 0:
        raise ValueError("--protect-suffix-messages must be >= 0")
    if args.extract_retries < 0:
        raise ValueError("--extract-retries must be >= 0")
    if args.extract_retry_interval < 0:
        raise ValueError("--extract-retry-interval must be >= 0")
    return args


def configure_logging(args: argparse.Namespace) -> None:
    level = logging.WARNING if args.quiet else getattr(logging, args.log_level)
    handlers: List[logging.Handler]
    if args.log_file:
        handlers = [logging.FileHandler(args.log_file, encoding="utf-8")]
    else:
        handlers = [logging.StreamHandler()]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args)
    proxy = C2KVProxy(args)
    server = ThreadingHTTPServer(
        (args.listen_host, args.listen_port),
        make_handler(proxy),
    )
    LOGGER.info(
        "C2KV proxy listening host=%s port=%s upstream=%s reuse_pattern=%s "
        "reuse_ratio=%s cache_size=%s protect_prefix=%s protect_suffix=%s "
        "extract_retries=%s",
        args.listen_host,
        args.listen_port,
        args.upstream_base_url.rstrip("/"),
        args.reuse_pattern,
        args.reuse_ratio,
        args.cache_size,
        args.protect_prefix_messages,
        args.protect_suffix_messages,
        args.extract_retries,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("C2KV proxy interrupted")
    finally:
        server.server_close()
        LOGGER.info("C2KV proxy stopped")


if __name__ == "__main__":
    main()
