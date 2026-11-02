"""[OPTIMIZATION] CachedTool: transparent result-caching wrapper for smolagents tools.

Motivation
----------
LLM agents frequently re-issue the same tool call inside one trajectory (an
uncertain retry, a plan re-check, or a redundant re-query). Every duplicate
call re-executes the tool (expensive: network, compute, time) and the result
is appended to the model context again, inflating token cost and latency.

CachedTool wraps any `smolagents.Tool` with an LRU cache keyed on a canonical
form of the call arguments, without changing the tool's name/description/schema
(so the agent sees an identical tool) and without changing results.

Design notes
------------
- Cache key = stable JSON of sorted kwargs (and positional args, when given).
- LRU eviction keeps memory bounded (default maxsize=128 entries).
- `hits` / `misses` counters are exposed for evaluation and telemetry.
- Optional per-argument ignore-list (`ignore_args`) for volatile args (e.g.
  timestamps) that should not participate in the key.

Upstream project: https://github.com/huggingface/smolagents (Apache-2.0)
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from smolagents.tools import Tool


def canonical_key(args: tuple, kwargs: dict, ignore_args: set[str] = frozenset()) -> str:
    """Build a stable, hashable key from call arguments."""
    kw = {k: v for k, v in kwargs.items() if k not in ignore_args}
    payload = json.dumps(
        [list(args), kw], sort_keys=True, default=lambda o: repr(o), ensure_ascii=False
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class CachedTool(Tool):
    """A drop-in wrapper that adds an LRU result cache to any smolagents Tool.

    The wrapped tool's name / description / inputs / output_type are preserved
    so the agent prompt and schema remain identical; only execution is cached.
    """

    skip_forward_signature_validation = True  # generic **kwargs signature is intentional

    def __init__(self, tool: Tool, maxsize: int = 128, ignore_args: set[str] | None = None):
        super().__init__()
        self._tool = tool
        self.name = tool.name
        self.description = tool.description
        self.inputs = tool.inputs
        self.output_type = tool.output_type
        self.output_schema = getattr(tool, "output_schema", None)
        self._maxsize = maxsize
        self._ignore_args = set(ignore_args or ())
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    # ---- cache plumbing -------------------------------------------------
    def _cache_get(self, key: str) -> Any | None:
        if key in self._cache:
            self._cache.move_to_end(key)  # LRU: most recently used goes last
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def _cache_put(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # evict least recently used

    # ---- Tool interface ------------------------------------------------
    def forward(self, *args, **kwargs) -> Any:
        key = canonical_key(args, kwargs, self._ignore_args)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        result = self._tool(*args, **kwargs)
        self._cache_put(key, result)
        return result

    # ---- introspection helpers -----------------------------------------
    @property
    def cache_info(self) -> dict:
        return {
            "tool": self.name,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) else 0.0,
            "cache_size": len(self._cache),
            "maxsize": self._maxsize,
        }

    def clear_cache(self) -> None:
        self._cache.clear()
