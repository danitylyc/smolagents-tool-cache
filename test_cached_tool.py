"""Unit tests for CachedTool."""
import pytest

from smolagents import Tool

from cached_tool import CachedTool


class SlowSquared(Tool):
    name = "slow_squared"
    description = "Squares a number (pretend it is slow)."
    inputs = {"x": {"type": "integer", "description": "Input value."}}
    output_type = "integer"

    def __init__(self):
        super().__init__()
        self.executions = 0

    def forward(self, x: int) -> int:
        self.executions += 1
        return x * x


def test_transparency():
    tool = SlowSquared()
    cached = CachedTool(tool)
    assert cached.name == tool.name
    assert cached.description == tool.description
    assert cached.inputs == tool.inputs
    assert cached.output_type == tool.output_type


def test_hit_and_miss():
    tool = SlowSquared()
    cached = CachedTool(tool)
    assert cached(x=4) == 16
    assert tool.executions == 1
    assert cached.misses == 1 and cached.hits == 0
    # same args -> cache hit, no re-execution
    assert cached(x=4) == 16
    assert tool.executions == 1
    assert cached.hits == 1
    # different args -> miss
    assert cached(x=5) == 25
    assert tool.executions == 2


def test_lru_eviction():
    tool = SlowSquared()
    cached = CachedTool(tool, maxsize=2)
    cached(x=1); cached(x=2)
    cached(x=1)  # refresh x=1 as most-recent
    cached(x=3)  # evicts x=2
    cached(x=2)  # miss again -> re-execute
    assert tool.executions == 4


def test_ignore_args():
    tool = SlowSquared()
    cached = CachedTool(tool, ignore_args={"x"})
    r1 = cached(x=1)
    r2 = cached(x=999)
    assert r1 == r2 == 1  # key ignores x, so same cache entry (documented behavior)
    assert tool.executions == 1


def test_clear_cache():
    tool = SlowSquared()
    cached = CachedTool(tool)
    cached(x=1)
    cached.clear_cache()
    cached(x=1)
    assert tool.executions == 2
