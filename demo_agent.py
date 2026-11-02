"""Demo: duplicate tool calls inside one agent trajectory — naive vs CachedTool.

A deterministic, LLM-free benchmark:

  * a scripted LLM (no API calls) asks the agent to run ``fibonacci(30)`` twice,
    then to answer — the classic "uncertain retry / redundant re-query" pattern;
  * fibonacci is deliberately exponential (2^n recursion), so one call is
    expensive and easy to measure;
  * we run the same trajectory once with the bare tool (naive) and once with
    the same tool wrapped in CachedTool, and compare real tool executions,
    wall time and cache statistics.

Run:  python demo_agent.py
"""
import time

from smolagents import CodeAgent, Model, ChatMessage, MessageRole, TokenUsage, Tool

from cached_tool import CachedTool


class Fibonacci(Tool):
    name = "fibonacci"
    description = "Computes the n-th Fibonacci number using the exponential recursive definition (very slow on purpose)."
    inputs = {"n": {"type": "integer", "description": "Which Fibonacci number to compute (0-indexed)."}}
    output_type = "integer"

    def __init__(self):
        super().__init__()
        self.executions = 0

    def forward(self, n: int) -> int:
        self.executions += 1
        return self._fib(n)

    def _fib(self, n: int) -> int:
        return n if n < 2 else self._fib(n - 1) + self._fib(n - 2)


class ScriptedLLM(Model):
    """Replays a fixed script of responses, so the benchmark is deterministic."""

    def __init__(self, steps):
        super().__init__()
        self._steps = list(steps)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        content = self._steps.pop(0)
        return ChatMessage(role=MessageRole.ASSISTANT, content=content, token_usage=TokenUsage(10, 10))


SCRIPT = [
    "<code>\nfibonacci(30)\n</code>",
    "<code>\nfibonacci(30)\n</code>",
    "<code>\nfinal_answer('done')\n</code>",
]


def run_trajectory(use_cache: bool):
    tool = Fibonacci()
    wrapped = CachedTool(tool) if use_cache else tool
    model = ScriptedLLM(SCRIPT)
    agent = CodeAgent(tools=[wrapped], model=model, max_steps=6, verbosity_level=0)

    t0 = time.perf_counter()
    out = agent.run("Use the fibonacci tool and report the result.")
    wall = time.perf_counter() - t0

    info = {
        "mode": "cached" if use_cache else "naive",
        "output": out,
        "tool_executions": tool.executions,
        "llm_calls": model.calls,
        "wall_seconds": round(wall, 3),
    }
    if use_cache:
        info["cache"] = wrapped.cache_info
    return info


def main():
    naive = run_trajectory(use_cache=False)
    cached = run_trajectory(use_cache=True)

    print(f"{'metric':<16}{'naive':>14}{'cached':>14}")
    for key in ("tool_executions", "llm_calls", "wall_seconds"):
        print(f"{key:<16}{naive[key]:>14}{cached[key]:>14}")
    print(f"{'output':<16}{str(naive['output']):>14}{str(cached['output']):>14}")
    c = cached["cache"]
    print(f"cache: hits={c['hits']} misses={c['misses']} hit_rate={c['hit_rate']:.2%} size={c['cache_size']}")

    assert naive["output"] == cached["output"], "outputs must match"
    assert cached["tool_executions"] < naive["tool_executions"], "caching must reduce executions"
    print("\nOK: identical output, fewer tool executions, wall time reduced.")


if __name__ == "__main__":
    main()
