# CachedTool: result caching for smolagents tool calls

An optimization for **Hugging Face smolagents** (`https://github.com/huggingface/smolagents`, Apache-2.0): a transparent LRU cache around any `Tool`, eliminating redundant re-execution of identical tool calls inside one agent trajectory.

## Problem

LLM agents re-issue the *same* tool call with the *same* arguments surprisingly often — an uncertain retry, a plan re-check, a re-query after a formatting mistake. Every duplicate:

- **re-executes the tool** (network call, computation, API cost);
- **appends the result to the model context again** → more input tokens per step, longer latency, higher cost;
- can be the difference between a 3-step and a 6-step trajectory.

## The optimization

`CachedTool` wraps any `smolagents.Tool`:

- cache key = SHA-1 of the canonical JSON of `(args, sorted kwargs)`;
- **LRU eviction** keeps memory bounded (`maxsize=128` by default);
- **transparent**: `name`, `description`, `inputs`, `output_type`, `output_schema` are preserved, so the agent's prompt and schema are unchanged — the agent cannot tell it is talking to a cache;
- optional `ignore_args` for volatile arguments (timestamps, request IDs) that must not participate in the key;
- `hits` / `misses` counters exposed for evaluation and telemetry.

## Deterministic demo (no LLM API needed)

`demo_agent.py` replays a fixed scripted trajectory with a scripted LLM: it calls an exponential Fibonacci tool (`fib(30)`) twice, then answers — the classic retry pattern. Same trajectory, two configurations:

| metric | naive | cached |
|---|---|---|
| tool executions | 2 | 1 |
| wall time | 0.448 s | 0.219 s |
| output | done | done |

Cache: `hits=1 misses=1 hit_rate=50% size=1` — one execution, one cache hit, identical output.

## Reproducing the upstream framework

`smolagents` was cloned from the upstream repository (`smolagents-src/`) and installed as editable package. Reproduced a representative subset of the upstream test-suite on this machine:

```
110 passed, 3 skipped, 6 failed, 1 error
```

The failures are all **integration tests requiring optional dependencies or live services** (MCP streamable-HTTP/SSE servers, gradio demos, clean-environment import checks) — not defects in the core library. Scope is documented here for honesty.

## Files

- `cached_tool.py` — the optimization (`CachedTool`, canonical key, LRU)
- `demo_agent.py` — naive-vs-cached trajectory benchmark (scripted LLM, deterministic)
- `test_cached_tool.py` — 5 unit tests: transparency, hit/miss, LRU eviction, ignore_args, clear
- `smolagents-src/` — upstream repository clone (Apache-2.0), kept for reproducible install
- `LICENSE` — upstream Apache-2.0 license

## Run

```bash
python -m venv .venv && pip install -e smolagents-src pytest
python -m pytest test_cached_tool.py -q      # 5 passed
python demo_agent.py                         # benchmark table above
```

## Upstream attribution

**smolagents** — Hugging Face, `https://github.com/huggingface/smolagents`, Apache-2.0 (© Hugging Face). "A barebones library for agents that write Python code to call tools or orchestrate other agents."
