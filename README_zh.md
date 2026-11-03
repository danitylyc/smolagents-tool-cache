# CachedTool：为 smolagents 工具调用做结果缓存

针对 **Hugging Face smolagents**（`https://github.com/huggingface/smolagents`，Apache-2.0）的优化：给任意 `Tool` 包一层透明的 LRU 缓存，消除同一 agent 轨迹内重复的相同工具调用。

## 问题

LLM agent 以惊人频率用**相同参数**重发**相同**工具调用——不确定时的重试、计划复检、格式错误后的重查。每一次重复都会：

- **再次执行工具**（网络调用、计算、API 成本）；
- **把结果再次追加进模型上下文** → 每步输入 token 更多、延迟更长、成本更高；
- 可能就是 3 步轨迹变成 6 步轨迹的差别。

## 优化方案

`CachedTool` 包装任意 `smolagents.Tool`：

- 缓存键 = `(args, 排序后 kwargs)` 的规范化 JSON 的 SHA-1；
- **LRU 淘汰**控制内存上限（默认 `maxsize=128`）；
- **透明**：`name`、`description`、`inputs`、`output_type`、`output_schema` 全部保留——agent 的提示与 schema 不变，agent 察觉不到自己在跟缓存对话；
- 可选 `ignore_args`：时间戳、请求 ID 等易变参数不参与缓存键；
- 暴露 `hits` / `misses` 计数，供评估与遥测使用。

## 确定性演示（无需 LLM API）

`demo_agent.py` 用脚本化 LLM 重放一条固定的脚本化轨迹：调用指数级斐波那契工具（`fib(30)`）两次再作答——经典的重试模式。同一条轨迹、两种配置：

| 指标 | 无缓存 | 有缓存 |
|---|---|---|
| 工具执行次数 | 2 | 1 |
| 墙钟时间 | 0.448 s | 0.219 s |
| 输出 | done | done |

缓存：`hits=1 misses=1 hit_rate=50% size=1`——执行一次、命中一次，输出完全一致。

## 复现上游框架

`smolagents` 从上游仓库克隆（`smolagents-src/`）并做可编辑安装。在本机复现了上游测试套件中有代表性的子集：

```
110 passed, 3 skipped, 6 failed, 1 error
```

失败项全部是**需要可选依赖或在线服务的集成测试**（MCP streamable-HTTP/SSE 服务器、gradio 演示、干净环境导入检查）——不是核心库缺陷。范围如实记录。

## 文件

- `cached_tool.py` — 优化本体（`CachedTool`、规范化键、LRU）
- `demo_agent.py` — 无缓存 vs 有缓存的轨迹基准（脚本化 LLM，确定性）
- `test_cached_tool.py` — 5 个单元测试：透明性、命中/未命中、LRU 淘汰、ignore_args、clear
- `smolagents-src/` — 上游仓库克隆（Apache-2.0），用于可复现安装
- `LICENSE` — 上游 Apache-2.0 许可证

## 运行

```bash
python -m venv .venv && pip install -e smolagents-src pytest
python -m pytest test_cached_tool.py -q      # 5 passed
python demo_agent.py                         # 上表基准
```

## 上游归属

**smolagents** — Hugging Face，`https://github.com/huggingface/smolagents`，Apache-2.0（© Hugging Face）。"一个写 Python 代码来调用工具或编排其他 agent 的极简 agent 库。"
