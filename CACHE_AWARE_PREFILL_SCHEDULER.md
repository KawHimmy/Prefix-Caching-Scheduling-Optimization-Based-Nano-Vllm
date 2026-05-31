# 基于 nano-vLLM 的前缀缓存调度优化

## 项目简介

这是一个面向大模型推理调度优化的项目，核心目标是在 Prefix Cache 和 KV Cache 约束下，让 prefill 调度更加高效。项目实现了一个 **Cache-Aware Prefill Scheduler**，用来缓解 FIFO prefill admission 在混合请求场景下的队头阻塞问题，并提升缓存感知调度能力。

与单纯做可视化或 API 服务封装不同，这个仓库更聚焦于 **LLM Inference Scheduler** 本身：通过改造 `Scheduler`、`BlockManager` 和调度统计接口，构建一个可以复现实验结果的调度优化项目。

## 原始问题

原始 prefill admission 逻辑只检查 waiting 队列头部请求：

```python
seq = self.waiting[0]
num_cached_blocks = self.block_manager.can_allocate(seq)
if num_cached_blocks == -1:
    break
```

这会带来两个典型问题：

1. **队头阻塞**：队首长请求无法分配时，后续短请求也会被阻塞。
2. **Prefix Cache 利用不足**：高 cache-hit 请求不会被主动优先调度。

## 项目目标

本项目的目标是做一个适合 GitHub 展示和简历描述的调度优化项目：

- 保留 FIFO baseline 作为默认行为。
- 增加 Cache-Aware Prefill Scheduler。
- 支持有限窗口扫描 waiting 请求。
- 优先调度 cache 命中高、block 压力低、prefill token 少的请求。
- 在特定情况下绕过被阻塞的队首请求。
- 提供 scheduler-only benchmark 和真实统计结果。

## 核心改动

### 1. Allocation Estimate

在 `nanovllm/engine/block_manager.py` 中新增 `estimate_allocation(seq)`，返回：

```python
AllocationEstimate(
    can_allocate: bool,
    num_cached_blocks: int,
    num_required_blocks: int,
    num_total_blocks: int,
)
```

它为调度器在真正分配前提供关键状态：

- 当前请求可复用多少 prefix block
- 当前请求还需要多少 free KV blocks
- 当前请求是否可被 admit

### 2. Cache-Aware Prefill Policy

新增配置项：

```python
prefill_schedule_policy: str = "fifo"
prefill_scan_window: int = 8
collect_scheduler_stats: bool = False
```

当 `prefill_schedule_policy="cache_aware"` 时，调度器会扫描 waiting 队列前 `prefill_scan_window` 个请求，并按以下优先级选择候选：

1. 可分配的请求优先
2. Prefix Cache 命中更多的请求优先
3. 需要更少 KV blocks 的请求优先
4. 需要更少 prefill tokens 的请求优先
5. waiting index 更小的请求优先

### 3. 队头阻塞规避

如果 waiting 队首请求因为 KV Cache 压力而不可分配，而 scan window 内后续请求可分配，则允许后续请求提前执行。

### 4. 调度统计接口

调度器支持记录：

- `prefill_batches`
- `decode_batches`
- `prefill_tokens_scheduled`
- `cached_blocks_reused`
- `cached_tokens_reused`
- `prefill_candidates_scanned`
- `out_of_order_prefills`
- `head_of_line_prefills`
- `preemptions`

同时，`LLMEngine` 增加 `get_scheduler_stats()`，方便真实模型运行后读取统计信息。

## Benchmark 指标定义

为了让这个项目不仅停留在“方案描述”，仓库中的 `bench_cache_scheduler.py` 输出了可复现的实验指标：

- `prefill_batches`：prefill 调度批次数
- `decode_batches`：decode 调度批次数
- `prefill_tokens_scheduled`：prefill 阶段总调度 token 数
- `cached_blocks_reused`：复用的 prefix-cache block 数
- `cached_tokens_reused`：复用的 prefix-cache token 数
- `cache_hit_ratio`：`cached_tokens_reused / prefill_tokens_scheduled`
- `prefill_candidates_scanned`：cache-aware 模式扫描的候选请求数
- `out_of_order_prefills`：非 FIFO 顺序 prefill 次数
- `head_of_line_prefills`：绕过被阻塞队首请求的次数
- `preemptions`：decode 阶段抢占次数
- `first_prefill_summary`：请求首次进入 prefill 的最小值 / 最大值 / 平均值
- `finish_summary`：请求完成 step 的最小值 / 最大值 / 平均值

## Workloads

### shared-prefix

构造多条共享完整 block 前缀的请求，用于观察高 cache-hit 请求是否会被提前调度。

### hol

构造 KV block 紧张场景，让队首长请求暂时不可分配，但后续短请求可以执行。

### mixed

混合长 prompt、短 prompt 和共享前缀 prompt，观察综合调度行为与 trade-off。

## 实验结果

以下结果由当前仓库中的 `bench_cache_scheduler.py` 实际运行得到。

### 1. shared-prefix workload

| 指标 | FIFO | Cache-Aware | 变化 |
|---|---:|---:|---:|
| total_steps | 6 | 6 | 0 |
| prefill_tokens_scheduled | 28 | 28 | 0 |
| cached_tokens_reused | 16 | 16 | 0 |
| cache_hit_ratio | 57.14% | 57.14% | 0 |
| out_of_order_prefills | 0 | 3 | +3 |
| avg first_prefill_step | 1.50 | 0.75 | -50.0% |
| avg finish_step | 4.50 | 4.50 | 0 |

结论：在共享前缀场景下，Cache-Aware 没有改变总工作量，但能显著提前高 cache-hit 请求首次被服务的时间。

### 2. hol workload

| 指标 | FIFO | Cache-Aware | 变化 |
|---|---:|---:|---:|
| total_steps | 8 | 9 | +1 |
| prefill_batches | 4 | 5 | +1 |
| preemptions | 1 | 2 | +1 |
| out_of_order_prefills | 0 | 2 | +2 |
| head_of_line_prefills | 0 | 2 | +2 |
| avg first_prefill_step | 4.00 | 2.00 | -50.0% |
| avg finish_step | 4.60 | 4.60 | 0 |

结论：在队头阻塞场景下，Cache-Aware 明确触发了 Head-of-Line Blocking Avoidance，并将后续请求的平均首次 prefill 时延降低了 50%。

### 3. mixed workload

| 指标 | FIFO | Cache-Aware | 变化 |
|---|---:|---:|---:|
| total_steps | 8 | 9 | +1 |
| preemptions | 3 | 1 | -66.7% |
| cached_tokens_reused | 36 | 16 | -55.6% |
| cache_hit_ratio | 83.72% | 39.02% | -44.70pt |
| out_of_order_prefills | 0 | 4 | +4 |
| avg first_prefill_step | 2.20 | 1.80 | -18.2% |
| avg finish_step | 5.00 | 5.83 | +16.6% |

结论：在混合场景下，当前 heuristic 能改善首次服务顺序并减少抢占，但没有同时最大化 cache reuse 与整体完成时延，体现了真实的系统调度 trade-off。

## 如何运行

### Scheduler Microbenchmark

```bash
python bench_cache_scheduler.py --policy both --workload shared-prefix --scan-window 8
python bench_cache_scheduler.py --policy both --workload hol --scan-window 8
python bench_cache_scheduler.py --policy both --workload mixed --scan-window 8
```

### 真实模型使用

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/YOUR/MODEL/PATH",
    prefill_schedule_policy="cache_aware",
    prefill_scan_window=8,
    collect_scheduler_stats=True,
)

outputs = llm.generate([[1, 2, 3, 4], [1, 2, 3, 5]], SamplingParams(max_tokens=4), use_tqdm=False)
print(outputs)
print(llm.get_scheduler_stats())
```

## 致谢

本仓库延续了轻量级大模型推理实现的代码风格，并在此基础上扩展出一个面向缓存感知 prefill 调度的优化实验项目。
