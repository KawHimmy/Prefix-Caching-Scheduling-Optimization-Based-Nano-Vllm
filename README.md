# 基于 nano-vLLM 的前缀缓存调度优化

这是一个面向大模型推理调度优化的项目，核心目标是在 Prefix Cache 和 KV Cache 约束下，让 prefill 调度更加高效。项目实现了一个 **Cache-Aware Prefill Scheduler**，用于缓解 FIFO prefill admission 在混合请求场景下的队头阻塞问题，并提升缓存感知调度能力。

## 项目概述

本仓库关注 LLM Serving 中一个非常具体但很重要的问题：**当 Prefix Cache 和 KV Cache Block 都是稀缺资源时，如何更高效地调度 prefill 请求**。

相比原始的 FIFO prefill admission 逻辑，本项目新增了一个 **Cache-Aware Prefill Scheduler**，具备以下能力：

- 在真正分配前预估 prefix cache 可复用情况
- 不再只看 waiting 队列头部，而是扫描一个有限窗口
- 优先调度缓存复用收益更高、KV block 压力更低的请求
- 在特定场景下绕过被阻塞的队首请求
- 输出调度层面的统计信息，便于实验复现与结果分析

## 核心改动

### 1. Allocation Estimate 接口

`BlockManager` 新增了一个非破坏性的 allocation estimate 接口，用于返回：

- 当前请求是否可以被调度
- 当前请求可复用多少 prefix block
- 当前请求还需要多少 KV-cache block
- 当前请求总共需要多少逻辑 block

### 2. Cache-Aware Prefill 策略

在 `Config` 中新增调度配置：

```python
prefill_schedule_policy: str = "fifo"
prefill_scan_window: int = 8
collect_scheduler_stats: bool = False
```

当 `prefill_schedule_policy="cache_aware"` 时，调度器会扫描 waiting 队列前 `prefill_scan_window` 个请求，并优先选择：

1. Prefix Cache 命中更多的请求
2. 需要更少 KV-cache block 的请求
3. 需要更少 prefill token 的请求
4. 队列位置更靠前的请求作为公平性兜底

### 3. 队头阻塞规避

如果 waiting 队列头部请求因为 KV-cache 压力无法分配，而窗口内后续请求可以运行，那么 Cache-Aware Scheduler 会允许后续请求先执行，从而缓解队头阻塞。

### 4. 调度基准测试

仓库中新增了 `bench_cache_scheduler.py`，用于在合成 workload 下对比 FIFO 和 Cache-Aware 两种策略的行为差异。

## 仓库结构

```text
nanovllm/
  config.py
  engine/
    block_manager.py
    llm_engine.py
    scheduler.py
bench.py
bench_cache_scheduler.py
example.py
CACHE_AWARE_PREFILL_SCHEDULER.md
```

## 安装方式

先克隆仓库并安装依赖：

```bash
git clone https://github.com/KawHimmy/Prefix-Caching-Scheduling-Optimization-Based-Nano-Vllm.git
cd Prefix-Caching-Scheduling-Optimization-Based-Nano-Vllm
pip install -e .
```

也可以直接从 GitHub 安装：

```bash
pip install git+https://github.com/KawHimmy/Prefix-Caching-Scheduling-Optimization-Based-Nano-Vllm.git
```

## 快速开始

### 1. 运行调度 microbenchmark

不加载真实模型、只验证调度逻辑：

```bash
python bench_cache_scheduler.py --policy both --workload shared-prefix --scan-window 8
python bench_cache_scheduler.py --policy both --workload hol --scan-window 8
python bench_cache_scheduler.py --policy both --workload mixed --scan-window 8
```

### 2. 真实模型使用示例

本项目保留了原有 `LLM` 调用接口，并新增调度器控制参数：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/YOUR/MODEL/PATH",
    enforce_eager=True,
    tensor_parallel_size=1,
    prefill_schedule_policy="cache_aware",
    prefill_scan_window=8,
    collect_scheduler_stats=True,
)

sampling_params = SamplingParams(temperature=0.6, max_tokens=32)
outputs = llm.generate([[1, 2, 3, 4], [1, 2, 3, 5]], sampling_params, use_tqdm=False)
print(outputs[0]["text"])
print(llm.get_scheduler_stats())
```

## Benchmark 指标说明

`bench_cache_scheduler.py` 主要输出以下指标：

- `prefill_batches`：prefill 调度轮数
- `decode_batches`：decode 调度轮数
- `prefill_tokens_scheduled`：prefill 阶段总调度 token 数
- `cached_blocks_reused`：复用的 prefix-cache block 数
- `cached_tokens_reused`：复用的 prefix-cache token 数
- `cache_hit_ratio`：prefill 工作量中被缓存复用覆盖的比例
- `out_of_order_prefills`：偏离 FIFO 顺序的 prefill 次数
- `head_of_line_prefills`：绕过被阻塞队首请求的次数
- `preemptions`：decode 阶段发生的抢占次数
- `first_prefill_summary`：请求首次进入 prefill 的最小值 / 最大值 / 平均值
- `finish_summary`：请求完成 step 的最小值 / 最大值 / 平均值

## 实验结果

以下数据来自当前仓库内 `bench_cache_scheduler.py` 的实际运行结果。

### shared-prefix workload

| 指标 | FIFO | Cache-Aware | 变化 |
|---|---:|---:|---:|
| total_steps | 6 | 6 | 0 |
| prefill_tokens_scheduled | 28 | 28 | 0 |
| cached_tokens_reused | 16 | 16 | 0 |
| cache_hit_ratio | 57.14% | 57.14% | 0 |
| out_of_order_prefills | 0 | 3 | +3 |
| avg first_prefill_step | 1.50 | 0.75 | -50.0% |
| avg finish_step | 4.50 | 4.50 | 0 |

结论：在共享前缀场景下，Cache-Aware 并没有减少总工作量，但能显著提前高 cache-hit 请求首次被服务的时间。

### hol workload

| 指标 | FIFO | Cache-Aware | 变化 |
|---|---:|---:|---:|
| total_steps | 8 | 9 | +1 |
| prefill_batches | 4 | 5 | +1 |
| preemptions | 1 | 2 | +1 |
| out_of_order_prefills | 0 | 2 | +2 |
| head_of_line_prefills | 0 | 2 | +2 |
| avg first_prefill_step | 4.00 | 2.00 | -50.0% |
| avg finish_step | 4.60 | 4.60 | 0 |

结论：在队头阻塞场景下，Cache-Aware 明确触发了队头阻塞规避，并将后续请求的平均首次 prefill 时延降低了 50%。

### mixed workload

| 指标 | FIFO | Cache-Aware | 变化 |
|---|---:|---:|---:|
| total_steps | 8 | 9 | +1 |
| preemptions | 3 | 1 | -66.7% |
| cached_tokens_reused | 36 | 16 | -55.6% |
| cache_hit_ratio | 83.72% | 39.02% | -44.70pt |
| out_of_order_prefills | 0 | 4 | +4 |
| avg first_prefill_step | 2.20 | 1.80 | -18.2% |
| avg finish_step | 5.00 | 5.83 | +16.6% |

结论：在混合场景下，当前 heuristic 能改善首次服务顺序并减少部分抢占，但没有同时最大化 cache reuse 与整体完成时延，体现了真实的系统调度 trade-off。

## 项目局限性

- 这是一个面向调度器路径的优化项目，不是完整的大模型服务系统。
- `bench_cache_scheduler.py` 是 scheduler-only microbenchmark，不直接等价于真实 GPU 上的端到端吞吐。
- 当前策略是启发式调度，不是生产级公平调度器。
- Prefix Cache 仍然是 block 粒度复用，不能复用非 block 对齐的部分前缀。

## 致谢

本仓库延续了轻量级大模型推理实现的代码风格，并在此基础上扩展出一个面向缓存感知 prefill 调度的优化实验项目。
