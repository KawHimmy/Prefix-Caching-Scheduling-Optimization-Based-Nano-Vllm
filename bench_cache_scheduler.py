import argparse
from statistics import mean
from time import perf_counter
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


STATS_KEYS = (
    "prefill_batches",
    "decode_batches",
    "prefill_tokens_scheduled",
    "cached_blocks_reused",
    "cached_tokens_reused",
    "prefill_candidates_scanned",
    "out_of_order_prefills",
    "head_of_line_prefills",
    "preemptions",
)


def make_config(policy: str, scan_window: int, num_blocks: int, block_size: int, max_tokens: int, max_seqs: int):
    return SimpleNamespace(
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_tokens,
        eos=-1,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
        prefill_schedule_policy=policy,
        prefill_scan_window=scan_window,
        collect_scheduler_stats=True,
    )


def reset_sequence_counter():
    from itertools import count

    Sequence.counter = count()


def reset_stats(scheduler: Scheduler):
    for key in STATS_KEYS:
        scheduler.stats[key] = 0


def add_requests(scheduler: Scheduler, prompts: list[list[int]], max_tokens: int = 3):
    for prompt in prompts:
        scheduler.add(Sequence(prompt, SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=max_tokens)))


def run_steps(scheduler: Scheduler, max_steps: int = 10000):
    steps = 0
    prefill_orders = []
    first_prefill_step = {}
    finish_step = {}
    while not scheduler.is_finished():
        if steps >= max_steps:
            raise RuntimeError(f"scheduler did not finish within {max_steps} steps")
        seqs, is_prefill = scheduler.schedule()
        if is_prefill:
            for seq in seqs:
                prefill_orders.append(seq.seq_id)
                first_prefill_step.setdefault(seq.seq_id, steps)
        fake_token_ids = [1000 + steps] * len(seqs)
        scheduler.postprocess(seqs, fake_token_ids, is_prefill)
        for seq in seqs:
            if seq.is_finished:
                finish_step.setdefault(seq.seq_id, steps)
        steps += 1
    return steps, prefill_orders, first_prefill_step, finish_step


def warm_prefix_cache(scheduler: Scheduler, prompts: list[list[int]], block_size: int):
    add_requests(scheduler, prompts, max_tokens=1)
    run_steps(scheduler)
    reset_stats(scheduler)
    Sequence.block_size = block_size


def prime_running(scheduler: Scheduler, prompts: list[list[int]], max_tokens: int):
    add_requests(scheduler, prompts, max_tokens=max_tokens)
    while scheduler.waiting:
        seqs, is_prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [777] * len(seqs), is_prefill)
    reset_stats(scheduler)


def shared_prefix_workload(scheduler: Scheduler, block_size: int):
    shared = [1, 2, 3, 4, 5, 6, 7, 8]
    warm_prefix_cache(scheduler, [shared + [90, 91, 92, 93]], block_size)
    prompts = [
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
        shared + [10, 11, 12, 13],
        [50, 51, 52, 53, 54, 55, 56, 57],
        shared + [20, 21, 22, 23],
    ]
    add_requests(scheduler, prompts, max_tokens=2)


def hol_workload(scheduler: Scheduler, block_size: int):
    prime_running(
        scheduler,
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [11, 12, 13, 14, 15, 16, 17, 18],
        ],
        max_tokens=4,
    )
    prompts = [
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        [200, 201, 202, 203],
        [210, 211, 212, 213],
    ]
    add_requests(scheduler, prompts, max_tokens=2)


def mixed_workload(scheduler: Scheduler, block_size: int):
    shared = [7, 8, 9, 10, 11, 12, 13, 14]
    warm_prefix_cache(scheduler, [shared + [80, 81, 82, 83]], block_size)
    prime_running(scheduler, [[300, 301, 302, 303, 304, 305, 306, 307]], max_tokens=3)
    prompts = [
        [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411],
        shared + [20, 21, 22, 23],
        [500, 501, 502, 503],
        shared + [30, 31, 32, 33],
        [600, 601, 602, 603, 604, 605, 606, 607],
    ]
    add_requests(scheduler, prompts, max_tokens=2)


def build_workload(scheduler: Scheduler, workload: str, block_size: int):
    if workload == "shared-prefix":
        shared_prefix_workload(scheduler, block_size)
    elif workload == "hol":
        hol_workload(scheduler, block_size)
    elif workload == "mixed":
        mixed_workload(scheduler, block_size)
    else:
        raise ValueError(f"unknown workload: {workload}")


def summarize_steps(step_map: dict[int, int]):
    values = list(step_map.values())
    return dict(
        min=min(values) if values else -1,
        max=max(values) if values else -1,
        avg=mean(values) if values else -1,
    )


def run_policy(policy: str, workload: str, scan_window: int):
    reset_sequence_counter()
    block_size = 4
    config_by_workload = {
        "shared-prefix": dict(num_blocks=16, max_tokens=8, max_seqs=2),
        "hol": dict(num_blocks=6, max_tokens=8, max_seqs=3),
        "mixed": dict(num_blocks=10, max_tokens=8, max_seqs=3),
    }
    config = make_config(policy, scan_window, block_size=block_size, **config_by_workload[workload])
    Sequence.block_size = block_size
    scheduler = Scheduler(config)
    build_workload(scheduler, workload, block_size)
    start = perf_counter()
    steps, prefill_orders, first_prefill_step, finish_step = run_steps(scheduler)
    elapsed = perf_counter() - start
    stats = scheduler.get_stats()
    total_requests = len(first_prefill_step)
    stats.update(
        workload=workload,
        total_steps=steps,
        simulation_seconds=elapsed,
        total_requests=total_requests,
        prefill_order=prefill_orders,
        first_prefill_step=first_prefill_step,
        finish_step=finish_step,
        first_prefill_summary=summarize_steps(first_prefill_step),
        finish_summary=summarize_steps(finish_step),
        cache_hit_ratio=(stats["cached_blocks_reused"] / stats["prefill_tokens_scheduled"] * block_size) if stats["prefill_tokens_scheduled"] else 0.0,
    )
    return stats


def print_stats(stats: dict):
    print(f"\n[{stats['workload']}] policy={stats['policy']} scan_window={stats['scan_window']}")
    print(f"  total_steps={stats['total_steps']} simulation_seconds={stats['simulation_seconds']:.6f}")
    print(f"  prefill_batches={stats['prefill_batches']} decode_batches={stats['decode_batches']}")
    print(f"  prefill_tokens_scheduled={stats['prefill_tokens_scheduled']}")
    print(f"  cached_blocks_reused={stats['cached_blocks_reused']} cached_tokens_reused={stats['cached_tokens_reused']} cache_hit_ratio={stats['cache_hit_ratio']:.2%}")
    print(f"  candidates_scanned={stats['prefill_candidates_scanned']}")
    print(f"  out_of_order_prefills={stats['out_of_order_prefills']} head_of_line_prefills={stats['head_of_line_prefills']}")
    print(f"  preemptions={stats['preemptions']} free_blocks={stats['free_blocks']} used_blocks={stats['used_blocks']}")
    print(f"  prefill_order={stats['prefill_order']}")
    print(f"  first_prefill_step={stats['first_prefill_step']}")
    print(f"  first_prefill_summary={stats['first_prefill_summary']}")
    print(f"  finish_step={stats['finish_step']}")
    print(f"  finish_summary={stats['finish_summary']}")


def main():
    parser = argparse.ArgumentParser(description="Compare FIFO and cache-aware prefill scheduling on synthetic workloads.")
    parser.add_argument("--policy", choices=("fifo", "cache_aware", "both"), default="both")
    parser.add_argument("--workload", choices=("shared-prefix", "hol", "mixed"), default="shared-prefix")
    parser.add_argument("--scan-window", type=int, default=8)
    args = parser.parse_args()

    policies = ("fifo", "cache_aware") if args.policy == "both" else (args.policy,)
    for policy in policies:
        print_stats(run_policy(policy, args.workload, args.scan_window))


if __name__ == "__main__":
    main()
