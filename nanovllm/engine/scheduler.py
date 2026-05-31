from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import AllocationEstimate, BlockManager


@dataclass(slots=True)
class PrefillCandidate:
    index: int
    seq: Sequence
    num_cached_blocks: int
    num_tokens: int
    estimate: AllocationEstimate | None
    head_blocked: bool = False


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.prefill_schedule_policy = config.prefill_schedule_policy
        self.prefill_scan_window = config.prefill_scan_window
        self.collect_scheduler_stats = config.collect_scheduler_stats
        self.stats = dict(
            prefill_batches=0,
            decode_batches=0,
            prefill_tokens_scheduled=0,
            cached_blocks_reused=0,
            cached_tokens_reused=0,
            prefill_candidates_scanned=0,
            out_of_order_prefills=0,
            head_of_line_prefills=0,
            preemptions=0,
        )
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def get_stats(self):
        stats = dict(self.stats)
        stats.update(
            policy=self.prefill_schedule_policy,
            scan_window=self.prefill_scan_window,
            waiting=len(self.waiting),
            running=len(self.running),
            free_blocks=len(self.block_manager.free_block_ids),
            used_blocks=len(self.block_manager.used_block_ids),
        )
        return stats

    def _record(self, key: str, value: int = 1):
        if self.collect_scheduler_stats:
            self.stats[key] += value

    def _remove_waiting(self, index: int):
        if index == 0:
            return self.waiting.popleft()
        self.waiting.rotate(-index)
        seq = self.waiting.popleft()
        self.waiting.rotate(index)
        return seq

    def _fresh_candidate(self, index: int, seq: Sequence, remaining: int, allow_chunked: bool, head_blocked: bool = False):
        estimate = self.block_manager.estimate_allocation(seq)
        if not estimate.can_allocate:
            return None
        num_tokens = seq.num_tokens - estimate.num_cached_blocks * self.block_size
        if not allow_chunked and remaining < num_tokens:
            return None
        return PrefillCandidate(index, seq, estimate.num_cached_blocks, num_tokens, estimate, head_blocked)

    def _chunked_candidate(self, index: int, seq: Sequence, remaining: int, allow_chunked: bool):
        num_tokens = seq.num_tokens - seq.num_cached_tokens
        if not allow_chunked and remaining < num_tokens:
            return None
        return PrefillCandidate(index, seq, 0, num_tokens, None)

    def _select_fifo_candidate(self, remaining: int, allow_chunked: bool):
        seq = self.waiting[0]
        if seq.block_table:
            return self._chunked_candidate(0, seq, remaining, allow_chunked)
        return self._fresh_candidate(0, seq, remaining, allow_chunked)

    def _select_cache_aware_candidate(self, remaining: int, allow_chunked: bool):
        scan_count = min(len(self.waiting), self.prefill_scan_window)
        self._record("prefill_candidates_scanned", scan_count)
        head_blocked = False
        if self.waiting[0].block_table:
            return self._chunked_candidate(0, self.waiting[0], remaining, allow_chunked)
        head_estimate = self.block_manager.estimate_allocation(self.waiting[0])
        head_blocked = not head_estimate.can_allocate
        best_candidate = None
        best_score = None
        for index in range(scan_count):
            seq = self.waiting[index]
            if seq.block_table:
                candidate = self._chunked_candidate(index, seq, remaining, allow_chunked)
                score = (2, 0, 0, -candidate.num_tokens, -index) if candidate else None
            else:
                if index == 0:
                    estimate = head_estimate
                else:
                    estimate = self.block_manager.estimate_allocation(seq)
                if not estimate.can_allocate:
                    continue
                num_tokens = seq.num_tokens - estimate.num_cached_blocks * self.block_size
                if not allow_chunked and remaining < num_tokens:
                    continue
                candidate = PrefillCandidate(index, seq, estimate.num_cached_blocks, num_tokens, estimate, head_blocked)
                score = (
                    int(estimate.num_cached_blocks > 0),
                    estimate.num_cached_blocks,
                    -estimate.num_required_blocks,
                    -num_tokens,
                    -index,
                )
            if candidate is not None and (best_score is None or score > best_score):
                best_candidate = candidate
                best_score = score
        return best_candidate

    def _select_prefill_candidate(self, remaining: int, allow_chunked: bool):
        if not self.waiting:
            return None
        if self.prefill_schedule_policy == "cache_aware":
            return self._select_cache_aware_candidate(remaining, allow_chunked)
        return self._select_fifo_candidate(remaining, allow_chunked)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            candidate = self._select_prefill_candidate(remaining, allow_chunked=not scheduled_seqs)
            if candidate is None:
                break
            seq = candidate.seq
            if not seq.block_table:
                self.block_manager.allocate(seq, candidate.num_cached_blocks)
            seq.num_scheduled_tokens = min(candidate.num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if self.collect_scheduler_stats:
                self.stats["prefill_tokens_scheduled"] += seq.num_scheduled_tokens
                self.stats["cached_blocks_reused"] += candidate.num_cached_blocks
                self.stats["cached_tokens_reused"] += candidate.num_cached_blocks * self.block_size
                if candidate.index > 0:
                    self.stats["out_of_order_prefills"] += 1
                    if candidate.head_blocked:
                        self.stats["head_of_line_prefills"] += 1
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self._remove_waiting(candidate.index)
                self.running.append(seq)
            elif candidate.index > 0:
                self._remove_waiting(candidate.index)
                self.waiting.appendleft(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            self._record("prefill_batches")
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self._record("decode_batches")
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self._record("preemptions")
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
