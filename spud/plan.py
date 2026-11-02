"""Will it fit, and how fast will it go? A pre-flight check before anything is allocated.

Tier choice for a model of P parameters (W bytes of weights) running a batch of B tokens:
  resident  W + activations fit in ~85% of free VRAM: plain GPU inference
  ram       stream from pinned RAM; the store must fit the RAM headroom below
  disk      stream from an on-disk cache when even RAM is too small
  cpu       no usable GPU: run on the CPU, streaming from disk if needed

Step time = max(weight transfer, compute) because SPUD overlaps the two:
  transfer = W / bandwidth (PCIe for "ram", disk read speed for "disk", 0 for resident)
  compute  = 2 * P * B / (peak FLOPS * efficiency)
"""

from __future__ import annotations

from dataclasses import dataclass

from .hw import GiB, Hardware

# RAM headroom rules borrowed from Soup's layer-streaming planner (Apache-2.0): the store must use
# under 70% of *free* RAM and, because pinned memory can't be paged out, under 55% of *total* RAM.
RAM_FREE_HEADROOM = 0.70
RAM_TOTAL_HEADROOM = 0.55
VRAM_HEADROOM = 0.85
GPU_EFFICIENCY = 0.85      # measured: 1B blocks at 8192 tokens reach ~85% of the matmul peak
CPU_EFFICIENCY = 0.6
DEFAULT_DISK_GBPS = 1.5    # conservative NVMe read; `spud plan --measure-disk` measures it


@dataclass
class Plan:
    tier: str
    fits: bool
    step_seconds: float
    transfer_seconds: float
    compute_seconds: float
    peak_vram_gib: float
    host_store_gib: float
    bound: str          # "transfer" or "compute"
    notes: list[str]
    micro_batch: int | None = None

    @property
    def tokens_per_second(self) -> float:
        return self._batch / self.step_seconds if self.step_seconds > 0 else float("inf")


def activation_bytes(batch_tokens: int, d: int, mlp_ratio: int = 4, bytes_per: int = 2) -> int:
    """Peak extra memory inside one block. Measured, not derived: 13.0-16.5 x (tokens * d * 2 bytes)
    for our blocks at d = 1856-2560 (q/k/v copies, the MLP hidden and its GELU coexist). 17 adds margin."""
    return int(batch_tokens * d * bytes_per * 17 * (mlp_ratio / 4))


def batch_bytes(batch_tokens: int, d: int, bytes_per: int = 2) -> int:
    """The whole batch's residual stream is alive across blocks: inputs, outputs and the final concat."""
    return 3 * batch_tokens * d * bytes_per


ALLOCATOR_SLACK = 1.15  # caching-allocator fragmentation: we saw ~180 MiB reserved-but-unused at 1 GiB


def plan_model(hw: Hardware, params: int, weight_bytes: int, block_bytes: int, d: int, batch_tokens: int,
               n_slots: int = 2, force: str | None = None, block_seconds: float | None = None,
               depth: int | None = None, tokens_per_item: int = 1) -> Plan:
    """`block_seconds` (one measured block forward at this batch, see `stream.time_one_block`)
    replaces the FLOPS estimate: matmul efficiency depends on the exact shapes (we measured 12.7
    TFLOPS at d=1856 but 9.2 at d=2304 on the same laptop GPU)."""
    notes: list[str] = []
    act = activation_bytes(batch_tokens, d)
    flops = 2 * params * batch_tokens
    gpu = hw.gpu is not None and hw.vram_total > 0
    compute_gpu = flops / ((hw.gpu_tflops or 5.0) * 1e12 * GPU_EFFICIENCY) if gpu else None
    if gpu and block_seconds is not None and depth:
        compute_gpu = block_seconds * depth
        notes.append("compute time calibrated on one real block")
    compute_cpu = flops / ((hw.cpu_gflops or 50.0) * 1e9 * CPU_EFFICIENCY)
    ram_ok = weight_bytes < RAM_FREE_HEADROOM * hw.ram_free * GiB and weight_bytes < RAM_TOTAL_HEADROOM * hw.ram_total * GiB
    resident_ok = gpu and weight_bytes + act < VRAM_HEADROOM * hw.vram_free * GiB
    vram_budget = VRAM_HEADROOM * hw.vram_free * GiB
    stream_vram = (n_slots * block_bytes + act + batch_bytes(batch_tokens, d)) * ALLOCATOR_SLACK
    stream_ok = gpu and stream_vram < vram_budget
    micro = None
    budget = vram_budget / ALLOCATOR_SLACK - n_slots * block_bytes - batch_bytes(batch_tokens, d)
    if gpu and not stream_ok and budget > 0:
        # Activations are the problem, not the weights: split the batch inside each block.
        micro_tokens = int(budget // activation_bytes(1, d))
        if micro_tokens >= tokens_per_item:
            micro = max(1, micro_tokens // tokens_per_item)
            stream_vram = (n_slots * block_bytes + activation_bytes(micro * tokens_per_item, d)
                           + batch_bytes(batch_tokens, d)) * ALLOCATOR_SLACK
            stream_ok = True
            notes.append(f"activations don't fit: micro-batches of {micro:,} items inside each block")

    tier = force
    if tier is None:
        if resident_ok:
            tier = "resident"
        elif stream_ok:
            tier = "ram" if ram_ok else "disk"
        else:
            tier = "cpu"
    if tier == "resident":
        return Plan("resident", bool(resident_ok), compute_gpu or 0.0, 0.0, compute_gpu or 0.0,
                    (weight_bytes + act) / GiB, 0.0, "compute", notes)._with_batch(batch_tokens)
    if tier in ("ram", "disk"):
        bw = (hw.h2d_gbps or 6.0) if tier == "ram" else DEFAULT_DISK_GBPS
        transfer = weight_bytes / (bw * GiB)
        step = max(transfer, compute_gpu or 0.0)
        bound = "transfer" if transfer > (compute_gpu or 0.0) else "compute"
        if bound == "transfer" and compute_gpu:
            breakeven = int(batch_tokens * transfer / compute_gpu)
            notes.append(f"transfer-bound: batches above ~{breakeven:,} tokens hide the copies completely")
        if tier == "disk":
            notes.append("weights exceed the RAM headroom, so each step re-reads them from disk")
        return Plan(tier, bool(stream_ok) and (tier == "disk" or ram_ok), step, transfer, compute_gpu or 0.0,
                    stream_vram / GiB, weight_bytes / GiB if tier == "ram" else n_slots * block_bytes / GiB,
                    bound, notes, micro)._with_batch(batch_tokens)
    notes.append("no GPU budget left: running on the CPU")
    transfer = 0.0 if ram_ok else weight_bytes / (DEFAULT_DISK_GBPS * GiB)
    return Plan("cpu", True, max(transfer, compute_cpu), transfer, compute_cpu, 0.0,
                weight_bytes / GiB if ram_ok else 0.0, "compute" if compute_cpu >= transfer else "transfer",
                notes)._with_batch(batch_tokens)


def _with_batch(self, b):
    self._batch = b
    return self


Plan._with_batch = _with_batch
