"""Layer streaming: run a stack of blocks whose weights don't fit in VRAM (or even in RAM).

Idea credit: the layer-streaming design of Soup (github.com/MakazhanAlpamys/Soup, Apache-2.0),
which fine-tunes 8B LLMs on 4 GB GPUs. This is an independent, inference-only implementation
for *any* stack of same-shaped blocks (transformer layers, residual MLP blocks, ...), aimed at
robot policies that run over large batches of simulated robots.

How it works:
  * every block's parameters and buffers are packed into one contiguous tensor per dtype, kept
    either in page-locked ("pinned") RAM or, when RAM is too small, in a file per block on disk;
  * the GPU holds only `n_slots` copies of the block, whose tensors are *views* into one flat
    device buffer per dtype, so loading a block into a slot is one `copy_` per dtype;
  * copies run on a side CUDA stream. CUDA events make the compute stream wait until a slot is
    loaded and the copy stream wait until a slot is no longer read, so block i+1 loads while
    block i computes. With the disk store, the file read for the next block also happens on the
    host while the GPU is busy;
  * `micro_batch` splits a huge batch into chunks *inside* each block, so activations fit a tiny
    card while every weight copy is still shared by the whole batch.

Peak VRAM ~ n_slots x one block + one micro-batch of activations, independent of model depth.
"""

from __future__ import annotations

import copy
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


def _tensors(module: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """Parameters and buffers in a stable order."""
    out = OrderedDict()
    for name, p in module.named_parameters():
        out["p:" + name] = p
    for name, b in module.named_buffers():
        out["b:" + name] = b
    return out


def _signature(module: nn.Module):
    return [(k, tuple(t.shape), t.dtype) for k, t in _tensors(module).items()]


def _bytes_view(t: torch.Tensor):
    """A writable numpy view of a contiguous CPU tensor's raw bytes (works for bf16 too)."""
    return t.view(torch.uint8).numpy()


@dataclass
class StreamStats:
    blocks: int
    block_bytes: int
    store_bytes: int
    slots: int
    store: str
    pinned: bool


class StreamedBlocks(nn.Module):
    """Drop-in replacement for running `blocks` one after another (inference only).

    `forward(x, *args, **kwargs)` computes `x = block(x, *args, **kwargs)` for every block. With
    `micro_batch`, `x` is split along dim 0 and `args`/`kwargs` are passed unchanged to every chunk.
    """

    def __init__(self, blocks, device: str | torch.device = "cuda", n_slots: int = 2, pin: bool = True,
                 store: str = "ram", cache_dir: str | os.PathLike | None = None, micro_batch: int | None = None):
        """`blocks` may be a list or any iterable (e.g. a generator that builds or loads one block
        at a time), so a model never has to exist twice in memory: each block is packed into the
        store and dropped right after."""
        super().__init__()
        if store not in ("ram", "disk"):
            raise ValueError("store must be 'ram' or 'disk'")
        it = iter(blocks)
        try:
            first = next(it)
        except StopIteration:
            raise ValueError("need at least one block") from None
        sig = _signature(first)
        template = copy.deepcopy(first)
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.pinned = pin and self.cuda
        self.store = store
        self.micro_batch = micro_batch
        if store == "disk":
            self.cache_dir = Path(cache_dir or tempfile.mkdtemp(prefix="spud_"))
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._keys = [k for k, _, _ in sig]
        self._dtypes = sorted({dt for _, _, dt in sig}, key=str)
        self._layout = {}  # key -> (dtype, offset, numel, shape)
        sizes = dict.fromkeys(self._dtypes, 0)
        for k, shape, dt in sig:
            n = 1
            for s in shape:
                n *= s
            self._layout[k] = (dt, sizes[dt], n, shape)
            sizes[dt] += n
        self._sizes = sizes

        self._host = []  # per block: {dtype: pinned tensor} (ram) or {dtype: file path} (disk)
        self._pack(first)
        del first
        for i, b in enumerate(it, 1):
            if _signature(b) != sig:
                raise ValueError(f"block {i} has a different structure than block 0; streaming needs same-shaped blocks")
            self._pack(b)
            del b
        self.n_blocks = len(self._host)
        self.n_slots = max(1, min(n_slots, self.n_blocks))

        # Device slots: a skeleton copy of the block whose tensors view a flat device buffer.
        self._slots, self._dev, self._staging = [], [], []
        for _ in range(self.n_slots):
            flats = {dt: torch.empty(sizes[dt], dtype=dt, device=self.device) for dt in self._dtypes}
            skel = copy.deepcopy(template).to(self.device)
            self._rebind(skel, flats)
            skel.eval()
            self._slots.append(skel)
            self._dev.append(flats)
            if store == "disk" and self.cuda:  # host bounce buffers for disk -> pinned -> device
                self._staging.append({dt: torch.empty(sizes[dt], dtype=dt).pin_memory() for dt in self._dtypes})
        self._slots = nn.ModuleList(self._slots)

        if self.cuda:
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._ready = [torch.cuda.Event() for _ in range(self.n_slots)]
            self._free = [torch.cuda.Event() for _ in range(self.n_slots)]
        block_bytes = sum(sizes[dt] * torch.empty(0, dtype=dt).element_size() for dt in self._dtypes)
        self.stats = StreamStats(self.n_blocks, block_bytes, block_bytes * self.n_blocks, self.n_slots,
                                 store, self.pinned)

    # -------------------------------------------------------------- packing --

    def _pack(self, block: nn.Module):
        flats = {}
        for dt in self._dtypes:
            flat = torch.empty(self._sizes[dt], dtype=dt)
            if self.pinned and self.store == "ram":
                flat = flat.pin_memory()
            flats[dt] = flat
        for k, t in _tensors(block).items():
            dt, off, n, _ = self._layout[k]
            flats[dt][off:off + n].copy_(t.detach().reshape(-1))
        if self.store == "ram":
            self._host.append(flats)
            return
        paths = {}
        for dt, flat in flats.items():
            path = self.cache_dir / f"block{len(self._host):05d}_{str(dt).split('.')[-1]}.bin"
            _bytes_view(flat).tofile(path)
            paths[dt] = path
        self._host.append(paths)

    def _rebind(self, module: nn.Module, flats: dict):
        """Point every parameter/buffer of `module` at a slice of the flat device buffers."""
        for k in self._keys:
            dt, off, n, shape = self._layout[k]
            view = flats[dt][off:off + n].view(shape)
            kind, name = k.split(":", 1)
            *path, leaf = name.split(".")
            owner = module
            for part in path:
                owner = getattr(owner, part)
            if kind == "p":
                owner._parameters[leaf] = nn.Parameter(view, requires_grad=False)
            else:
                owner._buffers[leaf] = view

    # -------------------------------------------------------------- loading --

    @staticmethod
    def _read(path: Path, into: torch.Tensor):
        with open(path, "rb", buffering=0) as f:
            f.readinto(memoryview(_bytes_view(into)))

    def _load(self, block: int):
        s = block % self.n_slots
        src = self._host[block]
        if not self.cuda:
            for dt in self._dtypes:
                if self.store == "ram":
                    self._dev[s][dt].copy_(src[dt])
                else:
                    self._read(src[dt], self._dev[s][dt])
            return
        if self.store == "disk":
            # The previous copy out of this bounce buffer must be done before we refill it. The
            # GPU is meanwhile busy with the current block, so this file read overlaps compute.
            self._ready[s].synchronize()
            for dt in self._dtypes:
                self._read(src[dt], self._staging[s][dt])
            src = self._staging[s]
        self._copy_stream.wait_event(self._free[s])  # don't overwrite a slot still being read
        with torch.cuda.stream(self._copy_stream):
            for dt in self._dtypes:
                self._dev[s][dt].copy_(src[dt], non_blocking=True)
            self._ready[s].record(self._copy_stream)

    @torch.no_grad()
    def forward(self, x, *args, **kwargs):
        chunks = [x] if not self.micro_batch or x.shape[0] <= self.micro_batch else list(x.split(self.micro_batch))
        if self.cuda:
            compute = torch.cuda.current_stream(self.device)
            for e in self._free:
                e.record(compute)
        for i in range(self.n_slots):
            self._load(i)
        for i in range(self.n_blocks):
            s = i % self.n_slots
            if self.cuda:
                compute.wait_event(self._ready[s])
            blk = self._slots[s]
            chunks = [blk(c, *args, **kwargs) for c in chunks]
            if self.cuda:
                self._free[s].record(compute)
            if i + self.n_slots < self.n_blocks:
                self._load(i + self.n_slots)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def resident_copy(blocks: list[nn.Module], device="cuda") -> nn.Module:
    """The reference: the same blocks fully on the device, run one after another."""
    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(copy.deepcopy(b) for b in blocks)

        @torch.no_grad()
        def forward(self, x, *args, **kwargs):
            for b in self.blocks:
                x = b(x, *args, **kwargs)
            return x

    return Seq().to(device).eval()


@torch.no_grad()
def time_one_block(block: nn.Module, x: torch.Tensor, reps: int = 5) -> float:
    """Seconds for one resident forward of `block` on `x`: the planner's compute calibration."""
    blk = copy.deepcopy(block).to(x.device).eval()
    blk(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    for _ in range(reps):
        blk(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps
