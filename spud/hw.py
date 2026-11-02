"""What are we running on? Detects the machine and (optionally) pretends it is a smaller one."""

from __future__ import annotations

import ctypes
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass

GiB = 2**30


@dataclass
class Hardware:
    cpu_threads: int
    ram_total: float          # GiB
    ram_free: float           # GiB
    gpu: str | None
    vram_total: float         # GiB
    vram_free: float          # GiB
    h2d_gbps: float | None    # measured pinned host -> device bandwidth, GiB/s
    os: str
    gpu_tflops: float | None = None   # measured fp16 matmul throughput
    cpu_gflops: float | None = None   # measured fp32 matmul throughput
    limited: bool = False     # True if any limit below was applied

    def as_dict(self):
        return asdict(self)


def _ram() -> tuple[float, float]:
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / GiB, m.ullAvailPhys / GiB
    if os.path.exists("/proc/meminfo"):
        info = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":")
            info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"] / GiB, info.get("MemAvailable", info["MemFree"]) / GiB
    import subprocess  # macOS
    total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
    return total / GiB, total / GiB * 0.5


def _bandwidth(torch) -> float:
    n = 256 * 2**20
    src = torch.empty(n, dtype=torch.uint8).pin_memory()
    dst = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(4):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return 4 * n / (time.perf_counter() - t0) / GiB


def _matmul_flops(torch, device, dtype, n) -> float:
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    reps = 8 if device == "cuda" else 3
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    return 2 * n**3 * reps / (time.perf_counter() - t0)


def detect(measure: bool = True) -> Hardware:
    total, free = _ram()
    gpu, vt, vf, bw, tf, cf = None, 0.0, 0.0, None, None, None
    try:
        import torch

        if measure:
            cf = _matmul_flops(torch, "cpu", torch.float32, 1024) / 1e9
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            f, t = torch.cuda.mem_get_info()
            vt, vf = t / GiB, f / GiB
            if measure:
                bw = _bandwidth(torch)
                tf = _matmul_flops(torch, "cuda", torch.float16, 4096) / 1e12
    except ImportError:
        pass
    return Hardware(os.cpu_count() or 1, total, free, gpu, vt, vf, bw, f"{platform.system()} {platform.release()}", tf, cf)


def apply_limits(hw: Hardware, vram: float | None = None, ram: float | None = None,
                 threads: int | None = None, cpu_only: bool = False) -> Hardware:
    """Make this machine behave like a weaker one, so you can check a plan before buying anything.

    * vram: caps PyTorch's CUDA allocator (`set_per_process_memory_fraction`); going over raises OOM.
    * ram: a budget the planner and the host weight store respect (it can't be enforced by the OS).
    * threads: caps torch intra-op threads and the simulator's thread pool.
    * cpu_only: hides the GPU from SPUD.
    """
    import torch

    out = Hardware(**hw.as_dict())
    if cpu_only or not torch.cuda.is_available():
        out.gpu, out.vram_total, out.vram_free, out.h2d_gbps, out.gpu_tflops = None, 0.0, 0.0, None, None
    elif vram is not None:
        frac = min(1.0, vram / hw.vram_total)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        out.vram_total = vram
        out.vram_free = min(hw.vram_free, vram)
    if ram is not None:
        out.ram_total = min(hw.ram_total, ram)
        out.ram_free = min(hw.ram_free, ram)
    if threads is not None:
        torch.set_num_threads(threads)
        if out.cpu_gflops and hw.cpu_threads:
            out.cpu_gflops *= min(1.0, threads / hw.cpu_threads)  # rough: matmul scales ~linearly with cores
        out.cpu_threads = threads
    out.limited = any(v is not None for v in (vram, ram, threads)) or cpu_only
    return out
