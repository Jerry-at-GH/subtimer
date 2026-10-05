import ctypes
import os
from functools import cache
from importlib.metadata import distribution
from pathlib import Path

import torch

GIB = 1024**3
RUNTIME = Path(
    distribution("nvidia-cuda-runtime").locate_file("nvidia/cu13/lib/libcudart.so.13")
)


@cache
def runtime():
    ctypes.CDLL(str(RUNTIME), mode=ctypes.RTLD_GLOBAL)
    torch.cuda.init()
    configure()
    return torch


class Graph:
    def __init__(self, forward, pool, stream):
        self.forward, self.pool, self.stream = forward, pool, stream
        self.graph = None

    def __call__(self, value):
        if self.graph is None:
            self.input = value.detach().clone()
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    self.forward(self.input)
            torch.cuda.current_stream().wait_stream(self.stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=self.pool):
                self.output = self.forward(self.input)
        else:
            self.input.copy_(value)
        self.graph.replay()
        return self.output


def headroom(total):
    wsl = "microsoft" in os.uname().release.lower()
    return max(2 * GIB if wsl else GIB, min(3 * GIB, total // 10))


def available():
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    cached = torch.cuda.memory_reserved() - allocated
    limit = int(total * torch.cuda.get_per_process_memory_fraction()) - allocated
    return max(0, min(free + cached - headroom(total), limit))


def configure():
    free, total = torch.cuda.mem_get_info()
    ceiling = max(0, free + torch.cuda.memory_reserved() - headroom(total))

    ceiling = min(ceiling, int(os.environ.get("SUBTIMER_GPU_BUDGET_BYTES", ceiling)))
    torch.cuda.set_per_process_memory_fraction(min(1.0, ceiling / total))


def concurrency(count):
    if count < 2:
        return 1
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    ram = int(memory["MemAvailable"].split()[0]) * 1024
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.is_file() and (value := limit.read_text().strip()) != "max":
        ram = min(ram, int(value) - int(limit.with_name("memory.current").read_text()))
    return max(
        1,
        min(
            count,
            4,
            len(os.sched_getaffinity(0)) // 2,
            available() // (6 * GIB),
            ram // (6 * GIB),
        ),
    )


def batch_size(maximum, *, weights, workspace):
    capacity = (available() - weights) // workspace
    if capacity < 1:
        raise RuntimeError("insufficient free VRAM for one alignment batch")

    return min(maximum, 1 << (int(capacity).bit_length() - 1))
