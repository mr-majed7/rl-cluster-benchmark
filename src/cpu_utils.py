"""CPU-specific optimizations for distributed training."""

import os
from typing import Optional

import torch


def setup_cpu_optimization(num_threads: Optional[int] = None) -> int:
    """Configure PyTorch for optimal CPU performance.

    Args:
        num_threads: Number of threads to use (None for auto-detect)

    Returns:
        Number of threads configured
    """
    if num_threads is None:
        # Auto-detect optimal thread count
        num_threads = os.cpu_count() or 1

    # Set PyTorch thread count
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)

    # Set NumPy/OpenBLAS thread count
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    return num_threads


def get_cpu_info() -> dict:
    """Get CPU information for the current system.

    Returns:
        Dictionary with CPU information
    """
    info = {
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }

    # Try to get CPU model
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError, IOError):
        info["cpu_model"] = "Unknown"

    return info


def print_cpu_info():
    """Print CPU configuration information."""
    info = get_cpu_info()
    print("\n" + "=" * 60)
    print("CPU Configuration")
    print("=" * 60)
    print(f"CPU Model: {info.get('cpu_model', 'Unknown')}")
    print(f"CPU Cores: {info['cpu_count']}")
    print(f"PyTorch Threads: {info['torch_threads']}")
    print(f"PyTorch Interop Threads: {info['torch_interop_threads']}")
    print("=" * 60 + "\n")


def optimize_batch_for_cpu(
    total_samples: int, num_workers: int, memory_limit_gb: Optional[float] = None
) -> int:
    """Calculate optimal batch size for CPU training.

    Args:
        total_samples: Total number of samples to process
        num_workers: Number of worker processes/threads
        memory_limit_gb: Memory limit in GB (None for auto-detect)

    Returns:
        Optimal batch size
    """
    if memory_limit_gb is None:
        # Try to detect available memory
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "MemAvailable" in line:
                        # Get available memory in GB
                        mem_kb = int(line.split()[1])
                        memory_limit_gb = mem_kb / (1024 * 1024)
                        break
        except (FileNotFoundError, PermissionError, IOError, ValueError):
            memory_limit_gb = 4.0  # Conservative default

    # Conservative estimate: use 50% of available memory
    usable_memory_gb = memory_limit_gb * 0.5

    # Rough estimate: ~1MB per sample for Procgen
    sample_size_gb = 0.001
    max_batch_from_memory = int(usable_memory_gb / sample_size_gb)

    # Ensure batch size is divisible by num_workers for even distribution
    batch_size = min(total_samples, max_batch_from_memory)
    batch_size = (batch_size // num_workers) * num_workers

    return max(batch_size, num_workers)


class CPUMemoryMonitor:
    """Monitor CPU memory usage during training."""

    def __init__(self):
        self.peak_memory_gb = 0.0

    def update(self):
        """Update memory statistics."""
        try:
            import psutil

            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            self.peak_memory_gb = max(self.peak_memory_gb, memory_gb)
        except (ImportError, AttributeError):
            pass

    def get_stats(self) -> dict:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        try:
            import psutil

            process = psutil.Process()
            current_memory_gb = process.memory_info().rss / (1024**3)
            return {
                "current_memory_gb": current_memory_gb,
                "peak_memory_gb": self.peak_memory_gb,
            }
        except (ImportError, AttributeError):
            return {"current_memory_gb": 0.0, "peak_memory_gb": self.peak_memory_gb}
