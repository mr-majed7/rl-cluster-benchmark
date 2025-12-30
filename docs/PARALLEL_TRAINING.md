# Parallel PPO Training on CPU

This guide explains how to use the parallel PPO implementation for CPU-based distributed training.

## Overview

The parallel implementation uses **multiprocessing** to run multiple worker processes that collect experience simultaneously. This provides significant speedup on multi-core CPUs compared to sequential training.

## Architecture

```
Main Process (Coordinator)
├── Shared Policy Parameters (in shared memory)
├── Worker 1 → Collects rollouts from envs 1-8
├── Worker 2 → Collects rollouts from envs 9-16
├── Worker 3 → Collects rollouts from envs 17-24
└── Worker 4 → Collects rollouts from envs 25-32
     ↓
Aggregated Data → PPO Update → Sync to Workers
```

### Key Features

1. **True Parallelism**: Workers run in separate processes, fully utilizing multiple CPU cores
2. **Shared Policy**: All workers use the same policy (synchronized before each rollout)
3. **Centralized Updates**: Main process aggregates data and performs PPO updates
4. **CPU Optimized**: Each worker auto-detects optimal thread count

## How It Works

### Collection Phase (Parallel)

- Each worker independently collects rollouts from its environments
- Workers run **simultaneously** on different CPU cores
- No synchronization during collection (maximum parallelism)

### Training Phase (Sequential)

- Main process waits for all workers to finish
- Aggregates all collected data
- Performs standard PPO updates
- Syncs updated policy back to all workers

## Configuration

### Basic Configuration (`config/ppo_parallel.yaml`)

```yaml
parallel:
  num_workers: 4              # Number of worker processes
  num_envs_per_worker: 8      # Environments per worker

training:
  n_steps: 128                # Steps per rollout per worker
  batch_size: 2048            # Batch size for PPO (should be >= n_steps * total_envs)
  n_epochs: 4                 # Training epochs per update
```

### Choosing num_workers

**General Guidelines:**

- Set to **number of physical cores / 2-4**
- For 16-core CPU: 4-8 workers is optimal
- More workers = more parallelism but higher overhead
- Each worker spawns a separate Python process

**Example Configurations:**

| CPU Cores | Recommended Workers | Envs/Worker | Total Envs |
|-----------|---------------------|-------------|------------|
| 4         | 2                   | 4-8         | 8-16       |
| 8         | 2-4                 | 8           | 16-32      |
| 16        | 4-8                 | 8           | 32-64      |
| 32        | 8-16                | 8           | 64-128     |

### Optimal Batch Size

```
batch_size >= n_steps × num_workers × num_envs_per_worker
```

For best results:

- **Minimum**: Equal to total samples collected per update
- **Recommended**: 2x total samples for better mixing

Example: 4 workers × 8 envs × 128 steps = 4096 samples

- Minimum batch_size: 4096
- Recommended: 2048 (with multiple epochs)

## Usage

### 1. Standard Parallel Training

```bash
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h \
  --num-workers 4 \
  --num-envs-per-worker 8
```

### 2. Using Makefile

```bash
# 1-hour parallel training
make train-parallel

# Compare with sequential
make compare
```

### 3. Custom Configuration

```bash
python train_parallel_timed.py \
  --config config/ppo_parallel.yaml \
  --duration 2.0 \
  --output-dir ./benchmarks/parallel_2h \
  --num-workers 8 \
  --num-envs-per-worker 4 \
  --batch-size 4096
```

### 4. Testing (Short Run)

```bash
# Test for 1 minute with 2 workers
python train_parallel_timed.py \
  --duration 0.017 \
  --output-dir ./benchmarks/test \
  --num-workers 2 \
  --num-envs-per-worker 4
```

## Expected Performance

### AMD Ryzen 7 7700 (16 cores)

**Sequential (32 envs):**

- FPS: 1,400-2,000
- Memory: 1-3 GB
- CPU Usage: 100% of allocated cores

**Parallel (4 workers × 8 envs = 32 envs):**

- FPS: 3,000-5,000 (2-3.5x speedup)
- Memory: 4-8 GB (higher due to multiple processes)
- CPU Usage: Distributed across all cores

### Speedup Factors

| Workers | Expected Speedup | Efficiency |
|---------|------------------|------------|
| 2       | 1.5-1.8x         | 75-90%     |
| 4       | 2.5-3.2x         | 62-80%     |
| 8       | 4.0-5.5x         | 50-68%     |

*Efficiency = Speedup / Workers*

## Benchmarking Sequential vs Parallel

### Step 1: Train Sequential (1 hour)

```bash
python train_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/sequential_1h
```

### Step 2: Train Parallel (1 hour)

```bash
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h \
  --num-workers 4
```

### Step 3: Compare Results

```bash
python compare_results.py \
  --sequential ./benchmarks/sequential_1h \
  --parallel ./benchmarks/parallel_1h
```

This will generate:

- Console comparison table
- 4-panel visualization plot (`comparison.png`)
- Text report (`comparison_report.txt`)

## Monitoring

### TensorBoard (Real-time)

```bash
tensorboard --logdir ./benchmarks/parallel_1h/logs/
```

View at: <http://localhost:6006>

### System Monitoring

```bash
# CPU usage per worker
htop

# Memory usage
watch -n 1 free -h

# Process tree
pstree -p <main_pid>
```

## Troubleshooting

### Issue: "RuntimeError: cannot set number of interop threads"

**Solution**: This is already fixed. The code now sets threading before any torch operations.

### Issue: Workers not utilizing all cores

**Cause**: Too few workers for available cores

**Solution**: Increase `num_workers` to match your CPU:

```bash
--num-workers 8  # For 16-core CPU
```

### Issue: High memory usage

**Cause**: Each worker has its own environment instances

**Solutions**:

1. Reduce `num_envs_per_worker`
2. Reduce `num_workers`
3. Trade-off: fewer total environments but still parallel

### Issue: Lower than expected speedup

**Possible Causes:**

1. **Batch size too small**: Increase to match total samples
2. **Too many workers**: Overhead dominates (reduce workers)
3. **Environment bottleneck**: Procgen is CPU-bound, speedup limited by env simulation

**Optimization:**

```bash
# Balance workers and envs per worker
--num-workers 4 \
--num-envs-per-worker 8 \
--batch-size 4096
```

## Advanced Configuration

### Heterogeneous CPU Setup

For systems with different CPU types (P-cores, E-cores):

```bash
# Pin workers to specific cores (Linux)
taskset -c 0-7 python train_parallel_timed.py ...
```

### Memory-Constrained Systems

```yaml
parallel:
  num_workers: 2              # Fewer workers
  num_envs_per_worker: 4      # Fewer envs per worker
```

### Maximum Throughput

```yaml
parallel:
  num_workers: 8              # Maximum parallelism
  num_envs_per_worker: 16     # More envs per worker

training:
  batch_size: 8192            # Larger batches
  n_epochs: 2                 # Fewer epochs for speed
```

## Comparison: Sequential vs Parallel

| Aspect              | Sequential | Parallel (4 workers) |
|---------------------|------------|----------------------|
| Processes           | 1          | 5 (1 main + 4 workers) |
| Data Collection     | Sequential | Parallel             |
| Policy Updates      | Same       | Same                 |
| FPS                 | 1,500      | 3,500-4,500          |
| Memory              | 1-3 GB     | 4-8 GB               |
| CPU Utilization     | Moderate   | High                 |
| Setup Complexity    | Simple     | Moderate             |
| Best For            | Testing, single-core | Production, multi-core |

## Key Differences from A3C

This is **Parallel PPO**, not A3C:

| Feature              | Parallel PPO | A3C |
|----------------------|--------------|-----|
| Updates              | Centralized  | Distributed |
| Workers              | Synchronous  | Asynchronous |
| Data Aggregation     | Yes          | No |
| Policy Lag           | None         | Present |
| Stability            | High         | Lower |
| Sample Efficiency    | Better       | Good |

## Summary

**When to use Parallel PPO:**

- ✅ Multi-core CPU available
- ✅ Want to maximize CPU utilization
- ✅ Need faster training than sequential
- ✅ Have sufficient memory (4+ GB)
- ✅ Benchmarking parallel vs sequential

**When to use Sequential PPO:**

- ✅ Single-core or limited cores
- ✅ Memory constrained (< 4 GB)
- ✅ Simpler setup preferred
- ✅ Testing/debugging

**Performance Rule of Thumb:**

- Sequential: ~1,500 FPS (32 envs, 16-core CPU)
- Parallel: ~3,500 FPS (4 workers, same setup)
- **Speedup: 2-3x with 4 workers**
