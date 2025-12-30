# Training Modes Guide

This document explains the different training modes available and when to use each.

## Training Scripts Overview

### 1. **Full Epoch Training** (Recommended for final training)

Train for a fixed number of timesteps (epochs/iterations):

#### Sequential (Single Process)

```bash
# Using default config
python train_ppo_sequential.py --config config/ppo_sequential.yaml

# With custom parameters
python train_ppo_sequential.py \
  --total-timesteps 25000000 \
  --num-envs 32 \
  --learning-rate 0.0003

# Using Makefile
make train
```

#### Parallel (Multi-Process)

```bash
# Using default config
python train_ppo_parallel.py --config config/ppo_parallel.yaml

# With custom parameters
python train_ppo_parallel.py \
  --total-timesteps 25000000 \
  --num-workers 4 \
  --num-envs-per-worker 8

# Using Makefile
make train-parallel
```

**When to use:**

- Final training runs
- When you want consistent total timesteps across runs
- When comparing different hyperparameters
- Production training

**Output:**

- Model checkpoints in `./checkpoints/`
- TensorBoard logs in `./logs/`
- Regular console updates

---

### 2. **Timed Training** (For benchmarking)

Train for a fixed duration (hours/minutes):

#### Sequential Timed

```bash
# 1 hour training
python train_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/sequential_1h \
  --num-envs 32

# 10 minutes training
python train_timed.py \
  --duration 0.167 \
  --output-dir ./benchmarks/sequential_10min \
  --num-envs 32

# Using Makefile (1 hour default)
make train-timed
```

#### Parallel Timed

```bash
# 1 hour training
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h \
  --num-workers 4 \
  --num-envs-per-worker 8

# 10 minutes training
python train_parallel_timed.py \
  --duration 0.167 \
  --output-dir ./benchmarks/parallel_10min \
  --num-workers 4 \
  --num-envs-per-worker 8

# Using Makefile (1 hour default)
make train-parallel-timed
```

**When to use:**

- Benchmarking training speed
- Comparing sequential vs parallel performance
- Resource usage analysis
- Time-constrained experiments

**Output:**

- Model checkpoints in `<output-dir>/checkpoints/`
- TensorBoard logs in `<output-dir>/logs/`
- **Benchmark metrics** in `<output-dir>/benchmark_metrics.json`
- Console updates with training progress

---

## Configuration Files

### Sequential Config: `config/ppo_sequential.yaml`

```yaml
env:
  name: procgen-coinrun-v0

training:
  num_envs: 32              # Total environments
  n_steps: 128              # Steps per rollout
  total_timesteps: 25000000 # Total training timesteps
  batch_size: 1024          # Batch size for updates
  n_epochs: 4               # Optimization epochs per update
  learning_rate: 0.0003
  device: cpu
```

### Parallel Config: `config/ppo_parallel.yaml`

```yaml
env:
  name: procgen-coinrun-v0

parallel:
  num_workers: 4            # Number of worker processes
  num_envs_per_worker: 8    # Environments per worker (total = 4 × 8 = 32)

training:
  n_steps: 128              # Steps per rollout per worker
  total_timesteps: 25000000 # Total training timesteps
  batch_size: 2048          # Larger batch (all workers combined)
  n_epochs: 4               # Optimization epochs per update
  learning_rate: 0.0003
  device: cpu
```

**Important:** For fair comparison, ensure:

- `num_workers × num_envs_per_worker` (parallel) = `num_envs` (sequential)
- Same `n_steps`, `learning_rate`, `gamma`, etc.

---

## Comparing Results

### After Timed Training

```bash
# Compare sequential vs parallel
python compare_results.py \
  --sequential ./benchmarks/sequential_1h \
  --parallel ./benchmarks/parallel_1h \
  --output comparison.png

# Or using Makefile
make compare
```

This will:

1. Print comparison table with:
   - Duration, timesteps, FPS
   - Speedup (parallel/sequential)
   - Memory usage
   - Reward metrics
2. Generate comparison plots
3. Save a detailed report

### After Full Epoch Training

```bash
# Evaluate the trained model
python evaluate.py \
  --checkpoint ./checkmarks/ppo_sequential/ppo_sequential_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic

# Or using Makefile
make evaluate
```

---

## Performance Expectations

### Sequential Training

- **FPS**: ~1,500-2,000 (32 envs, AMD Ryzen 7 7700)
- **Iteration time**: ~0.3-0.5 seconds
- **Memory**: ~2-3 GB

### Parallel Training (4 workers)

- **Expected FPS**: ~4,000-6,000 (2-3x speedup)
- **Iteration time**: ~0.1-0.2 seconds
- **Memory**: ~4-6 GB (more workers = more memory)

**Note:** Actual performance depends on:

- CPU cores and threads
- Environment complexity
- Batch size and network size
- Number of workers

---

## Quick Start Examples

### 1. Quick Test (Fast iteration)

```bash
# Sequential - 500K timesteps
python train_ppo_sequential.py --total-timesteps 500000 --num-envs 16

# Parallel - 500K timesteps
python train_ppo_parallel.py --total-timesteps 500000 --num-workers 2 --num-envs-per-worker 8
```

### 2. Benchmark Comparison (10 minutes each)

```bash
# Sequential
python train_timed.py --duration 0.167 --output-dir ./benchmarks/seq_10min --num-envs 32

# Parallel
python train_parallel_timed.py --duration 0.167 --output-dir ./benchmarks/par_10min --num-workers 4 --num-envs-per-worker 8

# Compare
python compare_results.py --sequential ./benchmarks/seq_10min --parallel ./benchmarks/par_10min
```

### 3. Full Training (25M timesteps)

```bash
# Sequential (~4-5 hours)
python train_ppo_sequential.py --config config/ppo_sequential.yaml

# Parallel (~2-3 hours expected)
python train_ppo_parallel.py --config config/ppo_parallel.yaml
```

---

## Troubleshooting

### Parallel Training is Slower

If parallel training is slower than sequential:

1. **Check worker count**: Too many workers can cause overhead
   - Rule of thumb: num_workers ≤ CPU_cores / 2
   - For 16 cores: try 4-8 workers

2. **Check environments per worker**: Too few = overhead dominates
   - Minimum: 4-8 envs per worker
   - Optimal: 8-16 envs per worker

3. **Check CPU utilization**: Run `htop` or `top` during training
   - All cores should be ~70-90% utilized
   - Low utilization = bottleneck in main process

4. **Check batch size**: Should scale with total environments
   - Sequential (32 envs): batch_size = 1024
   - Parallel (32 envs): batch_size = 1024-2048

### Out of Memory

Reduce:

- `num_envs` or `num_workers × num_envs_per_worker`
- `batch_size`
- `n_steps`

### Training Too Slow

Increase:

- `num_envs` (sequential) or `num_workers` (parallel)
- `n_steps` (fewer updates needed)
- `batch_size` (more efficient GPU usage)

---

## Summary Table

| Feature | Sequential | Parallel | Timed Sequential | Timed Parallel |
|---------|-----------|----------|------------------|----------------|
| **Script** | `train_ppo_sequential.py` | `train_ppo_parallel.py` | `train_timed.py` | `train_parallel_timed.py` |
| **Makefile** | `make train` | `make train-parallel` | `make train-timed` | `make train-parallel-timed` |
| **Stop Condition** | Total timesteps | Total timesteps | Duration (hours) | Duration (hours) |
| **Benchmark Metrics** | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| **Use Case** | Standard training | Fast training | Benchmarking | Benchmarking |
| **Expected Speed** | ~1,500 FPS | ~4,000 FPS | ~1,500 FPS | ~4,000 FPS |
