# Complete Training, Evaluation & Comparison Guide

A step-by-step guide to train PPO agents sequentially and in parallel, then evaluate and compare their performance.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Sequential Training](#sequential-training)
3. [Parallel Training](#parallel-training)
4. [Evaluation](#evaluation)
5. [Comparison](#comparison)
6. [Understanding the Results](#understanding-the-results)

---

## Prerequisites

### Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Verify installation
python quick_test.py
```

### Understanding the Configurations

**Sequential Config** (`config/ppo_sequential.yaml`):

- **32 environments** running in a single process
- Optimized for single-threaded data collection
- Lower memory usage (~1-3 GB)
- Simpler, more stable

**Parallel Config** (`config/ppo_parallel.yaml`):

- **4 workers × 8 envs = 32 total environments**
- Parallel data collection across multiple processes
- Higher memory usage (~4-8 GB)
- 2-3x faster throughput

---

## Sequential Training

### Step 1: Review Sequential Configuration

View the configuration:

```bash
cat config/ppo_sequential.yaml
```

Key settings:

```yaml
env:
  name: "procgen-coinrun-v0"
  num_envs: 32              # Total environments
  seed: 42

training:
  n_steps: 128              # Steps per rollout
  batch_size: 1024          # Batch size
  n_epochs: 4               # Training epochs

hardware:
  device: "cpu"
  num_threads: null         # Auto-detect
```

### Step 2: Run 1-Hour Sequential Training

```bash
python train_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/sequential_1h
```

Or using Makefile:

```bash
make train-timed
```

**What this does:**

- Loads settings from `config/ppo_sequential.yaml` automatically
- Trains for exactly 1 hour
- Saves results to `./benchmarks/sequential_1h/`

### Step 3: Monitor Training

**Option A: Watch terminal output**

```
Training:  45%|████████▌         | 1234/2750 [00:15:23<00:18:45,  1.35it/s]
Update 1234 | Step 5,046,272 | FPS: 1523 | Reward: 6.42 ± 1.23
```

**Option B: TensorBoard (real-time)**

```bash
# In another terminal
tensorboard --logdir ./benchmarks/sequential_1h/logs/
```

Open: <http://localhost:6006>

### Step 4: Wait for Completion

After 1 hour, you'll see:

```
======================================================================
TRAINING SUMMARY
======================================================================
Duration: 1.00 hours
Total Timesteps: 5,471,232
Total Updates: 1,342
Average FPS: 1,520
Final FPS: 1,518

Performance:
  Initial Reward: 2.45
  Final Reward: 7.23
  Best Reward: 8.91

Resource Usage:
  Peak Memory: 2.34 GB

Results saved to: ./benchmarks/sequential_1h
======================================================================
```

### Output Structure

```
benchmarks/sequential_1h/
├── benchmark_metrics.json          # Summary metrics
├── config.yaml                     # Configuration used
├── checkpoints/
│   ├── ppo_sequential_step_*.pt   # Periodic checkpoints
│   └── ppo_sequential_final.pt    # Final model
└── logs/
    └── PPO_Sequential/
        ├── events.out.tfevents.*  # TensorBoard logs
        └── metrics.jsonl          # Detailed training logs
```

---

## Parallel Training

### Step 1: Review Parallel Configuration

View the configuration:

```bash
cat config/ppo_parallel.yaml
```

Key settings:

```yaml
parallel:
  num_workers: 4            # Number of parallel workers
  num_envs_per_worker: 8    # Envs per worker (4×8 = 32 total)

training:
  n_steps: 128
  batch_size: 2048          # Larger batch for parallel
  n_epochs: 4

hardware:
  device: "cpu"
  num_threads: null         # Auto-detect per worker
```

### Step 2: Run 1-Hour Parallel Training

```bash
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h
```

Or using Makefile:

```bash
make train-parallel
```

**What this does:**

- Loads settings from `config/ppo_parallel.yaml` automatically
- Spawns 4 worker processes for parallel data collection
- Trains for exactly 1 hour
- Saves results to `./benchmarks/parallel_1h/`

### Step 3: Monitor Training

**Terminal output:**

```
Worker 0 started with 8 environments
Worker 1 started with 8 environments
Worker 2 started with 8 environments
Worker 3 started with 8 environments

Training:  45%|████████▌         | 892/1980 [00:15:23<00:18:45,  0.97it/s]
Update 892 | Step 11,499,520 | FPS: 3,195 | Reward: 7.12 ± 1.05
```

**TensorBoard:**

```bash
tensorboard --logdir ./benchmarks/parallel_1h/logs/
```

### Step 4: Wait for Completion

After 1 hour:

```
======================================================================
TRAINING SUMMARY
======================================================================
Duration: 1.00 hours
Workers: 4
Total Environments: 32
Total Timesteps: 11,536,384
Total Updates: 2,832
Average FPS: 3,204
Final FPS: 3,198

Performance:
  Initial Reward: 2.38
  Final Reward: 8.45
  Best Reward: 9.67

Resource Usage:
  Peak Memory: 6.12 GB
======================================================================
```

### Output Structure

```
benchmarks/parallel_1h/
├── benchmark_metrics.json
├── config.yaml
├── checkpoints/
│   ├── ppo_parallel_step_*.pt
│   └── ppo_parallel_final.pt
└── logs/
    └── PPO_Parallel/
        ├── events.out.tfevents.*
        └── metrics.jsonl
```

---

## Evaluation

Evaluate the trained models to see how well they perform.

### Evaluate Sequential Model

```bash
python evaluate.py \
  --checkpoint ./benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic
```

**Expected output:**

```
Environment: procgen-coinrun-v0
Observation shape: (3, 64, 64)
Number of actions: 15
Device: cpu
Checkpoint: ./benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt

Loaded checkpoint from: ...

Evaluating: 100%|████████████████| 100/100 [02:15<00:00,  0.74it/s]

============================================================
Evaluation Results
============================================================
Episodes: 100
Mean Reward: 7.23 ± 1.34
Min/Max Reward: 4.50 / 9.80
Mean Episode Length: 245.67 ± 23.14
============================================================
```

### Evaluate Parallel Model

```bash
python evaluate.py \
  --checkpoint ./benchmarks/parallel_1h/checkpoints/ppo_parallel_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic
```

**Expected output:**

```
============================================================
Evaluation Results
============================================================
Episodes: 100
Mean Reward: 8.45 ± 1.12
Min/Max Reward: 5.80 / 10.50
Mean Episode Length: 256.34 ± 19.87
============================================================
```

### Evaluation Options

**Deterministic vs Stochastic:**

```bash
# Deterministic (use argmax, more stable)
--deterministic

# Stochastic (sample from distribution)
# (omit --deterministic flag)
```

**More Episodes (better statistics):**

```bash
--num-episodes 500
```

**Different Environment:**

```bash
--env procgen-starpilot-v0
```

---

## Comparison

### Automatic Comparison

Compare both training runs:

```bash
python compare_results.py \
  --sequential ./benchmarks/sequential_1h \
  --parallel ./benchmarks/parallel_1h \
  --output ./benchmarks/comparison.png
```

Or using Makefile:

```bash
make compare
```

### Comparison Output

**1. Console Summary Table:**

```
================================================================================
TRAINING COMPARISON
================================================================================
Metric                                   Sequential           Parallel            
--------------------------------------------------------------------------------
Duration (hours)                         1.00                 1.00                
Total Timesteps                          5,471,232            11,536,384          
Total Updates                            1,342                2,832               

Performance Metrics
--------------------------------------------------------------------------------
Average FPS                              1,520                3,204               
Speedup (Parallel/Sequential)            -                    2.11x               
Final FPS                                1,518                3,198               

Reward Metrics
--------------------------------------------------------------------------------
Initial Reward                           2.45                 2.38                
Final Reward                             7.23                 8.45                
Best Reward                              8.91                 9.67                

Resource Usage
--------------------------------------------------------------------------------
Peak Memory (GB)                         2.34                 6.12                

Efficiency Metrics
--------------------------------------------------------------------------------
Timesteps/Hour                           5,471,232            11,536,384          
Efficiency Gain (%)                      -                    110.9%              
================================================================================
```

**2. Visual Comparison Plot:**

Saved to: `./benchmarks/comparison.png`

Contains 4 panels:

- **Episode Rewards** over time
- **Training Speed (FPS)** over time
- **Policy Loss** over time
- **Training Progress** (timesteps vs wall-clock time)

**3. Text Report:**

Saved to: `comparison_report.txt`

```
================================================================================
SEQUENTIAL VS PARALLEL TRAINING COMPARISON REPORT
================================================================================

Sequential Directory: ./benchmarks/sequential_1h
Parallel Directory: ./benchmarks/parallel_1h

SEQUENTIAL TRAINING
--------------------------------------------------------------------------------
Duration: 1.00 hours
Total Timesteps: 5,471,232
Average FPS: 1,520
Peak Memory: 2.34 GB

PARALLEL TRAINING
--------------------------------------------------------------------------------
Duration: 1.00 hours
Total Timesteps: 11,536,384
Average FPS: 3,204
Peak Memory: 6.12 GB

COMPARISON
--------------------------------------------------------------------------------
Speedup (FPS): 2.11x
Efficiency Gain: 110.9%
```

---

## Understanding the Results

### Key Metrics Explained

#### Frames Per Second (FPS)

- **What it measures**: Training speed (environment steps/second)
- **Sequential**: 1,400-2,000 FPS
- **Parallel**: 3,000-5,000 FPS
- **Why it matters**: Higher FPS = faster training

#### Total Timesteps

- **What it measures**: Total environment interactions
- **Why it matters**: More timesteps in same time = better sample efficiency
- **Parallel advantage**: 2-3x more timesteps per hour

#### Episode Reward

- **What it measures**: Agent performance
- **Initial**: Untrained baseline (~2-3 for CoinRun)
- **Final**: Performance after training (goal: 8-10+)
- **Best**: Peak performance achieved

#### Memory Usage

- **Sequential**: 1-3 GB (single process)
- **Parallel**: 4-8 GB (multiple processes)
- **Trade-off**: More memory for faster training

### Performance Expectations

**On 16-core AMD Ryzen 7 7700:**

| Metric | Sequential | Parallel (4 workers) | Improvement |
|--------|------------|----------------------|-------------|
| FPS | 1,500 | 3,200 | 2.1x faster |
| Timesteps/hour | 5.4M | 11.5M | 2.1x more |
| Memory | 2-3 GB | 5-7 GB | 2-3x higher |
| Final Reward | 7-8 | 8-9 | Slightly better |
| Training Time (to 25M steps) | 4.6 hours | 2.2 hours | 2.1x faster |

### When to Use Each Mode

**Use Sequential When:**

- ✅ Testing and debugging
- ✅ Limited memory (< 4 GB)
- ✅ Single or few CPU cores
- ✅ Prefer simplicity

**Use Parallel When:**

- ✅ Production training
- ✅ Multi-core CPU (8+ cores)
- ✅ Need faster results
- ✅ Have sufficient memory (4+ GB)
- ✅ Benchmarking performance

---

## Complete Workflow Example

### Full 1-Hour Benchmark

```bash
# Step 1: Sequential training (1 hour)
python train_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/sequential_1h

# Step 2: Parallel training (1 hour)  
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h

# Step 3: Evaluate sequential model
python evaluate.py \
  --checkpoint ./benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt \
  --num-episodes 100 \
  --deterministic

# Step 4: Evaluate parallel model
python evaluate.py \
  --checkpoint ./benchmarks/parallel_1h/checkpoints/ppo_parallel_final.pt \
  --num-episodes 100 \
  --deterministic

# Step 5: Compare results
python compare_results.py \
  --sequential ./benchmarks/sequential_1h \
  --parallel ./benchmarks/parallel_1h
```

### Using Makefile (Simplified)

```bash
# Train both modes
make train-timed      # Sequential
make train-parallel   # Parallel

# Compare
make compare

# Evaluate (finds latest checkpoint automatically)
make evaluate
```

---

## Advanced Usage

### Custom Duration

```bash
# 2-hour training for better convergence
python train_timed.py --duration 2.0 --output-dir ./benchmarks/sequential_2h
python train_parallel_timed.py --duration 2.0 --output-dir ./benchmarks/parallel_2h

# 30-minute quick test
python train_timed.py --duration 0.5 --output-dir ./benchmarks/sequential_30min
```

### Override Configuration

```bash
# Use sequential config but with 64 environments
python train_timed.py \
  --config config/ppo_sequential.yaml \
  --duration 1.0 \
  --output-dir ./benchmarks/sequential_64envs \
  --num-envs 64

# Use parallel config but with 8 workers
python train_parallel_timed.py \
  --config config/ppo_parallel.yaml \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_8workers \
  --num-workers 8 \
  --num-envs-per-worker 8
```

### Different Environments

```bash
# Train on StarPilot
python train_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/starpilot_sequential \
  --env procgen-starpilot-v0

# Train on Bossfight
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/bossfight_parallel \
  --env procgen-bossfight-v0
```

### Multiple Seeds for Robust Results

```bash
# Run with different seeds
for seed in 42 43 44; do
  python train_timed.py \
    --duration 1.0 \
    --output-dir ./benchmarks/sequential_seed_${seed} \
    --seed ${seed}
done
```

---

## Troubleshooting

### Sequential Training

**Issue: Low FPS (< 1000)**

```bash
# Check CPU threads
python -c "import torch; print(f'Threads: {torch.get_num_threads()}')"

# Set threads explicitly
python train_timed.py --duration 1.0 --output-dir ./benchmarks/seq --num-threads 16
```

**Issue: Out of memory**

```bash
# Reduce batch size
python train_timed.py --duration 1.0 --output-dir ./benchmarks/seq --batch-size 512

# Reduce environments
python train_timed.py --duration 1.0 --output-dir ./benchmarks/seq --num-envs 16
```

### Parallel Training

**Issue: Workers not starting**

```bash
# Test multiprocessing
python -c "import multiprocessing as mp; mp.set_start_method('spawn', force=True); print('OK')"

# Check available cores
python -c "import os; print(f'CPU cores: {os.cpu_count()}')"
```

**Issue: Lower than expected speedup**

```bash
# Reduce workers (less overhead)
python train_parallel_timed.py --duration 1.0 --output-dir ./benchmarks/par --num-workers 2

# Increase batch size
python train_parallel_timed.py --duration 1.0 --output-dir ./benchmarks/par --batch-size 4096
```

---

## Summary

### Key Commands

```bash
# Sequential training (uses config/ppo_sequential.yaml)
python train_timed.py --duration 1.0 --output-dir ./benchmarks/sequential_1h

# Parallel training (uses config/ppo_parallel.yaml)
python train_parallel_timed.py --duration 1.0 --output-dir ./benchmarks/parallel_1h

# Evaluation
python evaluate.py --checkpoint <path> --num-episodes 100 --deterministic

# Comparison
python compare_results.py --sequential <dir> --parallel <dir>
```

### Expected Timeline

- **Sequential 1h**: ~5-7M timesteps, reward 7-8
- **Parallel 1h**: ~10-15M timesteps, reward 8-9
- **Evaluation**: ~2-3 minutes for 100 episodes
- **Comparison**: Instant (< 1 second)

### Files to Check

- `config/ppo_sequential.yaml` - Sequential configuration
- `config/ppo_parallel.yaml` - Parallel configuration
- `benchmarks/*/benchmark_metrics.json` - Summary metrics
- `benchmarks/*/logs/*/metrics.jsonl` - Detailed logs
- `comparison.png` - Visual comparison
- `comparison_report.txt` - Text summary

---

**Ready to start?**

```bash
# Complete workflow
make train-timed      # 1 hour
make train-parallel   # 1 hour  
make compare          # Compare results
```

Good luck with your CPU cluster benchmark! 🚀
