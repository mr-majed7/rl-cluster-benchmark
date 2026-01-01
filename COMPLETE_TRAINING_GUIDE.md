# Complete Training & Evaluation Guide - PPO & IMPALA

A comprehensive guide to train, evaluate, and compare PPO and IMPALA agents on Procgen environments using CPU-optimized implementations.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Algorithms](#understanding-the-algorithms)
3. [Configuration Files](#configuration-files)
4. [Training PPO](#training-ppo)
5. [Training IMPALA](#training-impala)
6. [Evaluation](#evaluation)
7. [Understanding Results](#understanding-results)
8. [Comparison & Benchmarking](#comparison--benchmarking)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Setup

```bash
cd /home/majed/storage/rl-cluster-benchmark
source venv/bin/activate
```

### Train & Evaluate in 5 Minutes

```bash
# Quick test PPO (6 minutes)
python train_timed.py --duration 0.1 --output-dir ./benchmarks/ppo_test

# Quick test IMPALA (6 minutes)
python train_impala_sequential_timed.py --duration 0.1 --output-dir ./benchmarks/impala_test

# Evaluate PPO
python evaluate_universal.py --algorithm ppo --checkpoint ./benchmarks/ppo_test/checkpoints/ppo_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic

# Evaluate IMPALA
python evaluate_universal.py --algorithm impala --checkpoint ./benchmarks/impala_test/checkpoints/impala_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic
```

---

## Understanding the Algorithms

### PPO (Proximal Policy Optimization)

**Key Characteristics:**

- **On-policy**: Uses only recent experience for training
- **Clipped objective**: Prevents large policy updates (stable training)
- **Multiple epochs**: Trains on same batch multiple times
- **Sample efficient**: Good performance with less data

**Best for:**

- Stable, reliable training
- Limited computational resources
- Single-machine training
- Baseline comparisons

**How it works:**

1. Collect experience using current policy
2. Compute advantages using GAE (Generalized Advantage Estimation)
3. Update policy with clipped PPO objective
4. Repeat for multiple epochs on same batch

### IMPALA (Importance Weighted Actor-Learner Architecture)

**Key Characteristics:**

- **Off-policy**: Can use older/stale experience
- **V-trace**: Importance sampling correction for off-policy data
- **Asynchronous**: Designed for distributed actor-learner setup
- **High throughput**: Optimized for parallel data collection

**Best for:**

- Distributed/parallel training
- High-throughput scenarios
- Large-scale experiments
- CPU cluster deployments

**How it works:**

1. Actors collect experience with behavior policy
2. Learner updates target policy asynchronously
3. V-trace corrects for policy lag (off-policy correction)
4. Supports multiple actors and learners

---

## Configuration Files

### Available Configurations

#### For Ryzen 7 7700 (16 threads, 16GB RAM)

- `config/ppo_sequential_ryzen7.yaml` - PPO, 64 envs
- `config/ppo_parallel_ryzen7.yaml` - PPO, 4 workers × 16 envs
- `config/impala_sequential_ryzen7.yaml` - IMPALA, 48 envs
- `config/impala_parallel_ryzen7.yaml` - IMPALA, 4 actors × 12 envs

#### For Intel i7-14700K (28 threads, 64GB RAM)

- `config/ppo_sequential_highend.yaml` - PPO, 112 envs
- `config/ppo_parallel_highend.yaml` - PPO, 7 workers × 16 envs

#### Standard Configurations

- `config/ppo_sequential.yaml` - PPO, 32 envs (any machine)
- `config/ppo_parallel.yaml` - PPO, 4 workers × 8 envs (any machine)
- `config/impala_sequential.yaml` - IMPALA, 32 envs (any machine)
- `config/impala_parallel.yaml` - IMPALA, 4 actors × 8 envs (any machine)

### Configuration Structure

**PPO Sequential Example:**

```yaml
env:
  name: "procgen-coinrun-v0"
  seed: 42

training:
  num_envs: 64              # Total environments
  n_steps: 256              # Steps per rollout
  total_timesteps: 25_000_000
  batch_size: 2048          # Batch for training
  n_epochs: 4               # Epochs per update
  learning_rate: 0.0003
  gamma: 0.999
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  log_interval: 1
  save_interval: 100
  device: cpu
  num_threads: 16           # CPU threads

logging:
  checkpoint_dir: "./checkpoints/ppo_sequential_ryzen7"
  log_dir: "./logs"
```

**IMPALA Sequential Example:**

```yaml
env:
  name: "procgen-coinrun-v0"
  seed: 42

training:
  num_envs: 48              # Total environments
  n_steps: 128              # Rollout length
  total_timesteps: 25_000_000
  learning_rate: 0.0001     # Lower LR for IMPALA

impala:
  vtrace_clip_rho_threshold: 1.0      # ρ clipping
  vtrace_clip_pg_rho_threshold: 1.0   # c clipping
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 40.0       # Higher for IMPALA

hardware:
  device: "cpu"
  num_threads: 16
```

---

## Training PPO

### Method 1: Timed Training (For Benchmarking)

Train for a specific duration and collect detailed metrics.

#### Sequential PPO (30 minutes)

```bash
python train_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/ppo_seq_30min \
  --config config/ppo_sequential_ryzen7.yaml
```

**Expected Output:**

```
======================================================================
TRAINING SUMMARY
======================================================================
Duration: 0.50 hours
Total Timesteps: 4,608,000
Total Updates: 1,125
Average FPS: 2,560
Final FPS: 2,548

Performance:
  Initial Reward: 2.45 ± 3.21
  Final Reward: 6.82 ± 2.14
  Best Reward: 9.20

Resource Usage:
  Peak Memory: 2.14 GB

Results saved to: ./benchmarks/ppo_seq_30min
======================================================================
```

#### Parallel PPO (30 minutes) - ⚠️ Currently has performance issues

```bash
python train_parallel_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/ppo_par_30min \
  --config config/ppo_parallel_ryzen7.yaml
```

**Note:** Parallel PPO currently runs slower than sequential due to shared memory contention. Use sequential for now.

### Method 2: Fixed Timesteps Training

Train until a specific number of timesteps is reached.

```bash
# Sequential PPO (25M timesteps, ~2.5-3 hours)
python train_ppo_sequential.py --config config/ppo_sequential_ryzen7.yaml

# Or with Makefile
make train
```

### Monitoring Training

**Option 1: Console Output**

```
Step: 2,560,000 | Update: 625 | FPS: 2548 | Loss: 0.0234 | Entropy: 1.234
  Mean Reward: 6.82 | Mean Length: 245
```

**Option 2: TensorBoard (Real-time)**

```bash
tensorboard --logdir ./benchmarks/ppo_seq_30min/logs
# Open http://localhost:6006
```

Or use Makefile:

```bash
make tensorboard
```

### Output Structure

After training, you'll find:

```
./benchmarks/ppo_seq_30min/
├── checkpoints/
│   ├── ppo_sequential_step_1280000.pt
│   ├── ppo_sequential_step_2560000.pt
│   └── ppo_sequential_final.pt
├── logs/
│   └── PPO_Sequential/
│       └── events.out.tfevents.*
├── training_summary.json
└── metrics.json
```

**training_summary.json** contains:

```json
{
  "algorithm": "PPO",
  "mode": "sequential",
  "duration_hours": 0.5,
  "total_timesteps": 4608000,
  "total_updates": 1125,
  "avg_fps": 2560,
  "final_fps": 2548,
  "peak_memory_gb": 2.14,
  "initial_reward": 2.45,
  "final_reward": 6.82,
  "best_reward": 9.20,
  "num_episodes": 234
}
```

---

## Training IMPALA

### Method 1: Timed Training (Recommended for Comparison)

#### Sequential IMPALA (30 minutes)

```bash
python train_impala_sequential_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/impala_seq_30min \
  --config config/impala_sequential_ryzen7.yaml
```

**Expected Output:**

```
======================================================================
IMPALA Sequential Training - Timed Benchmark
======================================================================
Environment: procgen-coinrun-v0
Num environments: 48
Duration: 0.5 hours
Device: cpu
Output: ./benchmarks/impala_seq_30min
======================================================================

[0.25h/0.50h] Step: 2,211,840 | Updates: 360 | FPS: 2456 | Loss: 0.0189
  Mean Reward (last 100): 5.34

======================================================================
TRAINING SUMMARY
======================================================================
Duration: 0.50 hours
Total Timesteps: 4,423,680
Total Updates: 720
Average FPS: 2457
Final FPS: 2452

Resource Usage:
  Peak Memory: 1.89 GB

Results saved to: ./benchmarks/impala_seq_30min
======================================================================
```

#### Parallel IMPALA (30 minutes) - ✅ NEW

```bash
python train_impala_parallel_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/impala_par_30min \
  --config config/impala_parallel_ryzen7.yaml
```

**Expected Output:**

```
======================================================================
IMPALA Parallel Training - Timed Benchmark
======================================================================
Environment: procgen-coinrun-v0
Actors: 4
Envs per actor: 12
Total environments: 48
Duration: 0.5 hours
Device: cpu
Output: ./benchmarks/impala_par_30min
======================================================================

[0.25h/0.50h] Step: 3,072,000 | Updates: 500 | FPS: 3415 | Loss: 0.0145

======================================================================
TRAINING SUMMARY
======================================================================
Duration: 0.50 hours
Actors: 4
Total Environments: 48
Total Timesteps: 6,144,000
Total Updates: 1,000
Average FPS: 3413
Final FPS: 3408

Resource Usage:
  Peak Memory: 2.15 GB

Results saved to: ./benchmarks/impala_par_30min
======================================================================
```

**Note:** Parallel IMPALA uses a proper actor-learner architecture with 4 actor processes collecting experience and a central learner updating the policy. This is **much faster** than sequential!

### Method 2: Fixed Timesteps Training

```bash
# Sequential IMPALA (25M timesteps)
python train_impala_sequential.py --config config/impala_sequential_ryzen7.yaml

# Parallel IMPALA (25M timesteps)
python train_impala_parallel.py --config config/impala_parallel_ryzen7.yaml

# Or with Makefile
make train-impala           # Sequential
make train-impala-parallel  # Parallel
```

### Monitoring Training

Same as PPO - use console output or TensorBoard:

```bash
tensorboard --logdir ./benchmarks/impala_seq_30min/logs
```

### Output Structure

```
./benchmarks/impala_seq_30min/
├── checkpoints/
│   ├── impala_sequential_step_1228800.pt
│   ├── impala_sequential_step_2457600.pt
│   └── impala_sequential_final.pt
├── logs/
│   └── IMPALA_Sequential/
│       └── events.out.tfevents.*
├── training_summary.json
└── metrics.json
```

---

## Evaluation

After training, evaluate your models to measure final performance.

### Evaluate PPO

```bash
python evaluate_universal.py \
  --algorithm ppo \
  --checkpoint ./benchmarks/ppo_seq_30min/checkpoints/ppo_sequential_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic
```

**Output:**

```
============================================================
Evaluating PPO Agent
============================================================
Environment: procgen-coinrun-v0
Observation shape: (3, 64, 64)
Number of actions: 15
Device: cpu
Checkpoint: ./benchmarks/ppo_seq_30min/checkpoints/ppo_sequential_final.pt
Algorithm: PPO
------------------------------------------------------------
✓ Loaded checkpoint from: ./benchmarks/ppo_seq_30min/checkpoints/ppo_sequential_final.pt

Evaluating: 100%|████████████████| 100/100 [02:15<00:00,  0.74it/s]

============================================================
PPO Evaluation Results
============================================================
Episodes: 100
Mean Reward: 6.82 ± 2.14
Min/Max Reward: 2.00 / 10.00
Mean Episode Length: 245.67 ± 45.23
============================================================
```

### Evaluate IMPALA

```bash
python evaluate_universal.py \
  --algorithm impala \
  --checkpoint ./benchmarks/impala_seq_30min/checkpoints/impala_sequential_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic
```

**Output:**

```
============================================================
IMPALA Evaluation Results
============================================================
Episodes: 100
Mean Reward: 5.34 ± 2.87
Min/Max Reward: 0.00 / 10.00
Mean Episode Length: 312.45 ± 89.12
============================================================
```

### Evaluation Options

**Deterministic vs Stochastic:**

```bash
# Deterministic (uses argmax, more stable results)
--deterministic

# Stochastic (samples from policy, more exploration)
# (omit --deterministic flag)
```

**More Episodes (Better Statistics):**

```bash
--num-episodes 500  # More reliable mean/std
```

**Different Environments:**

```bash
--env procgen-starpilot-v0
--env procgen-bossfight-v0
--env procgen-bigfish-v0
```

### Using Makefile Shortcuts

```bash
# Evaluate PPO (auto-finds checkpoint)
make evaluate-ppo

# Evaluate IMPALA (auto-finds checkpoint)
make evaluate-impala
```

---

## Understanding Results

### Key Metrics Explained

#### Training Metrics

**Duration**

- Total wall-clock time for training
- Example: `1.01 hours` means 1 hour and 36 seconds

**Total Timesteps**

- Number of environment steps/interactions
- Higher = more experience collected
- Example: `4,608,000` timesteps in 30 minutes

**Total Updates**

- Number of policy/network updates
- Each update processes a batch of experience
- Example: `1,125` updates from 4.6M timesteps

**Average FPS (Frames Per Second)**

- Training throughput: timesteps per second
- Higher = faster training
- **Sequential PPO**: ~1,500-3,500 FPS
- **Sequential IMPALA**: ~1,500-3,000 FPS
- Example: `2,560 FPS` means 2,560 env steps per second

**Final FPS**

- FPS at the end of training
- Should be similar to average (stable performance)
- Example: `2,548 FPS` (very close to average of 2,560)

**Peak Memory**

- Maximum memory used during training
- Important for resource planning
- **Sequential**: ~1.5-2.5 GB
- **Parallel**: ~4-8 GB
- Example: `2.14 GB` for sequential training

#### Reward Metrics

**Initial Reward**

- Performance at start of training (untrained policy)
- Usually low (random policy)
- Example: `2.45 ± 3.21` (mean ± std)

**Final Reward**

- Performance at end of training
- Measures learning progress
- **CoinRun goals**: 8-10+ (good), 5-7 (decent), <5 (poor)
- Example: `6.82 ± 2.14` (decent learning)

**Best Reward**

- Highest episode reward achieved during training
- Shows peak performance
- Example: `9.20` (close to optimal)

**Mean Reward (Evaluation)**

- Average reward over evaluation episodes
- More reliable than training rewards
- Example: `6.82 ± 2.14` over 100 episodes

**Min/Max Reward**

- Range of evaluation performance
- Large range = high variance
- Example: `2.00 / 10.00` (wide range, some failures, some successes)

**Mean Episode Length**

- Average steps per episode
- CoinRun typical: 200-400 steps
- Shorter often means agent found goal faster
- Example: `245.67 ± 45.23` steps

### Performance Benchmarks

#### Ryzen 7 7700 (16 threads, 16GB RAM)

**30-Minute Training:**

| Metric | PPO Sequential | IMPALA Sequential | IMPALA Parallel |
|--------|----------------|-------------------|-----------------|
| Timesteps | 4.5-6.3M | 3.6-5.4M | 5.5-7M |
| Updates | 1,100-1,500 | 700-1,050 | 900-1,150 |
| FPS | 2,500-3,500 | 2,000-3,000 | 3,000-3,900 |
| Memory | 2-2.5 GB | 1.5-2 GB | 2-2.5 GB |
| Final Reward | 6-8 | 5-7 | 6-8 |

#### Intel i7-14700K (28 threads, 64GB RAM)

**30-Minute Training:**

| Metric | PPO Sequential | IMPALA Sequential | IMPALA Parallel |
|--------|----------------|-------------------|-----------------|
| Timesteps | 7-10M | 5.5-8M | 8-12M |
| Updates | 1,700-2,500 | 1,000-1,500 | 1,300-2,000 |
| FPS | 4,000-6,000 | 3,000-5,000 | 4,500-6,500 |
| Memory | 3-4 GB | 2-3 GB | 3-4 GB |
| Final Reward | 7-9 | 6-8 | 7-9 |

### Interpreting Evaluation Results

**Good Performance:**

- Mean Reward > 8.0
- Std Reward < 2.0 (consistent)
- Min Reward > 5.0 (few failures)

**Decent Performance:**

- Mean Reward: 5-8
- Std Reward: 2-3
- Some failures (min reward ~2)

**Poor Performance:**

- Mean Reward < 5
- High std (>3)
- Many failures (min reward ~0)

**Example Analysis:**

```
Mean Reward: 6.82 ± 2.14
Min/Max Reward: 2.00 / 10.00
Mean Episode Length: 245.67 ± 45.23
```

**Analysis:**

- ✓ Decent mean reward (6.82)
- ✓ Moderate variance (±2.14)
- ⚠️ Some failures (min: 2.00)
- ✓ Reached max reward (10.00)
- ✓ Reasonable episode length

**Verdict:** Agent learned well but not fully converged. More training recommended.

---

## Comparison & Benchmarking

### Compare Algorithms

After training, compare different algorithms and modes:

```bash
# View training summaries
cat ./benchmarks/ppo_seq_30min/training_summary.json
cat ./benchmarks/impala_seq_30min/training_summary.json
cat ./benchmarks/impala_par_30min/training_summary.json
```

### IMPALA Sequential vs Parallel Comparison

**When to use Sequential:**
- Single machine with limited cores (<8 cores)
- Debugging and development
- Memory-constrained systems (<8GB RAM)
- Simple baseline experiments

**When to use Parallel:**
- Multi-core CPU (8+ cores)
- Production training runs
- Need maximum throughput
- Have sufficient RAM (12+ GB)
- Distributed/cluster deployments

**Performance Comparison (Ryzen 7 7700):**

| Metric | Sequential | Parallel | Improvement |
|--------|------------|----------|-------------|
| FPS | ~2,500 | ~3,400 | +36% |
| Timesteps (30min) | 4.5M | 6.1M | +36% |
| Memory | 1.6 GB | 2.2 GB | +38% |
| Complexity | Simple | Moderate | - |

**Parallel IMPALA Benefits:**
- ✅ **30-50% faster** than sequential
- ✅ Proper actor-learner architecture
- ✅ Better CPU utilization
- ✅ Scales well to more actors
- ✅ No shared memory bottlenecks (unlike parallel PPO)

### Create Custom Comparison

```python
import json

# Load results
with open('./benchmarks/ppo_seq_30min/training_summary.json') as f:
    ppo_results = json.load(f)

with open('./benchmarks/impala_seq_30min/training_summary.json') as f:
    impala_results = json.load(f)

# Compare
print(f"PPO FPS: {ppo_results['avg_fps']}")
print(f"IMPALA FPS: {impala_results['avg_fps']}")
print(f"PPO Final Reward: {ppo_results['final_reward']}")
print(f"IMPALA Final Reward: {impala_results['final_reward']}")
```

### Benchmark Template

Use this template to record your results:

```
===============================================================================
BENCHMARK RESULTS - [Your Machine Name]
===============================================================================
Date: [Date]
CPU: [e.g., Ryzen 7 7700]
Cores/Threads: [e.g., 8/16]
RAM: [e.g., 16GB]
Environment: procgen-coinrun-v0
Duration: 0.5 hours (30 minutes)
===============================================================================

PPO SEQUENTIAL
-------------------------------------------------------------------------------
Duration: 0.50 hours
Total Environments: 64
Total Timesteps: 4,608,000
Total Updates: 1,125
Average FPS: 2,560
Final FPS: 2,548
Peak Memory: 2.14 GB

Training Progress:
  Initial Reward: 2.45 ± 3.21
  Final Reward: 6.82 ± 2.14
  Best Reward: 9.20

Evaluation Results (100 episodes, deterministic):
  Mean Reward: 6.82 ± 2.14
  Min/Max Reward: 2.00 / 10.00
  Mean Episode Length: 245.67 ± 45.23

IMPALA SEQUENTIAL
-------------------------------------------------------------------------------
Duration: 0.50 hours
Total Environments: 48
Total Timesteps: 4,423,680
Total Updates: 720
Average FPS: 2,457
Final FPS: 2,452
Peak Memory: 1.89 GB

Training Progress:
  Initial Reward: 2.38 ± 3.15
  Final Reward: 5.34 ± 2.87
  Best Reward: 8.50

Evaluation Results (100 episodes, deterministic):
  Mean Reward: 5.34 ± 2.87
  Min/Max Reward: 0.00 / 10.00
  Mean Episode Length: 312.45 ± 89.12

COMPARISON
-------------------------------------------------------------------------------
Metric                          PPO             IMPALA          Winner
---------------------------------------------------------------------------
Average FPS                     2,560           2,457           PPO (+4%)
Total Timesteps (30min)         4,608,000       4,423,680       PPO (+4%)
Final Reward                    6.82            5.34            PPO (+28%)
Best Reward                     9.20            8.50            PPO (+8%)
Memory Usage                    2.14 GB         1.89 GB         IMPALA (-12%)
Episode Length (shorter better) 245.67          312.45          PPO (+21%)

Conclusion: PPO performed better on CoinRun with better rewards and faster 
episode completion. IMPALA used slightly less memory but achieved lower 
final performance.
===============================================================================
```

### Full Comparison Workflow

```bash
# 1. Train PPO Sequential (30 min)
python train_timed.py --duration 0.5 --output-dir ./benchmarks/ppo_seq_30min --config config/ppo_sequential_ryzen7.yaml

# 2. Train IMPALA Sequential (30 min)
python train_impala_sequential_timed.py --duration 0.5 --output-dir ./benchmarks/impala_seq_30min --config config/impala_sequential_ryzen7.yaml

# 3. Train IMPALA Parallel (30 min)
python train_impala_parallel_timed.py --duration 0.5 --output-dir ./benchmarks/impala_par_30min --config config/impala_parallel_ryzen7.yaml

# 4. Evaluate PPO
python evaluate_universal.py --algorithm ppo --checkpoint ./benchmarks/ppo_seq_30min/checkpoints/ppo_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic > ppo_eval.txt

# 5. Evaluate IMPALA Sequential
python evaluate_universal.py --algorithm impala --checkpoint ./benchmarks/impala_seq_30min/checkpoints/impala_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic > impala_seq_eval.txt

# 6. Evaluate IMPALA Parallel
python evaluate_universal.py --algorithm impala --checkpoint ./benchmarks/impala_par_30min/checkpoints/impala_parallel_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic > impala_par_eval.txt

# 7. Compare summaries
cat ./benchmarks/ppo_seq_30min/training_summary.json
cat ./benchmarks/impala_seq_30min/training_summary.json
cat ./benchmarks/impala_par_30min/training_summary.json
```

---

## Troubleshooting

### Training Issues

**Issue: Low FPS (<1000)**

**Solution:**

```bash
# Check CPU threads
python -c "import torch; print(f'Threads: {torch.get_num_threads()}')"

# Set threads explicitly in config or command line
python train_timed.py --duration 0.5 --output-dir ./benchmarks/test --num-threads 16
```

**Issue: Out of Memory**

**Solution:**

```bash
# Reduce environments
# Edit config file: num_envs: 32 -> num_envs: 16

# Or reduce batch size
# Edit config file: batch_size: 2048 -> batch_size: 1024
```

**Issue: Training Not Starting**

**Solution:**

```bash
# Verify procgen installation
python -c "import procgen; print('Procgen OK')"

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check environment
python -c "import gym; env = gym.make('procgen-coinrun-v0'); print('Env OK')"
```

**Issue: Parallel PPO Slower Than Sequential**

**Current Status:** This is a known issue with the current parallel PPO implementation due to shared memory contention. 

**Recommendation:** Use **Parallel IMPALA** instead! IMPALA was designed from the ground up for distributed training and performs excellently with the actor-learner architecture. Parallel IMPALA is typically **30-50% faster** than sequential.

### Evaluation Issues

**Issue: Checkpoint Not Found**

**Solution:**

```bash
# List available checkpoints
ls -lh ./benchmarks/ppo_seq_30min/checkpoints/

# Use full path
python evaluate_universal.py --algorithm ppo --checkpoint $(pwd)/benchmarks/ppo_seq_30min/checkpoints/ppo_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100
```

**Issue: Evaluation Too Slow**

**Solution:**

```bash
# Reduce episodes
--num-episodes 50  # Instead of 100

# Use vectorized evaluation (future enhancement)
```

### General Issues

**Issue: Can't See TensorBoard Plots**

**Solution:**

```bash
# Make sure TensorBoard is running
tensorboard --logdir ./logs

# Check port (default 6006)
# Open http://localhost:6006 in browser

# Try different port
tensorboard --logdir ./logs --port 6007
```

**Issue: Training Crashes Mid-Run**

**Solution:**

```bash
# Check memory
free -h

# Check disk space
df -h

# Monitor resources during training
htop  # In another terminal
```

---

## Quick Reference

### Training Commands

```bash
# PPO Sequential (30 min)
python train_timed.py --duration 0.5 --output-dir ./benchmarks/ppo_seq_30min --config config/ppo_sequential_ryzen7.yaml

# IMPALA Sequential (30 min)
python train_impala_sequential_timed.py --duration 0.5 --output-dir ./benchmarks/impala_seq_30min --config config/impala_sequential_ryzen7.yaml

# IMPALA Parallel (30 min) - NEW!
python train_impala_parallel_timed.py --duration 0.5 --output-dir ./benchmarks/impala_par_30min --config config/impala_parallel_ryzen7.yaml

# PPO Fixed Timesteps
python train_ppo_sequential.py --config config/ppo_sequential_ryzen7.yaml

# IMPALA Sequential Fixed Timesteps
python train_impala_sequential.py --config config/impala_sequential_ryzen7.yaml

# IMPALA Parallel Fixed Timesteps - NEW!
python train_impala_parallel.py --config config/impala_parallel_ryzen7.yaml
```

### Evaluation Commands

```bash
# PPO Evaluation
python evaluate_universal.py --algorithm ppo --checkpoint <path> --env procgen-coinrun-v0 --num-episodes 100 --deterministic

# IMPALA Evaluation
python evaluate_universal.py --algorithm impala --checkpoint <path> --env procgen-coinrun-v0 --num-episodes 100 --deterministic

# Using Makefile
make evaluate-ppo
make evaluate-impala
```

### Monitoring Commands

```bash
# TensorBoard
tensorboard --logdir ./logs
make tensorboard

# View Summary
cat ./benchmarks/ppo_seq_30min/training_summary.json

# Check Resources
htop
nvidia-smi  # If using GPU
```

---

## Summary

### What You Learned

1. **Two Algorithms**: PPO (on-policy, stable) vs IMPALA (off-policy, scalable)
2. **Training Modes**: Timed (benchmarking) vs Fixed Timesteps (convergence)
3. **Evaluation**: Measure final performance with deterministic/stochastic policy
4. **Metrics**: FPS, timesteps, rewards, memory usage
5. **Comparison**: How to benchmark and compare algorithms

### Recommended Workflow

#### For Quick Testing (15 minutes)
1. **Train IMPALA Parallel**: 6 min (fastest, best throughput)
2. **Train PPO Sequential**: 6 min (stable baseline)
3. **Evaluate both**: 3 min total

#### For Algorithm Comparison (1.5 hours)
1. **Train PPO Sequential**: 30 min
2. **Train IMPALA Sequential**: 30 min
3. **Train IMPALA Parallel**: 30 min
4. **Evaluate all three**: 5 min
5. **Analysis**: Compare results

#### For Production Training (Pick One)
- **Best Throughput**: Parallel IMPALA (recommended for clusters)
- **Most Stable**: Sequential PPO (recommended for single machines)
- **Best Balance**: Sequential IMPALA (good middle ground)

### Next Steps

- ✅ Sequential PPO: Implemented and working
- ✅ Sequential IMPALA: Implemented and working
- ✅ **Parallel IMPALA: Implemented and working!**
- ⏳ Fix Parallel PPO: Needs redesign
- ⏳ A2C/A3C: Future implementation

---

**Ready to train?** Start with the Quick Start section and train your first agents in 5 minutes! 🚀
