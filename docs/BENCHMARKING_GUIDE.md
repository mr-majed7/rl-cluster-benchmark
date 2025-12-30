# Benchmarking Guide: Sequential vs Parallel Training

This guide explains how to properly benchmark and compare sequential vs parallel training for your RL agents.

## Overview

The benchmarking workflow consists of:

1. **Sequential Training** - Train for a fixed duration (e.g., 1 hour) and collect metrics
2. **Parallel Training** - Train for the same duration with parallel/distributed setup
3. **Comparison** - Analyze and visualize the differences

## Quick Start

### 1. Run Sequential Training (1 Hour)

```bash
python train_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/sequential_1h \
  --num-envs 32 \
  --num-threads 16
```

This will:

- Train for exactly 1 hour
- Save all metrics to `./benchmarks/sequential_1h/`
- Create checkpoints and logs
- Generate a comprehensive metrics JSON file

### 2. Run Parallel Training (Coming Soon)

```bash
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h \
  --num-workers 4 \
  --num-envs 32
```

### 3. Compare Results

```bash
python compare_results.py \
  --sequential ./benchmarks/sequential_1h \
  --parallel ./benchmarks/parallel_1h \
  --output comparison.png
```

## Collected Metrics

The timed training script collects comprehensive metrics for fair comparison:

### Performance Metrics

- **FPS (Frames Per Second)**: Training speed
  - Average FPS over entire run
  - Final FPS (last measurement)
- **Total Timesteps**: Number of environment steps completed
- **Total Updates**: Number of policy updates performed

### Learning Metrics

- **Reward Progression**:
  - Initial reward (first episodes)
  - Final reward (last episodes)
  - Best reward achieved
  - Average of last 100 episodes
- **Training Losses**:
  - Policy loss
  - Value loss
  - Entropy loss

### Resource Metrics

- **Memory Usage**:
  - Peak memory usage (GB)
  - Average memory usage
- **CPU Utilization** (when available)
- **Time metrics**:
  - Total duration
  - Timesteps per hour

### Efficiency Metrics

- **Sample Efficiency**: Timesteps achieved per hour
- **Convergence Rate**: How quickly rewards improve
- **Stability**: Standard deviation of rewards

## Detailed Usage

### Sequential Training Options

```bash
python train_timed.py \
  --duration 1.0 \                    # Training duration in hours
  --output-dir ./benchmarks/seq \     # Output directory
  --env procgen:procgen-coinrun-v0 \  # Environment
  --num-envs 32 \                     # Number of parallel envs
  --num-threads 16 \                  # CPU threads
  --batch-size 1024 \                 # Batch size
  --n-steps 128 \                     # Rollout steps
  --seed 42                           # Random seed
```

### Output Structure

After running, you'll have:

```
benchmarks/sequential_1h/
├── benchmark_metrics.json      # Comprehensive metrics summary
├── config.yaml                 # Configuration used
├── checkpoints/                # Model checkpoints
│   ├── ppo_sequential_step_*.pt
│   └── ppo_sequential_final.pt
└── logs/                       # TensorBoard logs
    └── PPO_Sequential/
        ├── events.out.tfevents.*
        └── metrics.jsonl       # Detailed training logs
```

### Metrics JSON Format

```json
{
  "training_mode": "sequential",
  "start_time": "2025-01-01T10:00:00",
  "end_time": "2025-01-01T11:00:00",
  "duration_hours": 1.0,
  "total_timesteps": 1048576,
  "total_updates": 256,
  "average_fps": 8192,
  "final_fps": 8500,
  "performance": {
    "initial_reward": 3.5,
    "final_reward": 7.2,
    "best_reward": 8.1,
    "average_reward_last_100_episodes": 7.0
  },
  "resource_usage": {
    "peak_memory_gb": 4.2
  },
  "config": { ... }
}
```

## Comparison Analysis

### Generate Comparison Report

```bash
python compare_results.py \
  --sequential ./benchmarks/sequential_1h \
  --parallel ./benchmarks/parallel_1h
```

Output includes:

1. **Console Table**: Side-by-side comparison
2. **Visualization**: 4-panel comparison plot
   - Episode rewards over timesteps
   - Training speed (FPS) over time
   - Policy loss progression
   - Cumulative progress over wall-clock time
3. **Text Report**: `comparison_report.txt`

### Key Comparison Metrics

**Speedup**:

```
Speedup = Parallel FPS / Sequential FPS
```

Expected: 2-4x with 4 workers

**Efficiency Gain**:

```
Efficiency = (Parallel Timesteps/Hour - Sequential Timesteps/Hour) / Sequential Timesteps/Hour × 100%
```

**Sample Efficiency**:

- Compare final rewards at same timestep count
- Higher is better

**Resource Efficiency**:

- Memory usage per worker
- Total memory usage

## Best Practices

### 1. Fair Comparison

**Use Same Configuration**:

```bash
# Sequential
python train_timed.py \
  --duration 1.0 \
  --env procgen:procgen-coinrun-v0 \
  --seed 42 \
  --num-envs 32

# Parallel (total envs should match: 4 workers × 8 envs = 32 total)
python train_parallel_timed.py \
  --duration 1.0 \
  --env procgen:procgen-coinrun-v0 \
  --seed 42 \
  --num-workers 4 \
  --num-envs 8
```

### 2. Multiple Runs

For statistical significance:

```bash
# Run 3-5 times with different seeds
for seed in 42 43 44; do
  python train_timed.py \
    --duration 1.0 \
    --output-dir ./benchmarks/sequential_seed${seed} \
    --seed $seed
done
```

Then average the results.

### 3. Resource Monitoring

Monitor system resources during training:

```bash
# Terminal 1: Training
python train_timed.py --duration 1.0 --output-dir ./benchmarks/seq

# Terminal 2: Monitoring
watch -n 1 'ps aux | grep python | head -n 5'
```

Or use `htop`, `nvtop` (if GPU), or system monitors.

### 4. Consistent Hardware

- Run on same machine/cluster
- Close other applications
- Ensure stable system load
- Same CPU governor settings (performance mode)

## Example Workflow

### Complete 1-Hour Benchmark

```bash
#!/bin/bash

# 1. Sequential training
echo "Starting sequential training..."
python train_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/seq_coinrun_1h \
  --env procgen:procgen-coinrun-v0 \
  --num-envs 32 \
  --num-threads 16 \
  --seed 42

# 2. Wait and cool down (optional)
echo "Waiting 5 minutes for system cooldown..."
sleep 300

# 3. Parallel training (when implemented)
echo "Starting parallel training..."
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/par_coinrun_1h \
  --env procgen:procgen-coinrun-v0 \
  --num-workers 4 \
  --num-envs 8 \
  --seed 42

# 4. Compare results
echo "Generating comparison..."
python compare_results.py \
  --sequential ./benchmarks/seq_coinrun_1h \
  --parallel ./benchmarks/par_coinrun_1h \
  --output ./benchmarks/coinrun_comparison.png

echo "Benchmark complete!"
```

## Interpreting Results

### Good Speedup (2-4x)

```
Sequential: 8,000 FPS
Parallel (4 workers): 24,000 FPS
Speedup: 3.0x ✓
```

### Poor Speedup (<1.5x)

Possible causes:

- Network bottleneck (distributed training)
- Synchronization overhead
- Unbalanced workload
- I/O bottleneck (slow storage)

### Sample Efficiency Comparison

```
Both reach same reward:
Sequential: 500K timesteps
Parallel: 520K timesteps
Efficiency: Similar ✓
```

If parallel needs significantly more timesteps:

- Check synchronization frequency
- Verify identical hyperparameters
- Ensure consistent random seeds

## Advanced Analysis

### Learning Curve Analysis

```python
import json
import matplotlib.pyplot as plt

# Load sequential logs
with open('benchmarks/seq/logs/PPO_Sequential/metrics.jsonl') as f:
    seq_logs = [json.loads(line) for line in f]

# Extract rewards
rewards = [log['rollout/ep_reward_mean'] 
           for log in seq_logs 
           if 'rollout/ep_reward_mean' in log]

# Plot with confidence intervals
plt.plot(rewards)
plt.fill_between(range(len(rewards)), 
                 rewards - std, 
                 rewards + std, 
                 alpha=0.3)
```

### Statistical Significance

Use multiple seeds and t-tests:

```python
from scipy import stats

# Collect final rewards from multiple runs
seq_rewards = [7.2, 7.5, 7.1, 7.3, 7.4]
par_rewards = [7.3, 7.6, 7.2, 7.4, 7.5]

t_stat, p_value = stats.ttest_ind(seq_rewards, par_rewards)
print(f"P-value: {p_value}")
# If p < 0.05, difference is statistically significant
```

## Troubleshooting

### Issue: Training stops before time limit

**Check**:

- Disk space for logs/checkpoints
- Memory availability
- System stability

**Solution**: Monitor resources in parallel

### Issue: Inconsistent FPS

**Check**:

- System load (other processes)
- CPU frequency scaling
- Thermal throttling

**Solution**:

```bash
# Set CPU to performance mode (Linux)
sudo cpupower frequency-set -g performance
```

### Issue: Can't compare results

**Check**:

- Both runs completed successfully
- Metrics files exist
- Same environment used

**Solution**: Verify file paths and retry

## Next Steps

1. **Run sequential benchmark** (1 hour)
2. **Analyze sequential results** to understand baseline
3. **Implement parallel training** (A2C/A3C or IMPALA)
4. **Run parallel benchmark** (same duration)
5. **Compare and optimize** based on results

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [IMPALA Paper](https://arxiv.org/abs/1802.01561)
- [Procgen Benchmark](https://arxiv.org/abs/1912.01588)
