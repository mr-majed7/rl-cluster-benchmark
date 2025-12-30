# Quick Start: Benchmarking Sequential Training

This guide will help you run a 1-hour sequential training benchmark and collect metrics for comparison with parallel training later.

## Quick Command

```bash
# Run 1-hour sequential benchmark
python train_timed.py --duration 1.0 --output-dir ./benchmarks/sequential_1h
```

Or using Make:

```bash
make train-timed
```

## What Happens

The script will:

1. ✅ Train for exactly **1 hour**
2. ✅ Save all metrics to `./benchmarks/sequential_1h/`
3. ✅ Log to TensorBoard in real-time
4. ✅ Save checkpoints periodically
5. ✅ Generate comprehensive metrics JSON
6. ✅ Print summary at the end

## Output Structure

```
benchmarks/sequential_1h/
├── benchmark_metrics.json      ← Main metrics file for comparison
├── config.yaml                 ← Configuration used
├── checkpoints/                ← Model checkpoints
│   ├── ppo_sequential_step_*.pt
│   └── ppo_sequential_final.pt
└── logs/                       ← TensorBoard logs
    └── PPO_Sequential/
        ├── events.out.tfevents.*
        └── metrics.jsonl       ← Detailed training logs
```

## Monitoring Training

### Real-time Monitoring

In another terminal:

```bash
# TensorBoard
tensorboard --logdir benchmarks/sequential_1h/logs/

# Then open: http://localhost:6006
```

### Terminal Output

You'll see progress like:

```
Training: 45%|████████████▌             | 115/256 [27:30<32:15, 13.72s/it]
Update 115/256 | Step 471,040 | FPS: 9452 | Time: 0.46h | Reward: 6.73 ± 1.24
```

## After Training

### View Summary

The script automatically prints a summary:

```
============================================================
TRAINING SUMMARY
============================================================
Duration: 1.00 hours
Total Timesteps: 1,048,576
Total Updates: 256
Average FPS: 8,734
Final FPS: 9,125

Performance:
  Initial Reward: 3.52
  Final Reward: 7.18
  Best Reward: 8.01

Resource Usage:
  Peak Memory: 3.87 GB

Results saved to: ./benchmarks/sequential_1h
============================================================
```

### View Detailed Metrics

```bash
# View the metrics JSON
cat benchmarks/sequential_1h/benchmark_metrics.json | python -m json.tool

# Or just check specific values
cat benchmarks/sequential_1h/benchmark_metrics.json | grep "average_fps"
```

### Generate Plots

```bash
# Analyze sequential results
python compare_results.py \
  --sequential benchmarks/sequential_1h \
  --output sequential_analysis.png

# Or use Make
make compare
```

## Customization

### Different Duration

```bash
# 30 minutes
python train_timed.py --duration 0.5 --output-dir ./benchmarks/sequential_30min

# 2 hours
python train_timed.py --duration 2.0 --output-dir ./benchmarks/sequential_2h

# 4 hours (for serious benchmarking)
python train_timed.py --duration 4.0 --output-dir ./benchmarks/sequential_4h
```

### Different Environment

```bash
python train_timed.py \
  --duration 1.0 \
  --env procgen:procgen-starpilot-v0 \
  --output-dir ./benchmarks/seq_starpilot_1h
```

### Adjust Resources

```bash
# More environments (faster training, more memory)
python train_timed.py \
  --duration 1.0 \
  --num-envs 64 \
  --num-threads 32 \
  --output-dir ./benchmarks/sequential_1h_64envs

# Fewer environments (slower training, less memory)
python train_timed.py \
  --duration 1.0 \
  --num-envs 16 \
  --num-threads 8 \
  --output-dir ./benchmarks/sequential_1h_16envs
```

### Multiple Seeds (for statistical significance)

```bash
for seed in 42 43 44 45 46; do
  python train_timed.py \
    --duration 1.0 \
    --seed $seed \
    --output-dir ./benchmarks/sequential_seed${seed}
done
```

## Key Metrics Collected

### Performance Metrics

- **Average FPS**: Training speed (frames per second)
- **Total Timesteps**: Environment steps completed in 1 hour
- **Total Updates**: Policy updates performed

### Learning Metrics

- **Initial/Final/Best Reward**: Track learning progress
- **Reward curves**: Logged in TensorBoard
- **Loss values**: Policy, value, and entropy losses

### Resource Metrics

- **Peak Memory**: Maximum RAM usage
- **CPU Utilization**: Thread usage

## Next Steps

### 1. Analyze Results

```bash
python compare_results.py --sequential benchmarks/sequential_1h
```

### 2. Run Multiple Times

For robust benchmarks, run 3-5 times:

```bash
./run_benchmark_suite.sh
```

### 3. Prepare for Parallel Comparison

When parallel training is implemented:

```bash
# Run parallel training for same duration
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h

# Compare both
python compare_results.py \
  --sequential benchmarks/sequential_1h \
  --parallel benchmarks/parallel_1h
```

## Tips for Good Benchmarks

### 1. System Preparation

```bash
# Close other applications
# Ensure stable power (no power saving mode)
# Check system is not under load

# Set CPU to performance mode (Linux)
sudo cpupower frequency-set -g performance

# Check current CPU frequency
watch -n 1 'cat /proc/cpuinfo | grep MHz'
```

### 2. Reproducibility

Always use:

- Same random seed
- Same hyperparameters
- Same environment
- Same system configuration

### 3. Resource Monitoring

Monitor during training:

```bash
# Terminal 1: Training
python train_timed.py --duration 1.0 --output-dir ./benchmarks/seq

# Terminal 2: Resource monitoring
htop

# Or log resources
while true; do
  date >> resource_usage.log
  ps aux | grep python >> resource_usage.log
  free -h >> resource_usage.log
  sleep 60
done
```

## Troubleshooting

### Training stops early

**Check**: Disk space, memory, system logs

```bash
df -h  # Check disk space
free -h  # Check memory
dmesg | tail -n 50  # Check kernel logs
```

### Low FPS

**Check**: CPU usage, other processes

```bash
top -b -n 1 | head -n 20
```

**Solution**: Close other apps, increase `--num-threads`

### Out of Memory

**Solution**: Reduce `--num-envs`

```bash
python train_timed.py --duration 1.0 --num-envs 16
```

## Complete Example

```bash
# 1. Prepare system
sudo cpupower frequency-set -g performance

# 2. Run benchmark
python train_timed.py \
  --duration 1.0 \
  --env procgen:procgen-coinrun-v0 \
  --num-envs 32 \
  --num-threads 16 \
  --seed 42 \
  --output-dir ./benchmarks/sequential_coinrun_1h

# 3. Monitor in another terminal
tensorboard --logdir benchmarks/sequential_coinrun_1h/logs/

# 4. After completion, analyze
python compare_results.py \
  --sequential benchmarks/sequential_coinrun_1h \
  --output analysis.png

# 5. View results
cat benchmarks/sequential_coinrun_1h/benchmark_metrics.json
```

## Expected Results (Reference)

**16-core CPU, 32 environments, 1 hour:**

- Total Timesteps: ~800K - 1.2M
- Average FPS: 8,000 - 12,000
- Peak Memory: 3-5 GB
- Final Reward: Depends on environment (coinrun: ~6-8)

**32-core CPU, 64 environments, 1 hour:**

- Total Timesteps: ~1.5M - 2.5M
- Average FPS: 15,000 - 25,000
- Peak Memory: 6-10 GB

## Questions?

- See full guide: `docs/BENCHMARKING_GUIDE.md`
- Check CPU optimization: `docs/CPU_OPTIMIZATION.md`
- View README: `README.md`
