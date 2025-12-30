# Parallel PPO Training - Quick Start Guide

Get started with parallel PPO training in 5 minutes!

## 🚀 Quick Commands

### 1. Test Parallel Training (1 minute)

```bash
python train_parallel_timed.py \
  --duration 0.017 \
  --output-dir ./benchmarks/test_parallel \
  --num-workers 2 \
  --num-envs-per-worker 4
```

### 2. Full 1-Hour Parallel Training

```bash
# Using Makefile
make train-parallel

# Or directly
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h \
  --num-workers 4 \
  --num-envs-per-worker 8
```

### 3. Compare Sequential vs Parallel

```bash
# First run sequential (if not done)
make train-timed

# Then run parallel
make train-parallel

# Compare results
make compare
```

## 📊 Complete Benchmark Workflow

```bash
# 1. Sequential training (1 hour)
python train_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/sequential_1h

# 2. Parallel training (1 hour)
python train_parallel_timed.py \
  --duration 1.0 \
  --output-dir ./benchmarks/parallel_1h \
  --num-workers 4

# 3. Compare results
python compare_results.py \
  --sequential ./benchmarks/sequential_1h \
  --parallel ./benchmarks/parallel_1h
```

## ⚙️ Configuration Options

### Number of Workers

Choose based on your CPU:

```bash
# 4-core CPU
--num-workers 2

# 8-core CPU
--num-workers 2-4

# 16-core CPU (like Ryzen 7 7700)
--num-workers 4-8

# 32-core CPU
--num-workers 8-16
```

### Environments Per Worker

```bash
# Memory constrained
--num-envs-per-worker 4

# Balanced (recommended)
--num-envs-per-worker 8

# High throughput
--num-envs-per-worker 16
```

### Complete Example

```bash
python train_parallel_timed.py \
  --config config/ppo_parallel.yaml \
  --duration 2.0 \
  --output-dir ./benchmarks/parallel_2h \
  --num-workers 8 \
  --num-envs-per-worker 8 \
  --batch-size 8192 \
  --n-steps 128
```

## 📈 Expected Results (16-core CPU)

### Sequential (32 envs)

- **FPS**: 1,400-2,000
- **Memory**: 1-3 GB
- **Timesteps/hour**: ~5-7 million

### Parallel (4 workers × 8 envs)

- **FPS**: 3,000-5,000
- **Memory**: 4-8 GB
- **Timesteps/hour**: ~10-18 million
- **Speedup**: 2-3.5x

## 🔍 Monitor Training

### TensorBoard

```bash
# Sequential
tensorboard --logdir ./benchmarks/sequential_1h/logs/

# Parallel
tensorboard --logdir ./benchmarks/parallel_1h/logs/

# Both
tensorboard --logdir ./benchmarks/
```

### System Resources

```bash
# CPU usage
htop

# Memory
watch -n 1 free -h

# Processes
ps aux | grep python
```

## 📊 Analyze Results

### View Metrics

```bash
# Sequential metrics
cat ./benchmarks/sequential_1h/benchmark_metrics.json | jq

# Parallel metrics
cat ./benchmarks/parallel_1h/benchmark_metrics.json | jq

# Compare
python compare_results.py \
  --sequential ./benchmarks/sequential_1h \
  --parallel ./benchmarks/parallel_1h
```

### Output Files

```
benchmarks/
├── sequential_1h/
│   ├── benchmark_metrics.json    # Performance metrics
│   ├── config.yaml               # Configuration used
│   ├── checkpoints/              # Model checkpoints
│   │   └── ppo_sequential_final.pt
│   └── logs/                     # TensorBoard logs
│       └── PPO_Sequential/
└── parallel_1h/
    ├── benchmark_metrics.json
    ├── config.yaml
    ├── checkpoints/
    │   └── ppo_parallel_final.pt
    └── logs/
        └── PPO_Parallel/
```

## 🎯 Evaluate Trained Models

### Parallel Model

```bash
python evaluate.py \
  --checkpoint ./benchmarks/parallel_1h/checkpoints/ppo_parallel_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic
```

### Compare Models

```bash
# Sequential model
python evaluate.py \
  --checkpoint ./benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt \
  --num-episodes 100

# Parallel model
python evaluate.py \
  --checkpoint ./benchmarks/parallel_1h/checkpoints/ppo_parallel_final.pt \
  --num-episodes 100
```

## 🐛 Troubleshooting

### Issue: Low speedup

**Try:**

```bash
# Increase workers
--num-workers 8

# Larger batch size
--batch-size 8192
```

### Issue: High memory usage

**Try:**

```bash
# Fewer workers
--num-workers 2

# Fewer envs per worker
--num-envs-per-worker 4
```

### Issue: Workers not starting

**Check:**

```bash
# Test multiprocessing
python -c "import multiprocessing as mp; mp.set_start_method('spawn', force=True); print('OK')"

# Check available memory
free -h
```

## 📚 Next Steps

1. **Read full guide**: `docs/PARALLEL_TRAINING.md`
2. **Optimize settings**: Tune workers and batch size for your CPU
3. **Try other games**: Change `--env procgen-starpilot-v0`
4. **Longer training**: Increase `--duration 4.0` for better results

## 🎓 Key Takeaways

- **Parallel is 2-3x faster** than sequential on multi-core CPUs
- **Use 4-8 workers** for 16-core CPUs
- **Memory usage is higher** (4-8 GB vs 1-3 GB)
- **Same algorithm (PPO)**, just parallel data collection
- **Perfect for benchmarking** CPU cluster performance

---

**Ready to train?** Start with:

```bash
make train-parallel
```
