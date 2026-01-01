# Quick Start Guide - Ryzen 7 7700 (16 threads, 16GB RAM)

## 🖥️ Your Machine Specs

- **CPU**: AMD Ryzen 7 7700 (8 cores, 16 threads)
- **RAM**: 16 GB
- **Optimal Config**: 48-64 environments, 16 threads

## 📦 Setup

```bash
cd /home/majed/storage/rl-cluster-benchmark
source venv/bin/activate
```

## 🚀 Training Commands

### PPO Training (30 minutes each)

```bash
# Sequential PPO
python train_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/ppo_seq_30min \
  --config config/ppo_sequential_ryzen7.yaml

# Parallel PPO (Warning: Currently slower than sequential!)
python train_parallel_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/ppo_par_30min \
  --config config/ppo_parallel_ryzen7.yaml
```

### IMPALA Training (30 minutes)

```bash
# Sequential IMPALA
python train_impala_sequential_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/impala_seq_30min \
  --config config/impala_sequential_ryzen7.yaml
```

## 📊 Expected Performance

### Sequential Training

- **PPO**: ~2,500-3,500 FPS with 64 environments
- **IMPALA**: ~2,000-3,000 FPS with 48 environments
- **Memory**: ~1.5-2.5 GB

### Training Time Estimates (30 minutes)

- **PPO Sequential**: ~4.5-6.3 million timesteps
- **IMPALA Sequential**: ~3.6-5.4 million timesteps

## 🎯 Quick Test (6 minutes)

Test before running full 30-minute training:

```bash
# Test PPO (6 minutes)
python train_timed.py \
  --duration 0.1 \
  --output-dir ./benchmarks/ppo_test \
  --config config/ppo_sequential_ryzen7.yaml

# Test IMPALA (6 minutes)
python train_impala_sequential_timed.py \
  --duration 0.1 \
  --output-dir ./benchmarks/impala_test \
  --config config/impala_sequential_ryzen7.yaml
```

## 📈 Evaluation

After training, evaluate your models:

```bash
# Evaluate PPO
python evaluate_universal.py \
  --algorithm ppo \
  --checkpoint ./benchmarks/ppo_seq_30min/checkpoints/ppo_sequential_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic

# Evaluate IMPALA
python evaluate_universal.py \
  --algorithm impala \
  --checkpoint ./benchmarks/impala_seq_30min/checkpoints/impala_sequential_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic
```

Or use Makefile shortcuts:

```bash
make evaluate-ppo
make evaluate-impala
```

## 🔄 Full Comparison Workflow

### Step 1: Train Both Algorithms (1 hour total)

```bash
# PPO (30 min)
python train_timed.py --duration 0.5 --output-dir ./benchmarks/ppo_seq_30min --config config/ppo_sequential_ryzen7.yaml

# IMPALA (30 min)
python train_impala_sequential_timed.py --duration 0.5 --output-dir ./benchmarks/impala_seq_30min --config config/impala_sequential_ryzen7.yaml
```

### Step 2: Evaluate Both

```bash
# Evaluate PPO
python evaluate_universal.py --algorithm ppo --checkpoint ./benchmarks/ppo_seq_30min/checkpoints/ppo_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic

# Evaluate IMPALA
python evaluate_universal.py --algorithm impala --checkpoint ./benchmarks/impala_seq_30min/checkpoints/impala_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic
```

### Step 3: Compare Results

Check the training summaries:

```bash
cat ./benchmarks/ppo_seq_30min/training_summary.json
cat ./benchmarks/impala_seq_30min/training_summary.json
```

## 📊 View Training Progress

Launch TensorBoard to see real-time training metrics:

```bash
tensorboard --logdir ./logs
# Open browser to http://localhost:6006
```

Or use:

```bash
make tensorboard
```

## 🗂️ Configuration Files

Your Ryzen 7 configs are optimized for your hardware:

- `config/ppo_sequential_ryzen7.yaml` - 64 envs, 16 threads
- `config/ppo_parallel_ryzen7.yaml` - 4 workers × 16 envs = 64 total
- `config/impala_sequential_ryzen7.yaml` - 48 envs, 16 threads

## 💾 Results Location

After training, find your results here:

```
./benchmarks/
├── ppo_seq_30min/
│   ├── checkpoints/
│   │   └── ppo_sequential_final.pt
│   ├── logs/
│   ├── training_summary.json
│   └── metrics.json
└── impala_seq_30min/
    ├── checkpoints/
    │   └── impala_sequential_final.pt
    ├── logs/
    ├── training_summary.json
    └── metrics.json
```

## 🎓 Understanding the Algorithms

### PPO (Proximal Policy Optimization)

- **On-policy**: Uses recent experience
- **Stable**: Clipped objective prevents large updates
- **Sample efficient**: Multiple epochs on same data
- **Best for**: Stable, reliable training

### IMPALA (Importance Weighted Actor-Learner)

- **Off-policy**: Can use older experience
- **V-trace**: Corrects for off-policy data
- **Scalable**: Designed for distributed training
- **Best for**: High throughput, parallel training

## 🔧 Troubleshooting

### Low FPS?

- Check CPU usage: `htop` or `top`
- Verify thread count: Should see 16 threads active
- Reduce `num_envs` if memory constrained

### Out of Memory?

- Reduce `num_envs` in config (try 32 instead of 64)
- Check memory usage: `free -h`

### Training Not Starting?

```bash
# Verify environment
source venv/bin/activate

# Test procgen installation
python -c "import procgen; print('Procgen OK')"

# Test GPU/CPU
python -c "import torch; print(f'Torch: {torch.__version__}')"
```

## 📝 Notes

1. **Sequential is currently faster** than parallel due to shared memory issues in parallel PPO
2. **IMPALA** is designed for parallel training - sequential version is for comparison
3. **30 minutes** is enough to see meaningful learning progress
4. **Evaluation** uses deterministic policy for consistent results

## 🚀 Next Steps

After running sequential versions:

1. Compare PPO vs IMPALA performance
2. Implement parallel IMPALA (coming next)
3. Test on different Procgen games (bossfight, starpilot, etc.)

## 📞 Quick Commands Reference

```bash
# Training
make train-impala-timed    # IMPALA 30 min
make train-timed            # PPO 30 min

# Evaluation  
make evaluate-ppo           # Evaluate PPO
make evaluate-impala        # Evaluate IMPALA

# Monitoring
make tensorboard            # View training curves

# Cleanup
make clean                  # Remove generated files
```
