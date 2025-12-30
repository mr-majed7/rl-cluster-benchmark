# Quick Reference Guide

## ✅ What Works Now

### 1. Sequential Training (RECOMMENDED - Works Great!)

**For full epochs/iterations:**

```bash
# Train for 25M timesteps (default)
python train_ppo_sequential.py --config config/ppo_sequential.yaml

# Or with custom timesteps
python train_ppo_sequential.py --total-timesteps 10000000 --num-envs 32

# Quick shortcut
make train
```

**For timed benchmarking:**

```bash
# Train for 1 hour
python train_timed.py --duration 1.0 --output-dir ./benchmarks/sequential_1h --num-envs 32

# Train for 10 minutes
python train_timed.py --duration 0.167 --output-dir ./benchmarks/seq_10min --num-envs 32

# Quick shortcut (1 hour)
make train-timed
```

**Performance:** ~1,500-2,000 FPS with 32 environments

---

### 2. Evaluation

```bash
# Evaluate trained model
python evaluate.py \
  --checkpoint ./benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt \
  --env procgen-coinrun-v0 \
  --num-episodes 100 \
  --deterministic

# Quick shortcut
make evaluate
```

---

### 3. View Training Metrics

**From timed training:**

```bash
# View the benchmark metrics JSON
cat ./benchmarks/sequential_1h/benchmark_metrics.json | python -m json.tool

# Or use compare_results.py
python compare_results.py --sequential ./benchmarks/sequential_1h
```

**This gives you:**

- Duration (hours)
- Total timesteps
- Total updates  
- Average FPS
- Final FPS
- Initial/Final/Best rewards
- Peak memory usage

**From regular training:**

- TensorBoard logs: `tensorboard --logdir logs/`
- Checkpoints: `./checkpoints/ppo_sequential/`

---

## ⚠️ Parallel Training (Currently Has Performance Issues)

The parallel implementation exists but is currently **slower than sequential** due to:

- Shared memory contention between workers
- Expensive parameter synchronization
- Queue communication overhead

**Status:** Needs optimization before use. Sequential training is faster for now.

---

## Common Commands

### Quick Test (5 minutes)

```bash
python train_ppo_sequential.py --total-timesteps 500000 --num-envs 16
```

### Benchmark Run (1 hour)

```bash
python train_timed.py --duration 1.0 --output-dir ./benchmarks/sequential_1h --num-envs 32
```

### Full Training (25M timesteps, ~4-5 hours)

```bash
python train_ppo_sequential.py --config config/ppo_sequential.yaml
```

### Monitor Training

```bash
# In another terminal
tensorboard --logdir logs/
# Then open http://localhost:6006
```

---

## Configuration Files

### `config/ppo_sequential.yaml`

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
  gamma: 0.999
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  device: cpu
  num_threads: 16           # CPU threads
```

You can modify these values and save as a custom config file.

---

## Understanding the Output

### During Training

```
Update: 100/6104 | FPS: 1,520 | Steps: 409,600
  Policy Loss: 0.0234 | Value Loss: 0.1234
  Entropy: 2.345 | Explained Var: 0.89
  Episode Reward: 5.67 ± 2.34 (50 episodes)
```

- **FPS**: Frames (timesteps) per second - higher is better
- **Policy Loss**: How much the policy is changing
- **Value Loss**: How accurate the value predictions are
- **Entropy**: Exploration level (higher = more exploration)
- **Episode Reward**: Average reward per episode

### After Timed Training

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
```

This summary is saved in `benchmark_metrics.json` for later analysis.

---

## Troubleshooting

### Training is slow

- Check CPU usage with `htop` - should be 70-90%
- Increase `num_envs` (e.g., 32 → 64)
- Increase `n_steps` (e.g., 128 → 256)

### Out of memory

- Reduce `num_envs` (e.g., 32 → 16)
- Reduce `batch_size` (e.g., 1024 → 512)
- Reduce `n_steps` (e.g., 128 → 64)

### Reward not improving

- Train longer (increase `total_timesteps`)
- Adjust learning rate (try 1e-4 to 5e-4)
- Check TensorBoard for learning curves

### Can't find checkpoint

- Timed training saves to: `<output-dir>/checkpoints/`
- Regular training saves to: `./checkpoints/ppo_sequential/`

---

## File Locations

```
rl-cluster-benchmark/
├── benchmarks/              # Timed training results
│   └── sequential_1h/
│       ├── benchmark_metrics.json  # ← Summary metrics
│       ├── checkpoints/
│       │   └── ppo_sequential_final.pt
│       └── logs/
│           └── PPO_Sequential/
│               └── metrics.jsonl   # ← Per-update logs
│
├── checkpoints/             # Regular training checkpoints
│   └── ppo_sequential/
│       └── ppo_sequential_final.pt
│
└── logs/                    # TensorBoard logs
    └── PPO_Sequential/
```

---

## Next Steps

1. **Run a quick test** (5 min):

   ```bash
   python train_ppo_sequential.py --total-timesteps 500000 --num-envs 16
   ```

2. **Run a 1-hour benchmark**:

   ```bash
   make train-timed
   ```

3. **View results**:

   ```bash
   python compare_results.py --sequential ./benchmarks/sequential_1h
   ```

4. **Evaluate the model**:

   ```bash
   make evaluate
   ```

5. **Run full training** (4-5 hours):

   ```bash
   make train
   ```

---

## Summary

✅ **Use sequential training** - it works great and is well-optimized
✅ **Use timed training** for benchmarking with metrics
✅ **Use `compare_results.py`** to view training summaries
❌ **Avoid parallel training** for now - it needs optimization

The sequential implementation is solid and ready for your experiments!
