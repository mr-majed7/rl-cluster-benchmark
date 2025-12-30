# Quick Start Guide - CPU-Optimized Training

This guide will help you get started with training PPO agents on Procgen environments using **CPU-only training**, optimized for single machines and cluster computing.

## Installation (2 minutes)

### Option 1: Using the setup script (Recommended)

```bash
./setup.sh
source venv/bin/activate
```

### Option 2: Manual installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

Run the quick test to ensure everything is working:

```bash
python quick_test.py
```

You should see output confirming that PyTorch, Gymnasium, Procgen, and all project modules are working correctly.

## Training Your First Agent

### Quick Test Training (15-20 minutes on 16-core CPU)

For a quick test with minimal compute:

```bash
python train_ppo_sequential.py \
  --total-timesteps 1000000 \
  --num-envs 16 \
  --num-threads 16 \
  --save-interval 10
```

The training is **CPU-optimized** by default and will automatically:

- Detect and use optimal number of CPU threads
- Configure multi-threading for PyTorch, NumPy, and OpenBLAS
- Monitor memory usage
- Report FPS and training metrics

### Full Training (8-10 hours on 16-core CPU)

For a full training run:

```bash
python train_ppo_sequential.py --config config/ppo_sequential.yaml
```

Or simply:

```bash
make train
```

### Adjust for Your CPU

**For fewer cores (4-8 cores):**

```bash
python train_ppo_sequential.py \
  --num-envs 8 \
  --num-threads 4 \
  --batch-size 512
```

**For many cores (32+ cores):**

```bash
python train_ppo_sequential.py \
  --num-envs 64 \
  --num-threads 32 \
  --batch-size 2048
```

## Monitoring Training

### Option 1: Terminal Output

Training progress is displayed in the terminal with a progress bar showing:

- Current update and step count
- FPS (frames per second)
- Episode rewards (mean ± std)

### Option 2: TensorBoard

Launch TensorBoard in a separate terminal:

```bash
tensorboard --logdir logs/
```

Then open <http://localhost:6006> in your browser to see:

- Episode rewards over time
- Loss curves (policy, value, entropy)
- Training metrics (KL divergence, clip fraction, etc.)

### Option 3: Plot Results

After training, generate plots:

```bash
python plot_results.py --log-file logs/PPO_Sequential/metrics.jsonl
```

## Evaluating Your Model

After training completes, evaluate the trained agent:

```bash
python evaluate.py \
  --checkpoint checkpoints/ppo_sequential/ppo_sequential_final.pt \
  --num-episodes 100
```

Or use the Makefile:

```bash
make evaluate
```

## Common Use Cases

### Train on a Different Environment

```bash
python train_ppo_sequential.py --env procgen:procgen-starpilot-v0
```

Available environments:

- `procgen:procgen-coinrun-v0` (default)
- `procgen:procgen-starpilot-v0`
- `procgen:procgen-bigfish-v0`
- `procgen:procgen-bossfight-v0`
- `procgen:procgen-caveflyer-v0`
- And 11 more!

### Optimize for Your Hardware

**Maximize CPU utilization:**

```bash
# Auto-detect optimal settings
python train_ppo_sequential.py

# Manual thread configuration
python train_ppo_sequential.py --num-threads $(nproc)
```

**For limited memory systems:**

```bash
python train_ppo_sequential.py \
  --num-envs 8 \
  --batch-size 512 \
  --n-steps 64
```

**For high-memory systems:**

```bash
python train_ppo_sequential.py \
  --num-envs 128 \
  --batch-size 4096 \
  --n-steps 256
```

### Custom Hyperparameters

```bash
python train_ppo_sequential.py \
  --learning-rate 0.0003 \
  --gamma 0.99 \
  --ent-coef 0.02 \
  --num-envs 64 \
  --total-timesteps 50000000
```

## Typical Training Times (CPU)

**Consumer CPU (16 cores, e.g., AMD Ryzen 9 5950X):**

- Quick test (1M steps, 16 envs): ~15-20 minutes
- Medium run (10M steps, 32 envs): ~3-4 hours
- Full training (25M steps, 32 envs): ~8-10 hours

**Server CPU (64 cores, e.g., AMD EPYC 7763):**

- Quick test (1M steps, 64 envs): ~5-8 minutes
- Medium run (10M steps, 64 envs): ~1-2 hours
- Full training (25M steps, 64 envs): ~3-4 hours

**Laptop CPU (8 cores, e.g., Intel i7):**

- Quick test (1M steps, 8 envs): ~30-40 minutes
- Medium run (10M steps, 16 envs): ~6-8 hours
- Full training (25M steps, 16 envs): ~16-20 hours

> **Tip:** Training scales almost linearly with CPU cores up to the environment limit.

## Project Structure

```
rl-cluster-benchmark/
├── src/                          # Source code
│   ├── models.py                 # Neural network architectures
│   ├── ppo.py                    # PPO algorithm
│   ├── buffer.py                 # Experience buffer
│   ├── trainer.py                # Training pipeline
│   └── utils.py                  # Utilities & logging
├── config/                       # Configuration files
│   └── ppo_sequential.yaml       # PPO config
├── train_ppo_sequential.py       # Training script
├── evaluate.py                   # Evaluation script
├── plot_results.py               # Plotting script
├── quick_test.py                 # Installation test
└── setup.sh                      # Setup script
```

## Troubleshooting

### High Memory Usage

Reduce the number of environments or batch size:

```bash
python train_ppo_sequential.py --num-envs 16 --batch-size 512
```

### Slow Training

Ensure you're using all available CPU cores:

```bash
# Check detected threads
python quick_test.py

# Manually set threads
python train_ppo_sequential.py --num-threads $(nproc)
```

### CPU Usage Not at 100%

This is normal! Training involves:

- Environment simulation (CPU bound)
- Neural network updates (can benefit from vectorization)
- Data collection and preprocessing

Typical CPU usage: 60-90%

### Procgen Installation Issues

On some systems, you may need to install additional dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libosmesa6

# macOS
brew install glfw3
```

### Training Not Improving

Try adjusting hyperparameters:

- Increase learning rate: `--learning-rate 0.001`
- Increase entropy coefficient: `--ent-coef 0.02`
- Train for more steps: `--total-timesteps 50000000`

## Next Steps

1. **Experiment with different environments**: Try all 16 Procgen games
2. **Hyperparameter tuning**: Adjust learning rate, entropy coefficient, etc.
3. **Longer training**: Run for 100M+ steps for best results
4. **Scale to cluster**: Prepare for distributed CPU training (coming soon)
5. **Monitor efficiency**: Use TensorBoard to track FPS and memory usage

## CPU Cluster Preparation

For future distributed training on a CPU cluster:

1. Ensure all nodes have the same Python environment
2. Set up shared filesystem for checkpoints
3. Configure SSH access between nodes
4. Review `config/ppo_cpu_cluster.yaml` for cluster settings

## Getting Help

If you encounter issues:

1. Check the main [README.md](README.md) for detailed documentation
2. Run `python quick_test.py` to verify installation
3. Check TensorBoard for training diagnostics
4. Review the configuration in `config/ppo_sequential.yaml`

Happy training! 🚀
