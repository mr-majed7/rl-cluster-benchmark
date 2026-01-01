# RL Cluster Benchmark

Benchmarking reinforcement learning algorithms (PPO, IMPALA) on Procgen environments with **CPU-optimized** sequential and distributed training for cluster computing.

> **📖 NEW: [Complete Training & Evaluation Guide](COMPLETE_TRAINING_GUIDE.md)** - Comprehensive guide for training and evaluating PPO & IMPALA with detailed metrics explanations.

## Overview

This project implements and benchmarks different RL algorithms on the Procgen benchmark environments, **specifically optimized for CPU clusters**. The focus is on efficient CPU utilization and distributed training across multiple CPU nodes, making it ideal for large-scale cluster deployments without GPUs.

### Design Philosophy

- **CPU-First**: All implementations are optimized for CPU training with multi-threading support
- **Cluster-Ready**: Designed for distributed training across multiple CPU nodes
- **Memory-Efficient**: Optimized memory usage for long-running cluster jobs
- **Scalable**: Easy to scale from single machine to large CPU clusters

### Implemented Algorithms

- ✅ **PPO (Proximal Policy Optimization)** - CPU-optimized sequential implementation
- 🚧 **A2C/A3C** - Coming soon (CPU-optimized)
- 🚧 **IMPALA** - Coming soon (distributed CPU implementation)

## Installation

### Requirements

- Python 3.8+
- Multi-core CPU (recommended: 16+ cores)
- 16GB+ RAM recommended
- Linux/macOS (Windows may require additional setup for Procgen)

### Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd rl-cluster-benchmark
```

1. Run the automated setup:

```bash
./setup.sh
source venv/bin/activate
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. Verify installation:

```bash
python quick_test.py
```

## Usage

### Training PPO (Sequential, CPU-Optimized)

Train a PPO agent on a Procgen environment using CPU:

```bash
python train_ppo_sequential.py --config config/ppo_sequential.yaml
```

Or use the Makefile:

```bash
make train
```

#### CPU-Specific Options

```bash
# Specify number of CPU threads
python train_ppo_sequential.py --num-threads 16

# Adjust for your CPU capacity
python train_ppo_sequential.py \
  --num-envs 32 \
  --num-threads 16 \
  --batch-size 1024

# Quick test on limited resources
python train_ppo_sequential.py \
  --total-timesteps 1000000 \
  --num-envs 8 \
  --num-threads 4
```

#### Configuration Options

```bash
# Use a different environment
python train_ppo_sequential.py --env procgen:procgen-starpilot-v0

# Adjust training parameters
python train_ppo_sequential.py \
  --num-envs 64 \
  --total-timesteps 50000000 \
  --learning-rate 0.0003

# Custom checkpoint and log directories
python train_ppo_sequential.py \
  --checkpoint-dir ./my_checkpoints \
  --log-dir ./my_logs
```

### Available Procgen Environments

All 16 Procgen environments are supported:

- `procgen:procgen-coinrun-v0` (default)
- `procgen:procgen-starpilot-v0`
- `procgen:procgen-bigfish-v0`
- `procgen:procgen-bossfight-v0`
- `procgen:procgen-caveflyer-v0`
- `procgen:procgen-chaser-v0`
- `procgen:procgen-climber-v0`
- `procgen:procgen-dodgeball-v0`
- `procgen:procgen-fruitbot-v0`
- `procgen:procgen-heist-v0`
- `procgen:procgen-jumper-v0`
- `procgen:procgen-leaper-v0`
- `procgen:procgen-maze-v0`
- `procgen:procgen-miner-v0`
- `procgen:procgen-ninja-v0`
- `procgen:procgen-plunder-v0`

### Evaluating a Trained Model

```bash
python evaluate.py \
  --checkpoint checkpoints/ppo_sequential/ppo_sequential_final.pt \
  --env procgen:procgen-coinrun-v0 \
  --num-episodes 100
```

Options:

- `--deterministic`: Use deterministic policy (argmax instead of sampling)
- `--render`: Render the environment during evaluation
- `--seed`: Set random seed for reproducibility

### Monitoring Training

View training progress in TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open <http://localhost:6006> to view:

- Episode rewards over time
- Loss curves (policy, value, entropy)
- CPU memory usage
- FPS and training speed

Generate matplotlib plots:

```bash
python plot_results.py --log-file logs/PPO_Sequential/metrics.jsonl
```

## Project Structure

```
rl-cluster-benchmark/
├── src/
│   ├── __init__.py
│   ├── models.py          # Neural network architectures
│   ├── ppo.py             # PPO algorithm (CPU-optimized)
│   ├── buffer.py          # Experience replay buffer
│   ├── trainer.py         # Sequential training pipeline
│   ├── utils.py           # Logging and utilities
│   ├── cpu_utils.py       # CPU optimization utilities
│   └── config.py          # Configuration management
├── config/
│   ├── ppo_sequential.yaml    # Single-node CPU config
│   └── ppo_cpu_cluster.yaml   # Multi-node cluster config
├── train_ppo_sequential.py    # Training entry point
├── evaluate.py                # Evaluation script
├── plot_results.py            # Plotting utility
├── quick_test.py              # Installation test
├── setup.sh                   # Automated setup
├── Makefile                   # Convenient commands
├── requirements.txt
├── README.md
└── QUICKSTART.md
```

## CPU Optimization Features

### Automatic Thread Management

- Auto-detects optimal CPU thread count
- Configures PyTorch, NumPy, and OpenBLAS threading
- Manual override with `--num-threads`

### Memory Monitoring

- Real-time memory usage tracking
- Peak memory reporting
- Automatic logging to TensorBoard

### Efficient Batching

- CPU-optimized batch sizes
- Memory-efficient rollout buffer
- Optimized for cache locality

### Multi-threading Support

- Vectorized environment execution
- Parallel policy updates
- Thread-safe data collection

## Configuration

Training parameters can be configured via YAML files or command-line arguments.

### CPU-Optimized Hyperparameters

Default values in `config/ppo_sequential.yaml`:

```yaml
env:
  num_envs: 32  # Reduced for CPU efficiency

training:
  n_steps: 128  # Optimized for CPU memory
  batch_size: 1024  # CPU-friendly batch size
  n_epochs: 4

hardware:
  device: "cpu"
  num_threads: null  # Auto-detect
```

### Cluster Configuration

For distributed CPU cluster training (coming soon), see `config/ppo_cpu_cluster.yaml`:

```yaml
cluster:
  num_workers: 4
  distributed_backend: "gloo"  # CPU-optimized backend
  master_addr: "localhost"
  master_port: 29500
```

## Performance Tips

### For Single Machine

1. **Thread Count**: Set to number of physical cores (not hyperthreads)

   ```bash
   python train_ppo_sequential.py --num-threads $(nproc)
   ```

2. **Environment Count**: Balance between throughput and memory
   - 16GB RAM: `--num-envs 16-32`
   - 32GB RAM: `--num-envs 32-64`
   - 64GB+ RAM: `--num-envs 64-128`

3. **Batch Size**: Should be divisible by `num_envs * n_steps`

   ```bash
   # Example: 32 envs × 128 steps = 4096 samples
   python train_ppo_sequential.py --batch-size 1024
   ```

### For CPU Clusters

1. **Node Configuration**: Each node should run independent workers
2. **Network**: Use fast interconnect (10GbE+) for parameter synchronization
3. **Storage**: Shared filesystem for checkpoints and logs
4. **Load Balancing**: Distribute environments evenly across nodes

## Typical Training Times (CPU)

**Modern 16-core CPU (e.g., AMD Ryzen 9 5950X):**

- Quick test (1M steps, 16 envs): ~15-20 minutes
- Medium run (10M steps, 32 envs): ~3-4 hours
- Full training (25M steps, 32 envs): ~8-10 hours

**High-end Server CPU (e.g., AMD EPYC 7763, 64 cores):**

- Quick test (1M steps, 64 envs): ~5-8 minutes
- Medium run (10M steps, 64 envs): ~1-2 hours
- Full training (25M steps, 64 envs): ~3-4 hours

**CPU Cluster (4 nodes × 32 cores):**

- Full training (100M steps, 256 envs): ~6-8 hours

## Cluster Deployment Guide

### Prerequisites

- Multiple CPU nodes with SSH access
- Shared filesystem (NFS, Lustre, etc.)
- Python environment on all nodes

### Basic Cluster Setup (Coming Soon)

```bash
# On master node
python train_ppo_distributed.py \
  --config config/ppo_cpu_cluster.yaml \
  --num-workers 4 \
  --master-addr 192.168.1.100

# On worker nodes (automatically launched via SSH or job scheduler)
# Support for SLURM, PBS, and manual launch coming soon
```

## Benchmarking Results

Coming soon: Comparative benchmarks of different algorithms on various Procgen environments with CPU-only training.

## Contributing

Contributions are welcome! Areas of interest:

- Distributed CPU training implementations
- Algorithm-specific CPU optimizations
- Cluster deployment scripts
- Performance profiling and optimization

## Future Work

- [x] CPU-optimized PPO implementation
- [ ] Distributed PPO with Gloo backend
- [ ] A2C/A3C CPU implementation
- [ ] IMPALA CPU-cluster implementation
- [ ] SLURM/PBS cluster integration
- [ ] Multi-node synchronization strategies
- [ ] Advanced CPU profiling tools
- [ ] Memory-mapped experience buffers for large-scale training

## References

- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Procgen Benchmark](https://arxiv.org/abs/1912.01588)
- [Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438)
- [IMPALA: Scalable Distributed Deep-RL](https://arxiv.org/abs/1802.01561)

## License

MIT License
