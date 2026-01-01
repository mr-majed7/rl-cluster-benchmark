#!/bin/bash
# 30-minute comparison script for high-end machine (28 threads, 64GB RAM)

echo "================================================================"
echo "RL Training Comparison - 30 Minutes Each"
echo "Machine: Intel i7-14700K (28 threads, 64GB RAM)"
echo "================================================================"
echo ""

# Activate virtual environment
source venv/bin/activate

echo "🚀 Starting Sequential Training (30 minutes)..."
echo "Config: 112 environments, 28 threads"
echo "Expected FPS: 4,000-6,000"
echo "------------------------------------------------"

python train_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/sequential_30min_highend \
  --config config/ppo_sequential_highend.yaml

echo ""
echo "✅ Sequential training completed!"
echo ""

echo "🚀 Starting Parallel Training (30 minutes)..."
echo "Config: 7 workers × 16 envs = 112 total environments"
echo "Expected FPS: 12,000-18,000 (if working properly)"
echo "------------------------------------------------"

python train_parallel_timed.py \
  --duration 0.5 \
  --output-dir ./benchmarks/parallel_30min_highend \
  --config config/ppo_parallel_highend.yaml

echo ""
echo "✅ Parallel training completed!"
echo ""

echo "📊 Generating Comparison Report..."
echo "=================================================="

python compare_results.py \
  --sequential ./benchmarks/sequential_30min_highend \
  --parallel ./benchmarks/parallel_30min_highend \
  --output ./benchmarks/comparison_30min_highend.png

echo ""
echo "🎉 Comparison complete!"
echo "Check ./benchmarks/ for results and comparison.png for plots"
