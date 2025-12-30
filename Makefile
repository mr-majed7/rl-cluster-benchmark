.PHONY: help install test train train-timed train-parallel train-fast evaluate compare tensorboard clean

help:
	@echo "RL Cluster Benchmark - Available commands:"
	@echo ""
	@echo "  make install         - Install dependencies"
	@echo "  make test            - Run quick tests"
	@echo "  make train           - Train PPO agent (sequential)"
	@echo "  make train-timed     - Timed sequential training (1 hour)"
	@echo "  make train-parallel  - Timed parallel training (1 hour)"
	@echo "  make train-fast      - Quick training test"
	@echo "  make evaluate        - Evaluate trained model"
	@echo "  make compare         - Compare sequential vs parallel results"
	@echo "  make tensorboard     - Launch TensorBoard"
	@echo "  make clean           - Clean generated files"
	@echo ""

install:
	pip install -r requirements.txt

test:
	python quick_test.py

train:
	python train_ppo_sequential.py --config config/ppo_sequential.yaml

train-timed:
	python train_timed.py --duration 1.0 --output-dir ./benchmarks/sequential_1h --num-envs 32

train-parallel:
	python train_parallel_timed.py --duration 1.0 --output-dir ./benchmarks/parallel_1h --num-workers 4 --num-envs-per-worker 8

train-fast:
	python train_ppo_sequential.py --total-timesteps 500000 --num-envs 16 --save-interval 10

evaluate:
	@if [ -f "checkpoints/ppo_sequential/ppo_sequential_final.pt" ]; then \
		python evaluate.py --checkpoint checkpoints/ppo_sequential/ppo_sequential_final.pt --num-episodes 50; \
	elif [ -f "benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt" ]; then \
		python evaluate.py --checkpoint benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt --num-episodes 50; \
	else \
		echo "No checkpoint found. Train a model first with 'make train' or 'make train-timed'"; \
	fi

compare:
	@if [ -d "benchmarks/sequential_1h" ] && [ -d "benchmarks/parallel_1h" ]; then \
		python compare_results.py --sequential benchmarks/sequential_1h --parallel benchmarks/parallel_1h --output benchmarks/comparison.png; \
	elif [ -d "benchmarks/sequential_1h" ]; then \
		python compare_results.py --sequential benchmarks/sequential_1h --output benchmarks/sequential_analysis.png; \
	else \
		echo "No benchmark results found. Run 'make train-timed' and/or 'make train-parallel' first"; \
	fi

tensorboard:
	tensorboard --logdir logs/

clean:
	rm -rf __pycache__ src/__pycache__
	rm -rf logs/* checkpoints/* benchmarks/*
	rm -rf *.pyc src/*.pyc
	rm -rf comparison.png comparison_report.txt
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

