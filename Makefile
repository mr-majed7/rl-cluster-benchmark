.PHONY: help install test train train-parallel train-timed train-parallel-timed train-impala train-impala-timed train-impala-parallel train-impala-parallel-timed train-fast evaluate evaluate-ppo evaluate-impala compare tensorboard clean

help:
	@echo "RL Cluster Benchmark - Available commands:"
	@echo ""
	@echo "PPO Training:"
	@echo "  make train                  - Train PPO agent (sequential, full epochs)"
	@echo "  make train-parallel         - Train PPO agent (parallel, full epochs)"
	@echo "  make train-timed            - Timed PPO sequential (1 hour)"
	@echo "  make train-parallel-timed   - Timed PPO parallel (1 hour)"
	@echo ""
	@echo "IMPALA Training:"
	@echo "  make train-impala           - Train IMPALA agent (sequential)"
	@echo "  make train-impala-timed     - Timed IMPALA sequential (30 min)"
	@echo "  make train-impala-parallel  - Train IMPALA agent (parallel)"
	@echo "  make train-impala-parallel-timed - Timed IMPALA parallel (30 min)"
	@echo ""
	@echo "General:"
	@echo "  make install                - Install dependencies"
	@echo "  make test                   - Run quick tests"
	@echo "  make train-fast             - Quick training test"
	@echo "  make evaluate               - Evaluate PPO model (legacy)"
	@echo "  make evaluate-ppo           - Evaluate PPO model"
	@echo "  make evaluate-impala        - Evaluate IMPALA model"
	@echo "  make compare                - Compare sequential vs parallel results"
	@echo "  make tensorboard            - Launch TensorBoard"
	@echo "  make clean                  - Clean generated files"
	@echo ""

install:
	pip install -r requirements.txt

test:
	python quick_test.py

train:
	python train_ppo_sequential.py --config config/ppo_sequential.yaml

train-parallel:
	python train_ppo_parallel.py --config config/ppo_parallel.yaml

train-timed:
	python train_timed.py --duration 1.0 --output-dir ./benchmarks/sequential_1h --num-envs 32

train-parallel-timed:
	python train_parallel_timed.py --duration 1.0 --output-dir ./benchmarks/parallel_1h --num-workers 4 --num-envs-per-worker 8

train-impala:
	python train_impala_sequential.py --config config/impala_sequential.yaml

train-impala-timed:
	python train_impala_sequential_timed.py --duration 0.5 --output-dir ./benchmarks/impala_sequential_30min

train-impala-parallel:
	python train_impala_parallel.py --config config/impala_parallel.yaml

train-impala-parallel-timed:
	python train_impala_parallel_timed.py --duration 0.5 --output-dir ./benchmarks/impala_parallel_30min

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

evaluate-ppo:
	@if [ -f "checkpoints/ppo_sequential/ppo_sequential_final.pt" ]; then \
		python evaluate_universal.py --algorithm ppo --checkpoint checkpoints/ppo_sequential/ppo_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic; \
	elif [ -f "benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt" ]; then \
		python evaluate_universal.py --algorithm ppo --checkpoint benchmarks/sequential_1h/checkpoints/ppo_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic; \
	else \
		echo "No PPO model found. Train one first with 'make train' or 'make train-timed'"; \
	fi

evaluate-impala:
	@if [ -f "checkpoints/impala_sequential/impala_sequential_final.pt" ]; then \
		python evaluate_universal.py --algorithm impala --checkpoint checkpoints/impala_sequential/impala_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic; \
	elif [ -f "benchmarks/impala_seq_30min/checkpoints/impala_sequential_final.pt" ]; then \
		python evaluate_universal.py --algorithm impala --checkpoint benchmarks/impala_seq_30min/checkpoints/impala_sequential_final.pt --env procgen-coinrun-v0 --num-episodes 100 --deterministic; \
	else \
		echo "No IMPALA model found. Train one first with 'make train-impala' or 'make train-impala-timed'"; \
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

