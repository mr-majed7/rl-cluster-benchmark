#!/usr/bin/env python3
"""Train PPO agent using sequential pipeline on Procgen environments."""
import argparse
import os

import torch

from src.config import load_config, save_config
from src.trainer import SequentialTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent sequentially on Procgen"
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="config/ppo_sequential.yaml",
        help="Path to config file",
    )

    # Environment
    parser.add_argument("--env", type=str, help="Environment name (overrides config)")
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Training
    parser.add_argument("--total-timesteps", type=int, help="Total training timesteps")
    parser.add_argument("--n-steps", type=int, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--n-epochs", type=int, help="Epochs per update")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, help="Value function coefficient")

    # Logging
    parser.add_argument("--log-interval", type=int, help="Logging interval")
    parser.add_argument("--save-interval", type=int, help="Checkpoint save interval")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, help="Log directory")

    # Hardware
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="Device to use"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load config file
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = {}

    # Override config with command line arguments
    if args.env:
        config.setdefault("env", {})["name"] = args.env
    if args.num_envs:
        config.setdefault("env", {})["num_envs"] = args.num_envs
    if args.seed is not None:
        config.setdefault("env", {})["seed"] = args.seed

    if args.total_timesteps:
        config.setdefault("training", {})["total_timesteps"] = args.total_timesteps
    if args.n_steps:
        config.setdefault("training", {})["n_steps"] = args.n_steps
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.n_epochs:
        config.setdefault("training", {})["n_epochs"] = args.n_epochs

    if args.learning_rate:
        config.setdefault("ppo", {})["learning_rate"] = args.learning_rate
    if args.gamma:
        config.setdefault("ppo", {})["gamma"] = args.gamma
    if args.gae_lambda:
        config.setdefault("ppo", {})["gae_lambda"] = args.gae_lambda
    if args.clip_range:
        config.setdefault("ppo", {})["clip_range"] = args.clip_range
    if args.ent_coef:
        config.setdefault("ppo", {})["ent_coef"] = args.ent_coef
    if args.vf_coef:
        config.setdefault("ppo", {})["vf_coef"] = args.vf_coef

    if args.log_interval:
        config.setdefault("logging", {})["log_interval"] = args.log_interval
    if args.save_interval:
        config.setdefault("logging", {})["save_interval"] = args.save_interval
    if args.checkpoint_dir:
        config.setdefault("logging", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir:
        config.setdefault("logging", {})["log_dir"] = args.log_dir

    if args.device:
        config.setdefault("hardware", {})["device"] = args.device

    # Extract parameters
    env_config = config.get("env", {})
    training_config = config.get("training", {})
    ppo_config = config.get("ppo", {})
    logging_config = config.get("logging", {})
    hardware_config = config.get("hardware", {})

    # Create checkpoint directory
    checkpoint_dir = logging_config.get(
        "checkpoint_dir", "./checkpoints/ppo_sequential"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the effective config
    save_config(config, os.path.join(checkpoint_dir, "config.yaml"))

    # Print configuration
    print("\n" + "=" * 60)
    print("PPO Sequential Training Configuration")
    print("=" * 60)
    print(f"Environment: {env_config.get('name', 'procgen:procgen-coinrun-v0')}")
    print(f"Number of environments: {env_config.get('num_envs', 64)}")
    print(f"Total timesteps: {training_config.get('total_timesteps', 25_000_000):,}")
    print(
        f"Device: {hardware_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')}"
    )
    print("=" * 60 + "\n")

    # Initialize trainer
    trainer = SequentialTrainer(
        env_name=env_config.get("name", "procgen:procgen-coinrun-v0"),
        num_envs=env_config.get("num_envs", 64),
        n_steps=training_config.get("n_steps", 256),
        total_timesteps=training_config.get("total_timesteps", 25_000_000),
        batch_size=training_config.get("batch_size", 2048),
        n_epochs=training_config.get("n_epochs", 3),
        learning_rate=ppo_config.get("learning_rate", 5e-4),
        gamma=ppo_config.get("gamma", 0.999),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        log_interval=logging_config.get("log_interval", 1),
        save_interval=logging_config.get("save_interval", 100),
        eval_episodes=logging_config.get("eval_episodes", 10),
        checkpoint_dir=checkpoint_dir,
        log_dir=logging_config.get("log_dir", "./logs"),
        seed=env_config.get("seed"),
        device=hardware_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        trainer.envs.close()
        trainer.logger.close()


if __name__ == "__main__":
    main()
