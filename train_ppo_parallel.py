#!/usr/bin/env python3
"""Train PPO agent in parallel mode on Procgen environments."""
import argparse

import yaml

from src.parallel_trainer import ParallelTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO with parallel workers")

    parser.add_argument(
        "--config",
        type=str,
        default="config/ppo_parallel.yaml",
        help="Path to config file",
    )
    parser.add_argument("--env", type=str, help="Environment name (overrides config)")
    parser.add_argument(
        "--num-workers", type=int, help="Number of workers (overrides config)"
    )
    parser.add_argument(
        "--num-envs-per-worker",
        type=int,
        help="Environments per worker (overrides config)",
    )
    parser.add_argument(
        "--total-timesteps", type=int, help="Total timesteps (overrides config)"
    )
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, help="Device to use (overrides config)")
    parser.add_argument(
        "--num-threads", type=int, help="Number of CPU threads (overrides config)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.env:
        config["env"]["name"] = args.env
    if args.num_workers:
        config["parallel"]["num_workers"] = args.num_workers
    if args.num_envs_per_worker:
        config["parallel"]["num_envs_per_worker"] = args.num_envs_per_worker
    if args.total_timesteps:
        config["training"]["total_timesteps"] = args.total_timesteps
    if args.seed:
        config["training"]["seed"] = args.seed
    if args.device:
        config["training"]["device"] = args.device
    if args.num_threads:
        config["training"]["num_threads"] = args.num_threads

    # Print configuration
    print("\n" + "=" * 70)
    print("PPO Parallel Training")
    print("=" * 70)
    print(f"Environment: {config['env']['name']}")
    print(f"Workers: {config['parallel']['num_workers']}")
    print(f"Envs per worker: {config['parallel']['num_envs_per_worker']}")
    print(
        f"Total envs: {config['parallel']['num_workers'] * config['parallel']['num_envs_per_worker']}"
    )
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Device: {config['training']['device']}")
    if config["training"].get("num_threads"):
        print(f"CPU threads: {config['training']['num_threads']}")
    print("=" * 70 + "\n")

    # Initialize trainer
    trainer = ParallelTrainer(
        env_name=config["env"]["name"],
        num_workers=config["parallel"]["num_workers"],
        num_envs_per_worker=config["parallel"]["num_envs_per_worker"],
        n_steps=config["training"]["n_steps"],
        total_timesteps=config["training"]["total_timesteps"],
        batch_size=config["training"]["batch_size"],
        n_epochs=config["training"]["n_epochs"],
        learning_rate=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        gae_lambda=config["training"]["gae_lambda"],
        clip_range=config["training"]["clip_range"],
        ent_coef=config["training"]["ent_coef"],
        vf_coef=config["training"]["vf_coef"],
        max_grad_norm=config["training"]["max_grad_norm"],
        log_interval=config["training"]["log_interval"],
        save_interval=config["training"]["save_interval"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
        log_dir=config["training"]["log_dir"],
        seed=config["training"].get("seed"),
        device=config["training"]["device"],
        num_threads=config["training"].get("num_threads"),
    )

    # Train
    trainer.train()

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
