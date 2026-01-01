#!/usr/bin/env python3
"""
Train IMPALA agent in parallel (actor-learner architecture).
"""

import argparse

import yaml

from src.impala_parallel_trainer import ParallelIMPALATrainer


def main():
    parser = argparse.ArgumentParser(description="Train IMPALA agent (parallel)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/impala_parallel.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create trainer
    trainer = ParallelIMPALATrainer(
        env_name=config["env"]["name"],
        num_actors=config["parallel"]["num_actors"],
        num_envs_per_actor=config["parallel"]["num_envs_per_actor"],
        n_steps=config["training"]["n_steps"],
        total_timesteps=config["training"]["total_timesteps"],
        learning_rate=config["training"]["learning_rate"],
        gamma=config["training"]["gamma"],
        vtrace_clip_rho_threshold=config["impala"]["vtrace_clip_rho_threshold"],
        vtrace_clip_pg_rho_threshold=config["impala"]["vtrace_clip_pg_rho_threshold"],
        entropy_coef=config["impala"]["entropy_coef"],
        value_coef=config["impala"]["value_coef"],
        max_grad_norm=config["impala"]["max_grad_norm"],
        policy_update_frequency=config["parallel"]["policy_update_frequency"],
        log_interval=config["logging"]["log_interval"],
        save_interval=config["logging"]["save_interval"],
        checkpoint_dir=config["logging"]["checkpoint_dir"],
        log_dir=config["logging"]["log_dir"],
        device=config["hardware"]["device"],
        num_threads=config["hardware"]["num_threads"],
        seed=config["env"].get("seed"),
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
