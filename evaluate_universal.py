#!/usr/bin/env python3
"""Evaluate trained RL agents (PPO, IMPALA) on Procgen environments."""
import argparse

import gym as old_gym
import gymnasium
import numpy as np
import torch
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from tqdm import tqdm

# Import procgen to register environments
try:
    import procgen  # noqa: F401
except ImportError:
    print("Warning: procgen not installed")

from src.impala import IMPALA
from src.ppo import PPO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RL agent on Procgen")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["ppo", "impala"],
        help="Algorithm type (ppo or impala)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="procgen-coinrun-v0",
        help="Environment name",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100, help="Number of episodes to evaluate"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic policy"
    )

    return parser.parse_args()


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """Preprocess observation - no normalization (models trained on raw pixels)."""
    obs = obs.astype(np.float32)  # Convert to float32 but keep [0, 255] range
    obs = np.transpose(obs, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    return np.expand_dims(obs, axis=0)  # Add batch dimension


def evaluate(
    agent, algorithm: str, env, num_episodes: int, deterministic: bool = False
):
    """Evaluate agent for multiple episodes.

    Args:
        agent: RL agent to evaluate (PPO or IMPALA)
        algorithm: Algorithm type ('ppo' or 'impala')
        env: Environment to evaluate on
        num_episodes: Number of episodes
        deterministic: Use deterministic policy

    Returns:
        Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        obs = preprocess_obs(obs)

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get action based on algorithm
            if algorithm == "ppo":
                action, _, _ = agent.predict(obs, deterministic=deterministic)
                action = action.item()
            else:  # impala
                action, _, _ = agent.get_action(obs, deterministic=deterministic)
                action = action.item()

            obs, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess_obs(obs)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create environment
    if "procgen" in args.env:
        old_env = old_gym.make(args.env)
        env = GymV21CompatibilityV0(env=old_env)
    else:
        env = gymnasium.make(args.env, render_mode="human" if args.render else None)

    # Get observation shape and action space
    obs_shape = env.observation_space.shape
    observation_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
    num_actions = env.action_space.n

    print(f"\n{'='*60}")
    print(f"Evaluating {args.algorithm.upper()} Agent")
    print(f"{'='*60}")
    print(f"Environment: {args.env}")
    print(f"Observation shape: {observation_shape}")
    print(f"Number of actions: {num_actions}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print("-" * 60)

    # Initialize agent based on algorithm
    if args.algorithm == "ppo":
        agent = PPO(
            observation_shape=observation_shape,
            num_actions=num_actions,
            device=args.device,
        )
    else:  # impala
        agent = IMPALA(
            observation_shape=observation_shape,
            num_actions=num_actions,
            device=args.device,
            num_threads=4,  # Use only 4 threads for evaluation (inference only)
        )

    # Load checkpoint
    agent.load(args.checkpoint)
    print(f"✓ Loaded checkpoint from: {args.checkpoint}\n")

    # Evaluate
    stats = evaluate(agent, args.algorithm, env, args.num_episodes, args.deterministic)

    # Print results
    print("\n" + "=" * 60)
    print(f"{args.algorithm.upper()} Evaluation Results")
    print("=" * 60)
    print(f"Episodes: {args.num_episodes}")
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
    print(
        f"Mean Episode Length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}"
    )
    print("=" * 60 + "\n")

    env.close()


if __name__ == "__main__":
    main()
