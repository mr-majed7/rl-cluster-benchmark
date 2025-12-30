"""Parallel training pipeline for PPO using multiprocessing (CPU)."""

import multiprocessing as mp
import time
from typing import Dict, Optional

import gym
import gymnasium
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from tqdm import tqdm

# Import procgen to register environments
try:
    import procgen  # noqa: F401
except ImportError:
    print("Warning: procgen not installed")

from .buffer import RolloutBuffer
from .cpu_utils import CPUMemoryMonitor, print_cpu_info, setup_cpu_optimization
from .ppo import PPO
from .utils import Logger


def worker_process(
    worker_id: int,
    env_name: str,
    num_envs_per_worker: int,
    n_steps: int,
    observation_shape: tuple,
    shared_policy,
    data_queue: mp.Queue,
    control_queue: mp.Queue,
    stop_event: mp.Event,
):
    """Worker process that collects rollouts using shared policy.

    Args:
        worker_id: Unique worker identifier
        env_name: Environment name
        num_envs_per_worker: Number of environments per worker
        n_steps: Steps per rollout
        observation_shape: Shape of observations
        shared_policy: Shared policy model with shared memory tensors
        data_queue: Queue to send collected data
        control_queue: Queue to receive control signals
        stop_event: Event to signal worker to stop
    """
    # Set CPU optimization for this worker
    setup_cpu_optimization(num_threads=None)  # Auto-detect per worker

    # Create environments for this worker
    def make_env():
        if "procgen" in env_name:
            old_env = gym.make(env_name)
            new_env = GymV21CompatibilityV0(env=old_env)
        else:
            new_env = gymnasium.make(env_name)
        return RecordEpisodeStatistics(new_env)

    envs = gymnasium.vector.SyncVectorEnv(
        [make_env for _ in range(num_envs_per_worker)]
    )

    # Get action space
    num_actions = envs.single_action_space.n

    # Use shared policy directly (already in shared memory)
    local_policy = shared_policy
    local_policy.eval()

    def _preprocess_obs(obs: np.ndarray) -> np.ndarray:
        """Preprocess observations."""
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))
        return obs

    print(f"Worker {worker_id} started with {num_envs_per_worker} environments")

    while not stop_event.is_set():
        try:
            # Wait for signal to collect (longer timeout for initialization)
            command = control_queue.get(timeout=30.0)

            if command == "collect":
                # Policy is already shared, no need to sync

                # Collect rollout
                observations = []
                actions = []
                rewards = []
                dones = []
                values = []
                log_probs = []
                episode_infos = []

                obs, _ = envs.reset()
                obs = _preprocess_obs(obs)

                for _ in range(n_steps):
                    obs_tensor = torch.from_numpy(obs).float()

                    with torch.no_grad():
                        action, log_prob, _, value = local_policy.get_action_and_value(
                            obs_tensor
                        )

                    next_obs, reward, terminated, truncated, info = envs.step(
                        action.numpy()
                    )
                    done = np.logical_or(terminated, truncated)

                    # Store transition
                    observations.append(obs)
                    actions.append(action.numpy())
                    rewards.append(reward)
                    dones.append(done)
                    values.append(value.numpy())
                    log_probs.append(log_prob.numpy())

                    obs = _preprocess_obs(next_obs)

                    # Collect episode info
                    if "final_info" in info:
                        for final_info in info["final_info"]:
                            if final_info is not None and "episode" in final_info:
                                episode_infos.append(
                                    {
                                        "reward": final_info["episode"]["r"],
                                        "length": final_info["episode"]["l"],
                                    }
                                )

                # Get final value for last observation
                obs_tensor = torch.from_numpy(obs).float()
                with torch.no_grad():
                    _, _, _, last_value = local_policy.get_action_and_value(obs_tensor)

                # Send collected data back
                data = {
                    "observations": np.array(observations),
                    "actions": np.array(actions),
                    "rewards": np.array(rewards),
                    "dones": np.array(dones),
                    "values": np.array(values),
                    "log_probs": np.array(log_probs),
                    "last_value": last_value.numpy(),
                    "episode_infos": episode_infos,
                    "worker_id": worker_id,
                }

                data_queue.put(data)

            elif command == "stop":
                break

        except Exception as e:
            import traceback

            print(f"Worker {worker_id} error: {e}")
            traceback.print_exc()
            break

    envs.close()
    print(f"Worker {worker_id} stopped")


class ParallelTrainer:
    """Parallel training pipeline for PPO using multiprocessing."""

    def __init__(
        self,
        env_name: str = "procgen-coinrun-v0",
        num_workers: int = 4,
        num_envs_per_worker: int = 8,
        n_steps: int = 128,
        total_timesteps: int = 25_000_000,
        batch_size: int = 2048,
        n_epochs: int = 4,
        learning_rate: float = 5e-4,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        log_interval: int = 1,
        save_interval: int = 50,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        seed: Optional[int] = None,
        device: str = "cpu",
        num_threads: Optional[int] = None,
    ):
        """Initialize parallel trainer.

        Args:
            env_name: Environment name
            num_workers: Number of worker processes
            num_envs_per_worker: Environments per worker
            n_steps: Steps per rollout per worker
            total_timesteps: Total timesteps to train
            batch_size: Batch size for PPO updates
            n_epochs: Epochs per update
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            log_interval: Logging interval
            save_interval: Save interval
            checkpoint_dir: Checkpoint directory
            log_dir: Log directory
            seed: Random seed
            device: Device (cpu)
            num_threads: Main process threads
        """
        self.env_name = env_name
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker
        self.total_envs = num_workers * num_envs_per_worker
        self.n_steps = n_steps
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        # Set CPU optimization FIRST, before any torch operations
        if device == "cpu":
            setup_cpu_optimization(num_threads)

        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create a single env to get specs
        def make_env():
            if "procgen" in env_name:
                old_env = gym.make(env_name)
                new_env = GymV21CompatibilityV0(env=old_env)
            else:
                new_env = gymnasium.make(env_name)
            return new_env

        env = make_env()
        obs_shape = env.observation_space.shape
        self.observation_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        self.num_actions = env.action_space.n
        env.close()

        print("\nParallel PPO Configuration:")
        print(f"  Workers: {num_workers}")
        print(f"  Envs per worker: {num_envs_per_worker}")
        print(f"  Total environments: {self.total_envs}")
        print(f"  Observation shape: {self.observation_shape}")
        print(f"  Number of actions: {self.num_actions}")

        # Initialize PPO agent (main process only)
        # CPU optimization already done earlier
        self.agent = PPO(
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            num_envs=self.total_envs,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            num_threads=num_threads,
        )

        # Initialize logger
        self.logger = Logger(log_dir, algorithm="PPO_Parallel")

        # Training state
        self.global_step = 0
        self.update_step = 0

        # Memory monitoring
        self.memory_monitor = CPUMemoryMonitor()

        # Print CPU info
        print_cpu_info()

        # Multiprocessing setup
        # Share the policy model's parameters across processes
        self.agent.policy.share_memory()

        self.data_queue = mp.Queue()
        self.control_queues = [mp.Queue() for _ in range(num_workers)]
        self.stop_event = mp.Event()
        self.workers = []

    def start_workers(self):
        """Start all worker processes."""
        print(f"\nStarting {self.num_workers} worker processes...")

        for i in range(self.num_workers):
            worker = mp.Process(
                target=worker_process,
                args=(
                    i,
                    self.env_name,
                    self.num_envs_per_worker,
                    self.n_steps,
                    self.observation_shape,
                    self.agent.policy,
                    self.data_queue,
                    self.control_queues[i],
                    self.stop_event,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        # Give workers time to initialize
        time.sleep(2)
        print(f"All {self.num_workers} workers started")

    def collect_rollouts(self) -> Dict:
        """Collect rollouts from all workers in parallel.

        Returns:
            Aggregated rollout data and statistics
        """
        # Policy is already in shared memory, workers see updates automatically

        # Signal all workers to collect
        for queue in self.control_queues:
            queue.put("collect")

        # Collect data from all workers
        all_observations = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []
        all_last_values = []
        episode_infos = []

        for _ in range(self.num_workers):
            data = self.data_queue.get()

            all_observations.append(data["observations"])
            all_actions.append(data["actions"])
            all_rewards.append(data["rewards"])
            all_dones.append(data["dones"])
            all_values.append(data["values"])
            all_log_probs.append(data["log_probs"])
            all_last_values.append(data["last_value"])
            episode_infos.extend(data["episode_infos"])

        # Aggregate data
        aggregated_data = {
            "observations": np.concatenate(
                all_observations, axis=1
            ),  # (steps, total_envs, ...)
            "actions": np.concatenate(all_actions, axis=1),
            "rewards": np.concatenate(all_rewards, axis=1),
            "dones": np.concatenate(all_dones, axis=1),
            "values": np.concatenate(all_values, axis=1),
            "log_probs": np.concatenate(all_log_probs, axis=1),
            "last_values": np.concatenate(all_last_values, axis=0),
        }

        # Update global step
        self.global_step += self.n_steps * self.total_envs

        # Compute episode statistics
        stats = {}
        if episode_infos:
            rewards = [info["reward"] for info in episode_infos]
            lengths = [info["length"] for info in episode_infos]
            stats = {
                "episode_reward_mean": np.mean(rewards),
                "episode_reward_std": np.std(rewards),
                "episode_length_mean": np.mean(lengths),
                "num_episodes": len(rewards),
            }

        return aggregated_data, stats

    def train(self):
        """Run parallel training loop."""
        print(f"\nStarting parallel training for {self.total_timesteps:,} timesteps")
        print(f"Updates per rollout: {self.n_epochs}")
        print(f"Steps per rollout: {self.n_steps}")
        print(
            f"Total updates: {self.total_timesteps // (self.n_steps * self.total_envs)}"
        )
        print("-" * 70)

        # Start workers
        self.start_workers()

        num_updates = self.total_timesteps // (self.n_steps * self.total_envs)
        start_time = time.time()

        try:
            for update in tqdm(range(num_updates), desc="Training"):
                update_start_time = time.time()

                # Collect rollouts from all workers in parallel
                rollout_data, rollout_stats = self.collect_rollouts()

                # Create buffer from aggregated data
                buffer = RolloutBuffer(
                    buffer_size=self.n_steps,
                    observation_shape=self.observation_shape,
                    num_envs=self.total_envs,
                    device=torch.device(self.device),
                )

                # Fill buffer
                for t in range(self.n_steps):
                    buffer.add(
                        rollout_data["observations"][t],
                        torch.from_numpy(rollout_data["actions"][t]),
                        rollout_data["rewards"][t],
                        rollout_data["dones"][t],
                        torch.from_numpy(rollout_data["values"][t]),
                        torch.from_numpy(rollout_data["log_probs"][t]),
                    )

                # Compute returns and advantages
                buffer.compute_returns_and_advantages(
                    torch.from_numpy(rollout_data["last_values"]),
                    self.agent.gamma,
                    self.agent.gae_lambda,
                )

                # Train on collected data
                train_stats = self.agent.train_step(
                    buffer, self.batch_size, self.n_epochs
                )

                # Update counter
                self.update_step += 1
                update_time = time.time() - update_start_time

                # Logging
                if self.update_step % self.log_interval == 0:
                    fps = (self.n_steps * self.total_envs) / update_time
                    elapsed_time = time.time() - start_time

                    log_data = {
                        "time/fps": fps,
                        "time/total_timesteps": self.global_step,
                        "time/elapsed_time": elapsed_time,
                        "train/policy_loss": train_stats["policy_loss"],
                        "train/value_loss": train_stats["value_loss"],
                        "train/entropy_loss": train_stats["entropy_loss"],
                        "train/total_loss": train_stats["total_loss"],
                        "train/clip_fraction": train_stats["clip_fraction"],
                        "train/approx_kl": train_stats["approx_kl"],
                    }

                    # Add memory stats
                    self.memory_monitor.update()
                    mem_stats = self.memory_monitor.get_stats()
                    log_data["system/memory_gb"] = mem_stats["current_memory_gb"]
                    log_data["system/peak_memory_gb"] = mem_stats["peak_memory_gb"]

                    if rollout_stats:
                        log_data.update(
                            {
                                "rollout/ep_reward_mean": rollout_stats[
                                    "episode_reward_mean"
                                ],
                                "rollout/ep_reward_std": rollout_stats[
                                    "episode_reward_std"
                                ],
                                "rollout/ep_length_mean": rollout_stats[
                                    "episode_length_mean"
                                ],
                            }
                        )

                        tqdm.write(
                            f"Update {self.update_step}/{num_updates} | "
                            f"Step {self.global_step:,}/{self.total_timesteps:,} | "
                            f"FPS: {fps:.0f} | "
                            f"Reward: {rollout_stats['episode_reward_mean']:.2f} ± {rollout_stats['episode_reward_std']:.2f}"
                        )

                    self.logger.log(log_data, self.global_step)

                # Save checkpoint
                if self.update_step % self.save_interval == 0:
                    import os

                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    checkpoint_path = (
                        f"{self.checkpoint_dir}/ppo_parallel_step_{self.global_step}.pt"
                    )
                    self.agent.save(checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")

            # Final save
            import os

            os.makedirs(self.checkpoint_dir, exist_ok=True)
            final_path = f"{self.checkpoint_dir}/ppo_parallel_final.pt"
            self.agent.save(final_path)
            print(f"\nTraining complete! Final model saved: {final_path}")
            print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")

        finally:
            # Stop workers
            self.stop_workers()
            self.logger.close()

    def stop_workers(self):
        """Stop all worker processes."""
        print("\nStopping workers...")
        self.stop_event.set()

        for queue in self.control_queues:
            try:
                queue.put("stop", timeout=1.0)
            except:
                pass

        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()

        print("All workers stopped")
