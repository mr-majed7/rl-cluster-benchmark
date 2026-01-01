"""
Parallel IMPALA Trainer with Actor-Learner Architecture.

Multiple actor processes collect experience while a central learner updates the policy.
V-trace handles off-policy correction naturally.
"""

import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Optional

import gym
import gymnasium
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

# Import procgen to register environments
try:
    import procgen  # noqa: F401
except ImportError:
    print("Warning: procgen not installed")

from .cpu_utils import CPUMemoryMonitor, print_cpu_info, setup_cpu_optimization
from .impala import IMPALA
from .utils import Logger


def actor_process(
    actor_id: int,
    env_name: str,
    num_envs_per_actor: int,
    n_steps: int,
    observation_shape: tuple,
    num_actions: int,
    experience_queue: mp.Queue,
    policy_queue: mp.Queue,
    stop_event: mp.Event,
    device: str = "cpu",
    num_threads: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Actor process that collects experience and sends it to the learner.

    Args:
        actor_id: Unique ID for this actor
        env_name: Environment name
        num_envs_per_actor: Number of environments per actor
        n_steps: Rollout length
        observation_shape: Shape of observations (C, H, W)
        num_actions: Number of actions
        experience_queue: Queue to send experience to learner
        policy_queue: Queue to receive policy updates
        stop_event: Event to signal stopping
        device: Device (cpu)
        num_threads: Number of CPU threads
        seed: Random seed
    """
    try:
        # Setup CPU optimization for this process
        if device == "cpu" and num_threads is not None:
            torch.set_num_threads(num_threads)

        # Set seed
        if seed is not None:
            np.random.seed(seed + actor_id)
            torch.manual_seed(seed + actor_id)

        # Create environments
        def make_env():
            if "procgen" in env_name:
                old_env = gym.make(env_name)
                new_env = GymV21CompatibilityV0(env=old_env)
            else:
                new_env = gymnasium.make(env_name)
            return RecordEpisodeStatistics(new_env)

        envs = gymnasium.vector.SyncVectorEnv(
            [make_env for _ in range(num_envs_per_actor)]
        )

        # Create local policy (behavior policy)
        from .models import CNNActorCritic

        local_policy = CNNActorCritic(observation_shape, num_actions).to(device)
        local_policy.eval()  # Always in eval mode for actors

        print(f"Actor {actor_id} started with {num_envs_per_actor} environments")

        # Reset environments
        obs, _ = envs.reset()

        # Main actor loop
        while not stop_event.is_set():
            # Check for policy updates (non-blocking)
            try:
                new_policy_state = policy_queue.get_nowait()
                local_policy.load_state_dict(new_policy_state)
            except queue.Empty:
                pass  # No update available, continue with current policy

            # Collect rollout
            observations = []
            actions = []
            rewards = []
            dones = []
            behavior_logits = []

            for _ in range(n_steps):
                # Transpose observation from (B, H, W, C) to (B, C, H, W)
                obs_transposed = np.transpose(obs, (0, 3, 1, 2))
                observations.append(obs_transposed)

                # Get action from behavior policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_transposed).to(device)
                    logits, _ = local_policy(obs_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    action_tensor = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    action = action_tensor.cpu().numpy()

                behavior_logits.append(logits.cpu().numpy())
                actions.append(action)

                # Step environments
                obs, reward, terminated, truncated, info = envs.step(action)
                done = terminated | truncated

                rewards.append(reward)
                dones.append(done)

            # Add final observation
            obs_transposed = np.transpose(obs, (0, 3, 1, 2))
            observations.append(obs_transposed)

            # Create experience batch
            experience = {
                "observations": np.array(observations),  # [T+1, B, C, H, W]
                "actions": np.array(actions),  # [T, B]
                "rewards": np.array(rewards),  # [T, B]
                "dones": np.array(dones),  # [T, B]
                "behavior_logits": np.array(behavior_logits),  # [T, B, A]
                "actor_id": actor_id,
            }

            # Send experience to learner (non-blocking with timeout)
            try:
                experience_queue.put(experience, timeout=5.0)
            except queue.Full:
                print(f"Actor {actor_id}: Experience queue full, skipping batch")

        print(f"Actor {actor_id} stopped")
        envs.close()

    except Exception as e:
        print(f"Actor {actor_id} error: {e}")
        import traceback

        traceback.print_exc()


class ParallelIMPALATrainer:
    """
    Parallel IMPALA trainer with proper actor-learner architecture.
    """

    def __init__(
        self,
        env_name: str = "procgen-coinrun-v0",
        num_actors: int = 4,
        num_envs_per_actor: int = 8,
        n_steps: int = 128,
        total_timesteps: int = 10_000_000,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        vtrace_clip_rho_threshold: float = 1.0,
        vtrace_clip_pg_rho_threshold: float = 1.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 40.0,
        policy_update_frequency: int = 1,
        log_interval: int = 10,
        save_interval: int = 100,
        checkpoint_dir: str = "./checkpoints/impala_parallel",
        log_dir: str = "./logs",
        device: str = "cpu",
        num_threads: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Parallel IMPALA trainer.

        Args:
            env_name: Procgen environment name
            num_actors: Number of actor processes
            num_envs_per_actor: Environments per actor
            n_steps: Rollout length
            total_timesteps: Total timesteps to train for
            learning_rate: Learning rate
            gamma: Discount factor
            vtrace_clip_rho_threshold: V-trace ρ clipping
            vtrace_clip_pg_rho_threshold: V-trace c clipping
            entropy_coef: Entropy coefficient
            value_coef: Value coefficient
            max_grad_norm: Max gradient norm
            policy_update_frequency: How often to send policy to actors (in updates)
            log_interval: Logging frequency
            save_interval: Checkpoint save frequency
            checkpoint_dir: Checkpoint directory
            log_dir: Log directory
            device: Device (cpu)
            num_threads: CPU threads per process
            seed: Random seed
        """
        self.env_name = env_name
        self.num_actors = num_actors
        self.num_envs_per_actor = num_envs_per_actor
        self.total_envs = num_actors * num_envs_per_actor
        self.n_steps = n_steps
        self.total_timesteps = total_timesteps
        self.policy_update_frequency = policy_update_frequency
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device

        # Setup CPU optimization
        if device == "cpu":
            if num_threads is None:
                num_threads = setup_cpu_optimization()
                # Divide threads among actors + learner
                self.threads_per_actor = max(1, num_threads // (num_actors + 1))
            else:
                setup_cpu_optimization(num_threads)
                self.threads_per_actor = max(1, num_threads // (num_actors + 1))
        else:
            self.threads_per_actor = None

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Get environment info (create temporary env)
        if "procgen" in env_name:
            temp_env = gym.make(env_name)
            temp_env = GymV21CompatibilityV0(env=temp_env)
        else:
            temp_env = gymnasium.make(env_name)

        obs_shape = temp_env.observation_space.shape
        observation_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
        num_actions = temp_env.action_space.n
        temp_env.close()

        print(f"Observation shape: {observation_shape}")
        print(f"Number of actions: {num_actions}")

        # Initialize learner (main IMPALA agent)
        self.agent = IMPALA(
            observation_shape=observation_shape,
            num_actions=num_actions,
            num_envs=self.total_envs,
            learning_rate=learning_rate,
            gamma=gamma,
            vtrace_clip_rho_threshold=vtrace_clip_rho_threshold,
            vtrace_clip_pg_rho_threshold=vtrace_clip_pg_rho_threshold,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            num_threads=self.threads_per_actor if device == "cpu" else None,
        )

        # Multiprocessing setup
        mp.set_start_method("spawn", force=True)
        self.experience_queue = mp.Queue(maxsize=num_actors * 2)
        self.policy_queues = [mp.Queue(maxsize=1) for _ in range(num_actors)]
        self.stop_event = mp.Event()
        self.actor_processes = []

        # Store observation shape and num actions for actors
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.seed = seed

        # Logger
        self.logger = Logger(log_dir, "IMPALA_Parallel")

        # Statistics
        self.global_step = 0
        self.update_step = 0

        # Memory monitoring
        self.memory_monitor = CPUMemoryMonitor() if device == "cpu" else None

        # Print CPU info
        if device == "cpu":
            print_cpu_info()

    def start_actors(self):
        """Start all actor processes."""
        print(f"\nStarting {self.num_actors} actor processes...")

        for actor_id in range(self.num_actors):
            p = mp.Process(
                target=actor_process,
                args=(
                    actor_id,
                    self.env_name,
                    self.num_envs_per_actor,
                    self.n_steps,
                    self.observation_shape,
                    self.num_actions,
                    self.experience_queue,
                    self.policy_queues[actor_id],
                    self.stop_event,
                    self.device,
                    self.threads_per_actor,
                    self.seed,
                ),
            )
            p.start()
            self.actor_processes.append(p)

        print(f"All {self.num_actors} actors started\n")

    def stop_actors(self):
        """Stop all actor processes."""
        print("\nStopping actors...")
        self.stop_event.set()

        for p in self.actor_processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        print("All actors stopped")

    def broadcast_policy(self):
        """Send current policy to all actors."""
        policy_state = self.agent.policy.state_dict()

        for policy_queue in self.policy_queues:
            # Clear old policies
            while not policy_queue.empty():
                try:
                    policy_queue.get_nowait()
                except queue.Empty:
                    break

            # Send new policy
            try:
                policy_queue.put_nowait(policy_state)
            except queue.Full:
                pass  # Actor will get it next time

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Parallel IMPALA Training")
        print("=" * 60)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Actors: {self.num_actors}")
        print(f"Envs per actor: {self.num_envs_per_actor}")
        print(f"Total environments: {self.total_envs}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")

        # Start actors
        self.start_actors()

        # Send initial policy
        self.broadcast_policy()

        # Training metrics
        episode_rewards = []
        start_time = time.time()

        try:
            # Main learner loop
            while self.global_step < self.total_timesteps:
                # Get experience from actors
                try:
                    experience = self.experience_queue.get(timeout=10.0)
                except queue.Empty:
                    print("Warning: No experience received in 10 seconds")
                    continue

                # Train on experience
                train_start = time.time()

                metrics = self.agent.train_step(
                    observations=experience["observations"],
                    actions=experience["actions"],
                    rewards=experience["rewards"],
                    dones=experience["dones"],
                    behavior_logits=experience["behavior_logits"],
                )

                train_time = time.time() - train_start

                self.update_step += 1

                # Update global step
                batch_timesteps = (
                    experience["actions"].shape[0] * experience["actions"].shape[1]
                )
                self.global_step += batch_timesteps

                # Update memory monitor
                if self.memory_monitor:
                    self.memory_monitor.update()

                # Broadcast policy periodically
                if self.update_step % self.policy_update_frequency == 0:
                    self.broadcast_policy()

                # Logging
                if self.update_step % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    fps = self.global_step / elapsed if elapsed > 0 else 0

                    # Log to TensorBoard
                    self.logger.log(
                        "train/policy_loss", metrics["policy_loss"], self.global_step
                    )
                    self.logger.log(
                        "train/value_loss", metrics["value_loss"], self.global_step
                    )
                    self.logger.log(
                        "train/entropy", metrics["entropy"], self.global_step
                    )
                    self.logger.log(
                        "train/grad_norm", metrics["grad_norm"], self.global_step
                    )
                    self.logger.log("time/fps", fps, self.global_step)
                    self.logger.log("time/train_time", train_time, self.global_step)

                    # Memory usage
                    if self.memory_monitor:
                        mem_stats = self.memory_monitor.get_stats()
                        self.logger.log(
                            "system/memory_gb",
                            mem_stats["current_memory_gb"],
                            self.global_step,
                        )

                    # Console output
                    print(
                        f"Step: {self.global_step:,} | "
                        f"Update: {self.update_step} | "
                        f"FPS: {fps:.0f} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Entropy: {metrics['entropy']:.4f}"
                    )

                # Save checkpoint
                if self.update_step % self.save_interval == 0:
                    checkpoint_path = (
                        self.checkpoint_dir
                        / f"impala_parallel_step_{self.global_step}.pt"
                    )
                    self.agent.save(str(checkpoint_path))
                    print(f"Saved checkpoint: {checkpoint_path}")

        finally:
            # Stop actors
            self.stop_actors()

        # Save final checkpoint
        final_path = self.checkpoint_dir / "impala_parallel_final.pt"
        self.agent.save(str(final_path))
        print(f"\nTraining complete! Final checkpoint saved: {final_path}")

        # Print final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Total timesteps: {self.global_step:,}")
        print(f"Total updates: {self.update_step}")
        print(f"Time elapsed: {elapsed:.2f}s ({elapsed/3600:.2f}h)")
        print(f"Average FPS: {self.global_step/elapsed:.0f}")
        print(f"Actors: {self.num_actors}")
        print(f"Total environments: {self.total_envs}")

        if self.memory_monitor:
            mem_stats = self.memory_monitor.get_stats()
            print(f"Peak memory usage: {mem_stats['peak_memory_gb']:.2f} GB")

        print("=" * 60)
