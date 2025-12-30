#!/usr/bin/env python3
"""Quick test script to verify the installation and basic functionality."""
import torch

print("=" * 60)
print("Quick Test: RL Cluster Benchmark")
print("=" * 60)

# Check PyTorch
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")

# Check Gymnasium
try:
    import gymnasium

    print(f"✓ Gymnasium version: {gymnasium.__version__}")
except ImportError as e:
    print(f"✗ Gymnasium import failed: {e}")

# Check Procgen
try:
    import gym as old_gym
    import procgen  # noqa
    from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

    print("✓ Procgen installed")

    # Try creating an environment with old gym + shimmy wrapper
    old_env = old_gym.make("procgen-coinrun-v0")
    env = GymV21CompatibilityV0(env=old_env)
    obs, _ = env.reset()
    print("✓ Procgen environment created successfully")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    env.close()
except Exception as e:
    print(f"✗ Procgen test failed: {e}")
    import traceback

    traceback.print_exc()

# Check our modules
try:
    from src.models import CNNActorCritic

    print("✓ All project modules imported successfully")

    # Quick model test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNActorCritic((3, 64, 64), 15).to(device)
    dummy_input = torch.randn(4, 3, 64, 64).to(device)
    logits, value = model(dummy_input)
    print("✓ Model forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Value shape: {value.shape}")

except Exception as e:
    print(f"✗ Module test failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete! You're ready to start training.")
print("=" * 60)
print("\nTo start training, run:")
print("  python train_ppo_sequential.py --config config/ppo_sequential.yaml")
print("\nFor a quick test with fewer steps, run:")
print("  python train_ppo_sequential.py --total-timesteps 100000 --num-envs 4")
print("\nFor timed 1-hour benchmark:")
print("  python train_timed.py --duration 1.0 --output-dir ./benchmarks/sequential_1h")
print("=" * 60 + "\n")
