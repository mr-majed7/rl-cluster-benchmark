# Parallel Training Status

## Current Status: ⚠️ Not Ready for Use

The parallel PPO implementation exists but has **significant performance issues** that make it **slower than sequential training**.

## Performance Comparison

| Mode | FPS | Iteration Time | Status |
|------|-----|----------------|--------|
| Sequential (32 envs) | ~1,500-2,000 | ~0.3-0.5s | ✅ Works great |
| Parallel (4 workers × 8 envs) | ~70-80 | ~14s | ❌ Very slow |

**Expected:** Parallel should be 2-3x faster
**Actual:** Parallel is ~20x slower

## Root Cause Analysis

### Problem 1: Shared Memory Contention

- All workers access the same shared policy simultaneously
- PyTorch shared memory causes severe contention with multiple readers
- Each worker blocks waiting for memory access

### Problem 2: Expensive Parameter Synchronization

- Workers need to sync policy parameters before each rollout
- `parameter.data.copy_()` operations are expensive
- Synchronization takes ~12-14 seconds per iteration

### Problem 3: Queue Communication Overhead

- Data transfer through multiprocessing queues is slow
- Large numpy arrays (observations, actions, values) take time to serialize/deserialize

## What Was Tried

1. ✅ **Fixed environment reset bug** - Workers were resetting envs every rollout
2. ✅ **Added timing diagnostics** - Identified the bottleneck (parameter sync)
3. ❌ **Shared policy directly** - Severe contention
4. ❌ **Local policy copies with sync** - Sync too expensive
5. ❌ **Parameter-level copying** - Still too slow

## What Needs to Be Done

### Short-term Fixes (to make it usable)

1. **Use state_dict with CPU tensors**
   - Pre-allocate buffers
   - Use shared memory more efficiently
   - Avoid repeated allocations

2. **Reduce synchronization frequency**
   - Sync every N rollouts instead of every rollout
   - Accept slight staleness in worker policies

3. **Optimize data transfer**
   - Use shared memory for observations/actions
   - Avoid queue serialization overhead
   - Pre-allocate shared buffers

### Long-term Solution (proper implementation)

**Switch to a different architecture:**

1. **IMPALA-style** (recommended for CPU clusters):
   - Workers collect data independently
   - Central learner processes batches asynchronously
   - No synchronization needed during collection
   - Natural fit for distributed CPU training

2. **Async A3C**:
   - Each worker updates independently
   - Lock-free parameter updates
   - Better for CPU parallelism

3. **Ray/RLlib**:
   - Use a proven distributed RL framework
   - Handles all the complexity
   - Optimized for both CPU and GPU

## Recommendation

**For now: Use sequential training**

- It's fast, stable, and well-tested
- ~1,500-2,000 FPS is good for CPU training
- Focus on getting results first

**For future: Implement IMPALA**

- Better architecture for CPU clusters
- Natural fit for distributed training
- Will provide real speedup (3-5x expected)

## Files Involved

- `src/parallel_trainer.py` - Current parallel implementation (broken)
- `train_ppo_parallel.py` - Entry point for parallel training
- `train_parallel_timed.py` - Timed parallel training
- `config/ppo_parallel.yaml` - Parallel configuration

## Testing Commands

If you want to test the current parallel implementation (not recommended):

```bash
# Quick test (will be slow)
python train_parallel_timed.py \
  --duration 0.01 \
  --output-dir ./benchmarks/test_parallel \
  --num-workers 2 \
  --num-envs-per-worker 4
```

You'll see it takes ~14 seconds per iteration vs ~0.3 seconds for sequential.

## Next Steps

1. ✅ Document the issue (this file)
2. ⏸️ Pause parallel development
3. ✅ Focus on sequential training (works great!)
4. 🔮 Future: Implement IMPALA or use Ray/RLlib

---

**Bottom line:** Stick with sequential training for now. It's fast enough for single-node experiments, and the parallel implementation needs a redesign to be useful.
