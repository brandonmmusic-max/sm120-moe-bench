# Sprint 13: NCCL AllReduce p99 Tail Latency — Root Cause & Fix

**Date:** 2026-03-26
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (PCIe Gen5, NO NVLink)

---

## Root Cause: GPU Context-Switching from Background Processes

The p99 AllReduce spikes were caused by **a stale Python process running at 100% SM utilization on GPU0**, forcing the GPU hardware scheduler to context-switch between the stale process and NCCL kernels. Since AllReduce requires ALL GPUs to synchronize, one blocked GPU stalls the entire collective.

### Spike Characteristics (Before Fix)
- **Magnitude:** Exactly ~2330μs (±10μs) — the fixed cost of a GPU context switch
- **Frequency:** ~0.7% of calls (71/10000)
- **Periodicity:** Every ~139 iterations (4.3ms period) — matches GPU time-slicing quantum
- **Uniform:** Zero variance in spike magnitude → hardware-level scheduling, not software

### Why NCCL Tuning Had Zero Effect

We tested **13 different NCCL configurations** (algorithms, protocols, buffer sizes, thread counts, channels, transport paths). Every single one showed identical spikes:

| NCCL Config | p99 (8KB) | Spike Rate | Notes |
|---|---|---|---|
| Default (baseline) | 2330μs | 0.71% | — |
| ALGO=Ring | 2309μs | 0.69% | No change |
| ALGO=Tree | 2336μs | 0.80% | Slightly worse |
| PROTO=Simple | 2337μs | 1.54% | Higher p50 → more spikes hit |
| PROTO=LL | 2327μs | 0.71% | No change |
| PROTO=LL128 | 2302μs | 1.22% | Higher p50 → more spikes hit |
| BUFFSIZE=4MB | 2327μs | 0.71% | No change |
| NTHREADS=64 | 2330μs | 0.68% | No change |
| NTHREADS=256 | 2327μs | 0.68% | No change |
| CHANNELS=1 | 2328μs | 0.70% | No change |
| CHANNELS=4 | 2327μs | 0.68% | No change |
| P2P_DISABLE=1 | 2326μs | 0.71% | No change |
| SHM_DISABLE=1 | 2378μs | 3.57% | Much worse (bad transport) |

**Conclusion:** The spike is external to NCCL — it's a GPU hardware scheduling event.

---

## Fix Applied

### Primary Fix: Kill stale GPU processes before inference
```bash
# Kill any non-inference processes on compute GPUs
# Check for orphan processes:
nvidia-smi pmon -c 1

# Kill stale processes on GPU 0 (or whichever GPUs are used for inference):
# Look for processes with 100% SM that aren't the inference server
kill -9 <stale_pid>
```

### Results After Fix

| Metric (8KB AllReduce) | Before | After | Improvement |
|---|---|---|---|
| **p50** | 13.7μs | 16.1μs | ~same |
| **p95** | 15.7μs | 18.1μs | ~same |
| **p99** | **2330μs** | **22μs** | **106x** |
| **p999** | 2330μs | 127μs | 18x |
| **max** | 2503μs | 7298μs | worse (rare OS jitter) |
| **Spike rate (>10x median)** | **0.71%** | **0.08%** | **9x fewer** |
| **p99/p50 ratio** | **170x** | **1.3x** | Essentially flat |

---

## GPU Isolation Analysis

| GPU Config | p99 (8KB) | Spike Rate | Root Cause |
|---|---|---|---|
| All 4 (with stale proc) | 2330μs | 0.71% | Stale Python at 100% SM on GPU0 |
| GPUs 0,1 (both graphics) | 2285μs | 0.42% | Same cause |
| GPUs 1,2,3 (no GPU0) | **29.8μs** | **0.03%** | Confirms GPU0 was the problem |
| GPUs 2,3 (no graphics) | **18.0μs** | **0.06%** | Cleanest config |
| All 4 (stale killed) | **22μs** | **0.08%** | Fixed |
| All 4 (Sunshine running, no stale) | **22μs** | **0.09%** | Sunshine idle is fine |

### Key Insight
- **Sunshine at idle (0% SM):** Negligible impact on AllReduce latency
- **Stale Python at 100% SM:** Catastrophic — forces periodic GPU context switches
- **COSMIC compositor on GPU1:** Negligible impact (it's Graphics-only, not competing for SM)

---

## OS-Level Tuning (Marginal Impact)

Tested but showed minimal additional improvement:

| OS Tuning | Effect |
|---|---|
| CPU C2 state disabled | No measurable change |
| ASPM policy=performance | No measurable change |
| GPU clocks locked (2800-3090MHz) | No change on spikes |
| CPU governor=performance | Already set (good) |

These were already reasonable. The residual spikes (0.08%, max ~3-7ms) are from:
1. Rare OS scheduler interference (kernel interrupts, RCU callbacks)
2. GPU1's compositor occasionally contending on PCIe root complex
3. PCIe bus contention from other devices

---

## Estimated Impact on vLLM Serving

### Before Fix
- 120 AllReduces per token × 0.71% spike rate = **0.85 spikes per token on average**
- Each spike costs ~2.3ms
- **Average spike overhead per token: ~2.0ms**
- At decode throughput of ~7.4ms/token (135 tok/s), this is a **27% overhead**
- Explains the low-end outlier runs (100-120 tok/s)

### After Fix
- 120 AllReduces per token × 0.08% spike rate = **0.096 spikes per token**
- ~90% of tokens see ZERO spikes
- Spike overhead when one occurs: ~0.1-3ms (variable, not fixed 2.3ms)
- **Average spike overhead per token: <0.1ms**
- **Throughput recovery: ~2ms/token → expect +20-25 tok/s at the tail**

---

## Production Recommendations

### Before Starting vLLM Inference
```bash
# 1. Kill stale GPU processes
for gpu_id in 0 1 2 3; do
    nvidia-smi -i $gpu_id --query-compute-apps=pid --format=csv,noheader | \
        xargs -r kill -9 2>/dev/null
done

# 2. Ensure persistence mode is on (reduces CUDA init latency)
sudo nvidia-smi -pm 1

# 3. Optional: Set exclusive compute mode to prevent accidental sharing
# sudo nvidia-smi -c EXCLUSIVE_PROCESS  # Requires no other processes on GPU

# 4. Optional: Lock GPU clocks for consistency (not required for p99 fix)
# sudo nvidia-smi -lgc 2800,3090
```

### For Best Results (Minimal Residual Jitter)
```bash
# Add to /etc/default/grub GRUB_CMDLINE_LINUX:
# nohz_full=24-47 rcu_nocbs=24-47 isolcpus=24-47
# Then pin NCCL proxy threads to isolated cores

# Set ASPM to performance
echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy

# Disable deeper C-states
for state in /sys/devices/system/cpu/cpu*/cpuidle/state[2-9]/disable; do
    echo 1 | sudo tee "$state" > /dev/null 2>&1
done
```

### NCCL Env Vars (Keep Defaults)
No NCCL environment variable tuning is needed. The default LL protocol and auto-selected algorithm are optimal for these small tensors. Specifically, do NOT set:
- `NCCL_PROTO=Simple` — doubles p50 and doubles spike rate
- `NCCL_SHM_DISABLE=1` — 5x higher p50, 5x more spikes
- `NCCL_ALGO=Tree` — slightly worse than default

---

## Web Research Summary

### Known Issue?
Yes — [NVIDIA/nccl#361](https://github.com/NVIDIA/nccl/issues/361) documents identical jitter patterns (200ms+ spikes on otherwise <1ms AllReduce). The cause is external to NCCL — OS scheduling, GPU context switching, or PCIe contention.

### NCCL 2.27 Improvements
Our system already has NCCL 2.27.5 with symmetric memory support. The SymmMem optimizations primarily benefit NVLink systems. On PCIe, the LL protocol is already optimal for small tensors (~14μs for 8KB).

### vLLM-Specific
vLLM's custom AllReduce is [disabled for >2 PCIe-only GPUs](https://github.com/vllm-project/vllm/issues/13719). With 4 PCIe GPUs, vLLM falls back to PyNCCL, which is what we benchmarked. The Torch SymmMem two_shot path hangs on SM 12.0 (Blackwell) per Sprint 11 findings.
