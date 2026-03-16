"""
Full FP8 V^T Flash Attention benchmark: single-GPU vs SDPA + multi-GPU 2D.
Tests at seq lengths: 1024, 2048, 16384, 32768, 65536
"""

import torch
import torch.nn.functional as F
import time
import os
import sys

from torch.utils.cpp_extension import load

def build_fp8():
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csrc')
    return load(
        name="sm120_fa_fp8_vtrans",
        sources=[
            os.path.join(src_dir, 'fp8_vtrans_binding.cpp'),
            os.path.join(src_dir, 'sm120_flash_attn_fp8_vtrans.cu'),
        ],
        extra_cuda_cflags=['-O3', '-std=c++17', '-gencode=arch=compute_120a,code=sm_120a', '--use_fast_math', '-lineinfo'],
        extra_ldflags=['-lcuda'],
        verbose=True,
    )


def benchmark_one(func, warmup=10, iters=30):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def tflops(B, Hq, Sq, Skv, D, ms):
    return 4.0 * B * Hq * Sq * Skv * D / (ms * 1e-3) / 1e12


class MultiGPU2D:
    def __init__(self, mod, B, Hq, Hkv, Sq, D):
        self.mod = mod
        self.B, self.Hq, self.Sq, self.D = B, Hq, Sq, D
        self.Sq_half = (Sq + 1) // 2
        self.streams = []
        self.events = []
        for g in range(4):
            torch.cuda.set_device(g)
            self.streams.append(torch.cuda.Stream(device=g))
            self.events.append(torch.cuda.Event())
        self.partial_O = []
        self.partial_lse = []
        for g in range(4):
            torch.cuda.set_device(g)
            sq_g = self.Sq_half if g < 2 else Sq - self.Sq_half
            self.partial_O.append(torch.empty(B, Hq, sq_g, D, dtype=torch.bfloat16, device=f'cuda:{g}'))
            self.partial_lse.append(torch.empty(B, Hq, sq_g, dtype=torch.float32, device=f'cuda:{g}'))

    def forward(self, Q_h, K_h, V_h):
        for g in range(4):
            torch.cuda.set_device(g)
            with torch.cuda.stream(self.streams[g]):
                results = self.mod.forward(Q_h[g], K_h[g], V_h[g], True)
                self.partial_O[g].copy_(results[0])
                self.partial_lse[g].copy_(results[1])
                self.events[g].record(self.streams[g])

        # Combine (0,1) on GPU 0
        torch.cuda.set_device(0)
        with torch.cuda.stream(self.streams[0]):
            self.streams[0].wait_event(self.events[1])
            O1 = self.partial_O[1].to('cuda:0')
            lse1 = self.partial_lse[1].to('cuda:0')
            m = torch.maximum(self.partial_lse[0], lse1)
            w0 = torch.exp(self.partial_lse[0] - m)
            w1 = torch.exp(lse1 - m)
            tw = w0 + w1
            O_top = ((w0.unsqueeze(-1) * self.partial_O[0].float() +
                       w1.unsqueeze(-1) * O1.float()) / tw.unsqueeze(-1)).to(torch.bfloat16)

        # Combine (2,3) on GPU 2
        torch.cuda.set_device(2)
        with torch.cuda.stream(self.streams[2]):
            self.streams[2].wait_event(self.events[3])
            O3 = self.partial_O[3].to('cuda:2')
            lse3 = self.partial_lse[3].to('cuda:2')
            m = torch.maximum(self.partial_lse[2], lse3)
            w0 = torch.exp(self.partial_lse[2] - m)
            w1 = torch.exp(lse3 - m)
            tw = w0 + w1
            O_bot = ((w0.unsqueeze(-1) * self.partial_O[2].float() +
                       w1.unsqueeze(-1) * O3.float()) / tw.unsqueeze(-1)).to(torch.bfloat16)

        torch.cuda.set_device(0)
        self.streams[0].synchronize()
        self.streams[2].synchronize()
        return torch.cat([O_top, O_bot.to('cuda:0')], dim=2)


def main():
    print("Building FP8 V^T kernel...")
    mod = build_fp8()

    B, Hq, Hkv, D = 1, 32, 8, 128
    kv_repeat = Hq // Hkv
    device = "cuda:0"
    dtype = torch.bfloat16
    num_gpus = torch.cuda.device_count()

    # ===================== SINGLE GPU =====================
    seq_lengths = [1024, 2048, 16384, 32768, 65536]

    print("\n" + "=" * 80)
    print("SINGLE GPU: FP8 V^T vs cuDNN SDPA")
    print("=" * 80)
    print(f"  B={B}, Hq={Hq}, Hkv={Hkv}, D={D}")
    print(f"{'Seq':>8s} | {'FP8 VT':>10s} {'TF':>7s} | {'SDPA':>10s} {'TF':>7s} | {'Speedup':>8s}")
    print("-" * 65)

    for seq in seq_lengths:
        torch.cuda.set_device(0)
        try:
            Q = torch.randn(B, Hq, seq, D, device=device, dtype=dtype)
            K = torch.randn(B, Hkv, seq, D, device=device, dtype=dtype)
            V = torch.randn(B, Hkv, seq, D, device=device, dtype=dtype)
            Ke = K.repeat_interleave(kv_repeat, dim=1)
            Ve = V.repeat_interleave(kv_repeat, dim=1)

            t_fp8 = benchmark_one(lambda: mod.forward(Q, K, V, False))
            tf_fp8 = tflops(B, Hq, seq, seq, D, t_fp8)

            t_sdpa = benchmark_one(lambda: F.scaled_dot_product_attention(Q, Ke, Ve))
            tf_sdpa = tflops(B, Hq, seq, seq, D, t_sdpa)

            speedup = t_sdpa / t_fp8
            print(f"{seq:>8d} | {t_fp8:>7.2f} ms {tf_fp8:>6.1f} | {t_sdpa:>7.2f} ms {tf_sdpa:>6.1f} | {speedup:>7.2f}x")

            del Q, K, V, Ke, Ve
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{seq:>8d} | ERROR: {e}")
            torch.cuda.empty_cache()

    # ===================== MULTI-GPU =====================
    if num_gpus < 4:
        print(f"\nSkipping multi-GPU (only {num_gpus} GPUs available, need 4)")
        return

    print("\n" + "=" * 80)
    print("4-GPU 2D-PARALLEL: FP8 V^T (2Q × 2KV)")
    print("=" * 80)
    print(f"  B={B}, Hq={Hq}, Hkv={Hkv}, D={D}")
    print(f"{'Seq':>8s} | {'4-GPU 2D':>10s} {'TF(eff)':>9s} | {'1-GPU FP8':>10s} {'TF':>7s} | {'Scaling':>8s}")
    print("-" * 70)

    for seq in seq_lengths:
        torch.cuda.set_device(0)
        try:
            Sq_half = (seq + 1) // 2
            Skv_half = (seq + 1) // 2

            Q = torch.randn(B, Hq, seq, D, device='cuda:0', dtype=dtype)
            K = torch.randn(B, Hkv, seq, D, device='cuda:0', dtype=dtype)
            V = torch.randn(B, Hkv, seq, D, device='cuda:0', dtype=dtype)

            Q_top = Q[:, :, :Sq_half, :].contiguous()
            Q_bot = Q[:, :, Sq_half:, :].contiguous()
            K_left = K[:, :, :Skv_half, :].contiguous()
            K_right = K[:, :, Skv_half:, :].contiguous()
            V_left = V[:, :, :Skv_half, :].contiguous()
            V_right = V[:, :, Skv_half:, :].contiguous()

            Q_h = [Q_top.to(f'cuda:0'), Q_top.to(f'cuda:1'),
                    Q_bot.to(f'cuda:2'), Q_bot.to(f'cuda:3')]
            K_h = [K_left.to(f'cuda:0'), K_right.to(f'cuda:1'),
                    K_left.to(f'cuda:2'), K_right.to(f'cuda:3')]
            V_h = [V_left.to(f'cuda:0'), V_right.to(f'cuda:1'),
                    V_left.to(f'cuda:2'), V_right.to(f'cuda:3')]
            for g in range(4):
                torch.cuda.synchronize(g)

            ctx = MultiGPU2D(mod, B, Hq, Hkv, seq, D)

            # Multi-GPU benchmark
            t_mg = benchmark_one(lambda: ctx.forward(Q_h, K_h, V_h), warmup=8, iters=20)
            tf_mg = tflops(B, Hq, seq, seq, D, t_mg)

            # Single-GPU benchmark for comparison
            torch.cuda.set_device(0)
            t_1g = benchmark_one(lambda: mod.forward(Q, K, V, False), warmup=8, iters=20)
            tf_1g = tflops(B, Hq, seq, seq, D, t_1g)

            scaling = t_1g / t_mg
            print(f"{seq:>8d} | {t_mg:>7.2f} ms {tf_mg:>8.1f} | {t_1g:>7.2f} ms {tf_1g:>6.1f} | {scaling:>7.2f}x")

            del Q, K, V, Q_top, Q_bot, K_left, K_right, V_left, V_right
            for g in range(4):
                torch.cuda.set_device(g)
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{seq:>8d} | ERROR: {e}")
            for g in range(4):
                torch.cuda.set_device(g)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
