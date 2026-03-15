"""
SM120 Multi-GPU 2D-Parallel Flash Attention

2D decomposition: 2-way Q-parallel × 2-way KV-parallel across 4 GPUs.
Each GPU computes 25% of the work. Only 2-way combine needed (not 4-way).

GPU 0: Q[0:Sq/2]  × KV[0:Skv/2]    GPU 1: Q[0:Sq/2]  × KV[Skv/2:Skv]
GPU 2: Q[Sq/2:Sq] × KV[0:Skv/2]    GPU 3: Q[Sq/2:Sq] × KV[Skv/2:Skv]

Combine: GPU 0+1 merge Q top half, GPU 2+3 merge Q bottom half.
"""

import torch
import sm120_flash_attn


class MultiGPU2DAttention:
    def __init__(self, num_gpus, B, Hq, Hkv, Sq, D, combine_gpus=(0, 2)):
        assert num_gpus == 4
        self.B, self.Hq, self.Sq, self.D = B, Hq, Sq, D
        self.Sq_half = (Sq + 1) // 2
        self.combine_gpus = combine_gpus  # GPU 0 combines top half, GPU 2 combines bottom

        self.streams = []
        self.done_events = []
        for g in range(4):
            torch.cuda.set_device(g)
            self.streams.append(torch.cuda.Stream(device=g))
            self.done_events.append(torch.cuda.Event())

        # Pre-allocate partial outputs on each GPU
        self.partial_O = []
        self.partial_lse = []
        for g in range(4):
            torch.cuda.set_device(g)
            sq_g = self.Sq_half if g < 2 else Sq - self.Sq_half
            self.partial_O.append(torch.empty(B, Hq, sq_g, D, dtype=torch.bfloat16, device=f'cuda:{g}'))
            self.partial_lse.append(torch.empty(B, Hq, sq_g, dtype=torch.float32, device=f'cuda:{g}'))

    def forward(self, Q_halves, K_halves, V_halves):
        """
        Q_halves[g]: Q for this GPU's Q partition, on device g
        K_halves[g]: K chunk for this GPU's KV partition, on device g
        V_halves[g]: V chunk, on device g
        """
        # Phase 1: All 4 GPUs compute in parallel
        for g in range(4):
            torch.cuda.set_device(g)
            with torch.cuda.stream(self.streams[g]):
                results = sm120_flash_attn.forward(Q_halves[g], K_halves[g], V_halves[g], True)
                self.partial_O[g].copy_(results[0])
                self.partial_lse[g].copy_(results[1])
                self.done_events[g].record(self.streams[g])

        # Phase 2: 2-way combine (GPU 0+1 for top Q half, GPU 2+3 for bottom)
        # Combine pair (0,1) on GPU 0
        torch.cuda.set_device(0)
        with torch.cuda.stream(self.streams[0]):
            self.streams[0].wait_event(self.done_events[1])
            O1_on_0 = self.partial_O[1].to('cuda:0')
            lse1_on_0 = self.partial_lse[1].to('cuda:0')
            # Online softmax merge (2-way)
            m = torch.maximum(self.partial_lse[0], lse1_on_0)
            w0 = torch.exp(self.partial_lse[0] - m)
            w1 = torch.exp(lse1_on_0 - m)
            total_w = w0 + w1
            O_top = ((w0.unsqueeze(-1) * self.partial_O[0].float() +
                       w1.unsqueeze(-1) * O1_on_0.float()) /
                      total_w.unsqueeze(-1)).to(torch.bfloat16)

        # Combine pair (2,3) on GPU 2
        torch.cuda.set_device(2)
        with torch.cuda.stream(self.streams[2]):
            self.streams[2].wait_event(self.done_events[3])
            O3_on_2 = self.partial_O[3].to('cuda:2')
            lse3_on_2 = self.partial_lse[3].to('cuda:2')
            m = torch.maximum(self.partial_lse[2], lse3_on_2)
            w0 = torch.exp(self.partial_lse[2] - m)
            w1 = torch.exp(lse3_on_2 - m)
            total_w = w0 + w1
            O_bot = ((w0.unsqueeze(-1) * self.partial_O[2].float() +
                       w1.unsqueeze(-1) * O3_on_2.float()) /
                      total_w.unsqueeze(-1)).to(torch.bfloat16)

        # Phase 3: Gather final output on GPU 0
        torch.cuda.set_device(0)
        self.streams[0].synchronize()
        self.streams[2].synchronize()
        O_bot_on_0 = O_bot.to('cuda:0')
        return torch.cat([O_top, O_bot_on_0], dim=2)


def benchmark(B=1, Hq=32, Hkv=8, Sq=8192, Skv=8192, D=128,
              warmup=15, iters=50):
    num_gpus = 4
    Sq_half = (Sq + 1) // 2
    Skv_half = (Skv + 1) // 2

    # Create data on GPU 0
    torch.cuda.set_device(0)
    Q = torch.randn(B, Hq, Sq, D, dtype=torch.bfloat16, device='cuda:0')
    K = torch.randn(B, Hkv, Skv, D, dtype=torch.bfloat16, device='cuda:0')
    V = torch.randn(B, Hkv, Skv, D, dtype=torch.bfloat16, device='cuda:0')

    # Distribute: 2D split
    Q_top = Q[:, :, :Sq_half, :]
    Q_bot = Q[:, :, Sq_half:, :]
    K_left = K[:, :, :Skv_half, :]
    K_right = K[:, :, Skv_half:, :]
    V_left = V[:, :, :Skv_half, :]
    V_right = V[:, :, Skv_half:, :]

    Q_halves = [
        Q_top.to('cuda:0').contiguous(), Q_top.to('cuda:1').contiguous(),
        Q_bot.to('cuda:2').contiguous(), Q_bot.to('cuda:3').contiguous(),
    ]
    K_halves = [
        K_left.to('cuda:0').contiguous(), K_right.to('cuda:1').contiguous(),
        K_left.to('cuda:2').contiguous(), K_right.to('cuda:3').contiguous(),
    ]
    V_halves = [
        V_left.to('cuda:0').contiguous(), V_right.to('cuda:1').contiguous(),
        V_left.to('cuda:2').contiguous(), V_right.to('cuda:3').contiguous(),
    ]
    for g in range(4):
        torch.cuda.synchronize(g)

    ctx = MultiGPU2DAttention(4, B, Hq, Hkv, Sq, D)

    # Correctness
    Ke = K.repeat_interleave(Hq // Hkv, dim=1)
    Ve = V.repeat_interleave(Hq // Hkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, Ke, Ve, is_causal=False)
    out = ctx.forward(Q_halves, K_halves, V_halves)
    diff = (out.to('cuda:0') - ref).abs().max().item()
    print(f'  Correctness: maxdiff={diff:.6f} {"PASS" if diff < 0.1 else "FAIL"}')

    # Benchmark single GPU
    torch.cuda.set_device(0)
    for _ in range(warmup):
        sm120_flash_attn.forward(Q, K, V)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        sm120_flash_attn.forward(Q, K, V)
    e.record()
    torch.cuda.synchronize()
    ms_single = s.elapsed_time(e) / iters
    tf_single = 4 * B * Hq * Sq * Skv * D / ms_single / 1e9

    # Benchmark cuDNN
    for _ in range(warmup):
        torch.nn.functional.scaled_dot_product_attention(Q, Ke, Ve)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        torch.nn.functional.scaled_dot_product_attention(Q, Ke, Ve)
    e.record()
    torch.cuda.synchronize()
    ms_sdpa = s.elapsed_time(e) / iters
    tf_sdpa = 4 * B * Hq * Sq * Skv * D / ms_sdpa / 1e9

    # Benchmark 2D multi-GPU
    for _ in range(warmup):
        ctx.forward(Q_halves, K_halves, V_halves)
    torch.cuda.set_device(0)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        ctx.forward(Q_halves, K_halves, V_halves)
    e.record()
    for g in range(4):
        torch.cuda.synchronize(g)
    ms_2d = s.elapsed_time(e) / iters
    tf_2d = 4 * B * Hq * Sq * Skv * D / ms_2d / 1e9

    print(f'  Single GPU:  {ms_single:>8.3f}ms = {tf_single:>7.1f} TF')
    print(f'  cuDNN SDPA:  {ms_sdpa:>8.3f}ms = {tf_sdpa:>7.1f} TF')
    print(f'  2D 4-GPU:    {ms_2d:>8.3f}ms = {tf_2d:>7.1f} TF (eff)')
    print(f'  Scaling: {ms_single/ms_2d:.2f}x single, {ms_sdpa/ms_2d:.2f}x cuDNN')


if __name__ == '__main__':
    # Switch back to BF16 v4 kernel
    print('=' * 80)
    print('SM120 2D-Parallel Attention (2Q × 2KV across 4 GPUs)')
    print('=' * 80)

    configs = [
        (1, 32, 8, 2048, 2048),
        (1, 32, 8, 4096, 4096),
        (1, 32, 8, 8192, 8192),
        (1, 32, 8, 16384, 16384),
        (1, 32, 8, 32768, 32768),
        (1, 32, 8, 65536, 65536),
        (1, 32, 8, 131072, 131072),
    ]

    for B, Hq, Hkv, Sq, Skv in configs:
        print(f'\nSq={Sq}:')
        try:
            benchmark(B, Hq, Hkv, Sq, Skv, warmup=10, iters=30)
        except Exception as ex:
            print(f'  ERROR: {ex}')
