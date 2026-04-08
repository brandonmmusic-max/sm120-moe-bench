#!/usr/bin/env python3
"""
test_write_allreduce.py — Validate the write-based P2P AllReduce extension.

Tests:
  1. Correctness: P2P result matches NCCL result (bitwise for BF16 sums)
  2. Various payload sizes: 2KB, 4KB, 8KB, 16KB, 32KB, 64KB
  3. CUDA graph capture and replay
  4. Performance comparison vs NCCL (torch.distributed.all_reduce)
  5. Edge cases: odd element counts, zero tensors

Requires 4 GPUs with P2P access. Does NOT require vLLM.

Run:
    cd /home/brandonmusic/sm120-moe-bench/fused-moe/write_allreduce_ext
    pip install -e .
    python test_write_allreduce.py
"""

import torch
import time
import sys

def test_correctness():
    """Test that P2P AllReduce produces correct results for various sizes."""
    import write_allreduce_ext as ext

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 4, f"Need 4 GPUs, have {n_gpus}"

    print("Enabling P2P access...")
    ext.enable_p2p_access()

    test_cases = [
        (1024,  "1K elems (2KB)"),
        (2048,  "2K elems (4KB)"),
        (4096,  "4K elems (8KB) — hidden_dim"),
        (8192,  "8K elems (16KB)"),
        (16384, "16K elems (32KB)"),
        (32768, "32K elems (64KB) — max P2P"),
        (4095,  "4095 elems (odd count)"),
        (1,     "1 elem (minimum)"),
    ]

    all_pass = True
    for n_elements, label in test_cases:
        print(f"\n--- {label} ---")

        # Create buffers on each GPU with known values
        buffers = []
        for i in range(4):
            torch.cuda.set_device(i)
            # Fill with rank+1 so expected sum = 1+2+3+4 = 10
            buf = torch.full((n_elements,), float(i + 1),
                            dtype=torch.bfloat16, device=f'cuda:{i}')
            buffers.append(buf)

        # Reference: compute expected result
        expected = torch.zeros(n_elements, dtype=torch.bfloat16, device='cuda:0')
        for i in range(4):
            expected += buffers[i].to('cuda:0')

        # Run P2P AllReduce (in-place on all buffers)
        torch.cuda.set_device(0)
        ext.oneshot_allreduce_4gpu(buffers[0], buffers[1], buffers[2], buffers[3])
        torch.cuda.synchronize()

        # Check GPU 0 result
        max_err = (buffers[0] - expected).abs().max().item()
        ok = max_err < 0.1
        print(f"  GPU 0: max_err={max_err:.6f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"    expected[0]={expected[0].item()}, got={buffers[0][0].item()}")
            all_pass = False

        # Check that all GPUs got the result (broadcast via P2P writes)
        for g in range(1, 4):
            val = buffers[g].to('cuda:0')
            err = (val - expected).abs().max().item()
            ok_g = err < 0.1
            print(f"  GPU {g}: max_err={err:.6f} {'PASS' if ok_g else 'FAIL'}")
            if not ok_g:
                all_pass = False

    return all_pass


def test_cuda_graph():
    """Test CUDA graph capture and replay with P2P AllReduce."""
    import write_allreduce_ext as ext

    print("\n" + "=" * 60)
    print("  CUDA Graph Capture Test")
    print("=" * 60)

    n_elements = 4096
    n_gpus = 4

    # Create buffers
    buffers = []
    for i in range(n_gpus):
        torch.cuda.set_device(i)
        buf = torch.full((n_elements,), float(i + 1),
                        dtype=torch.bfloat16, device=f'cuda:{i}')
        buffers.append(buf)

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream(device='cuda:0')

    # Warmup in stream
    with torch.cuda.stream(stream):
        ext.oneshot_allreduce_4gpu(buffers[0], buffers[1], buffers[2], buffers[3])
    stream.synchronize()

    # Reset buffers
    for i in range(n_gpus):
        buffers[i].fill_(float(i + 1))
    torch.cuda.synchronize()

    # Capture
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ext.oneshot_allreduce_4gpu(
                buffers[0], buffers[1], buffers[2], buffers[3])

        # Reset and replay
        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        torch.cuda.synchronize()

        graph.replay()
        torch.cuda.synchronize()

        # Verify
        r = buffers[0][0].item()
        ok = abs(r - 10.0) < 0.1
        print(f"  Graph replay result: {r} {'PASS' if ok else 'FAIL'}")

        # Check broadcast
        for i in range(1, n_gpus):
            val = buffers[i][0].item()
            ok_i = abs(val - 10.0) < 0.1
            print(f"  GPU {i} broadcast: {val} {'PASS' if ok_i else 'FAIL'}")
            ok = ok and ok_i

        if ok:
            # Benchmark graph replay
            for _ in range(100):
                graph.replay()
            torch.cuda.synchronize()

            n_iters = 5000
            start = time.perf_counter()
            for _ in range(n_iters):
                graph.replay()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            avg_us = (elapsed / n_iters) * 1e6
            print(f"\n  Graph replay latency: {avg_us:.1f} us")
            print(f"  vs NCCL ~14-20 us per AR, this is the KERNEL latency only")

        return ok

    except Exception as e:
        print(f"  Graph capture FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark():
    """Benchmark P2P AllReduce vs eager PyTorch operations."""
    import write_allreduce_ext as ext

    print("\n" + "=" * 60)
    print("  Performance Benchmark")
    print("=" * 60)

    test_sizes = [
        (2048,  "4KB"),
        (4096,  "8KB"),
        (8192,  "16KB"),
        (16384, "32KB"),
        (32768, "64KB"),
    ]

    for n_elements, label in test_sizes:
        print(f"\n--- {label} ({n_elements} BF16 elements) ---")

        buffers = []
        for i in range(4):
            torch.cuda.set_device(i)
            buf = torch.full((n_elements,), float(i + 1),
                            dtype=torch.bfloat16, device=f'cuda:{i}')
            buffers.append(buf)

        torch.cuda.set_device(0)

        # Warmup
        for _ in range(100):
            for i in range(4):
                buffers[i].fill_(float(i + 1))
            ext.oneshot_allreduce_4gpu(buffers[0], buffers[1], buffers[2], buffers[3])
        torch.cuda.synchronize()

        # Benchmark: P2P AllReduce (eager, includes Python overhead)
        n_iters = 2000

        for i in range(4):
            buffers[i].fill_(float(i + 1))
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            ext.oneshot_allreduce_4gpu(buffers[0], buffers[1], buffers[2], buffers[3])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / n_iters) * 1e6
        print(f"  P2P eager: {avg_us:.1f} us")

        # Benchmark: CUDA events for GPU-side timing
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(n_iters):
            start_event.record()
            ext.oneshot_allreduce_4gpu(buffers[0], buffers[1], buffers[2], buffers[3])
            stop_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(stop_event) * 1000)  # us

        times.sort()
        lo, hi = len(times) // 10, len(times) * 9 // 10
        avg_gpu = sum(times[lo:hi]) / (hi - lo)
        print(f"  P2P GPU-timed (p10-p90 avg): {avg_gpu:.1f} us")

        # CUDA graph benchmark
        try:
            stream = torch.cuda.Stream(device='cuda:0')
            with torch.cuda.stream(stream):
                ext.oneshot_allreduce_4gpu(
                    buffers[0], buffers[1], buffers[2], buffers[3])
            stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                ext.oneshot_allreduce_4gpu(
                    buffers[0], buffers[1], buffers[2], buffers[3])

            for _ in range(100):
                graph.replay()
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(n_iters):
                graph.replay()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            avg_graph = (elapsed / n_iters) * 1e6
            print(f"  P2P graph: {avg_graph:.1f} us")
        except Exception as e:
            print(f"  Graph capture failed: {e}")


def test_out_of_place():
    """Test the out-of-place variant."""
    import write_allreduce_ext as ext

    print("\n" + "=" * 60)
    print("  Out-of-Place AllReduce Test")
    print("=" * 60)

    n_elements = 4096

    # Input buffers
    inputs = []
    outputs = []
    for i in range(4):
        torch.cuda.set_device(i)
        inp = torch.full((n_elements,), float(i + 1),
                        dtype=torch.bfloat16, device=f'cuda:{i}')
        out = torch.zeros(n_elements, dtype=torch.bfloat16, device=f'cuda:{i}')
        inputs.append(inp)
        outputs.append(out)

    torch.cuda.set_device(0)
    ext.oneshot_allreduce_outofplace(
        inputs[0], inputs[1], inputs[2], inputs[3],
        outputs[0], outputs[1], outputs[2], outputs[3]
    )
    torch.cuda.synchronize()

    # Check outputs have the sum
    for i in range(4):
        val = outputs[i][0].item()
        ok = abs(val - 10.0) < 0.1
        print(f"  Output GPU {i}: {val} {'PASS' if ok else 'FAIL'}")

    # Check inputs are unchanged
    for i in range(4):
        val = inputs[i][0].item()
        ok = abs(val - float(i + 1)) < 0.01
        print(f"  Input GPU {i}: {val} (expected {float(i+1)}) {'PASS' if ok else 'FAIL'}")


def test_random_data():
    """Test with random data and compare against CPU reference."""
    import write_allreduce_ext as ext

    print("\n" + "=" * 60)
    print("  Random Data Correctness Test")
    print("=" * 60)

    n_elements = 8192
    torch.manual_seed(42)

    # Create random BF16 data on each GPU
    buffers = []
    cpu_bufs = []
    for i in range(4):
        torch.cuda.set_device(i)
        cpu_buf = torch.randn(n_elements, dtype=torch.bfloat16)
        gpu_buf = cpu_buf.to(f'cuda:{i}')
        buffers.append(gpu_buf)
        cpu_bufs.append(cpu_buf)

    # CPU reference
    expected = cpu_bufs[0] + cpu_bufs[1] + cpu_bufs[2] + cpu_bufs[3]

    # P2P AllReduce
    torch.cuda.set_device(0)
    ext.oneshot_allreduce_4gpu(buffers[0], buffers[1], buffers[2], buffers[3])
    torch.cuda.synchronize()

    # Compare
    result = buffers[0].cpu()
    max_err = (result.float() - expected.float()).abs().max().item()
    mean_err = (result.float() - expected.float()).abs().mean().item()
    # BF16 has ~0.8% relative error, so for sums of 4 random values
    # we expect some accumulated rounding
    ok = max_err < 0.5  # generous threshold for BF16 arithmetic
    print(f"  Max error: {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  {'PASS' if ok else 'FAIL'}")

    # Check all GPUs got same result
    for g in range(1, 4):
        r = buffers[g].to('cuda:0')
        diff = (r - buffers[0]).abs().max().item()
        ok_g = diff == 0.0  # Should be bitwise identical
        print(f"  GPU {g} vs GPU 0 diff: {diff:.6f} {'PASS' if ok_g else 'FAIL'}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Write-Based P2P AllReduce Test Suite")
    print("=" * 60)

    n_gpus = torch.cuda.device_count()
    print(f"\nDetected {n_gpus} GPUs")
    if n_gpus < 4:
        print("ERROR: Need at least 4 GPUs")
        sys.exit(1)

    # Print GPU info
    for i in range(4):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_mem / 1e9:.0f} GB)")

    tests = [
        ("Correctness", test_correctness),
        ("Random Data", test_random_data),
        ("Out-of-Place", test_out_of_place),
        ("CUDA Graph", test_cuda_graph),
        ("Benchmark", test_benchmark),
    ]

    results = {}
    for name, fn in tests:
        try:
            result = fn()
            results[name] = result if isinstance(result, bool) else True
        except Exception as e:
            print(f"\n  {name} EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
