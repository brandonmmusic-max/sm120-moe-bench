#!/usr/bin/env python3
"""
Decode benchmark for VLLM_CUTLASS backend comparison.
Measures single-user decode tok/s.
"""
import requests
import time
import json
import sys

BASE = "http://localhost:9200/v1"
MODEL = "qwen3.5-397b-nvfp4"

def decode_benchmark(prompt="Write a detailed analysis of", max_tokens=512, temperature=0.7):
    start = time.perf_counter()
    resp = requests.post(f"{BASE}/completions", json={
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    })
    elapsed = time.perf_counter() - start
    data = resp.json()
    tokens = data["usage"]["completion_tokens"]
    return tokens, elapsed

def main():
    # Extended warmup
    print("Extended warmup (8 runs, 256 tokens)...")
    for i in range(8):
        tokens, elapsed = decode_benchmark(max_tokens=256)
        tps = tokens / elapsed
        print(f"  Warmup {i+1}: {tps:.1f} tok/s")

    # Steady-state benchmark
    n_runs = 10
    print(f"\nSteady-state benchmark ({n_runs} runs, 512 tokens):")
    print(f" Run |   Tok/s")
    print("-" * 20)

    tps_list = []
    for i in range(n_runs):
        tokens, elapsed = decode_benchmark(max_tokens=512)
        tps = tokens / elapsed
        tps_list.append(tps)
        print(f"  {i+1:2d} |   {tps:.1f}")

    avg = sum(tps_list) / len(tps_list)
    sorted_tps = sorted(tps_list)
    med = sorted_tps[len(sorted_tps)//2]
    mn = sorted_tps[0]
    mx = sorted_tps[-1]
    print("-" * 20)
    print(f" Avg |   {avg:.1f}")
    print(f" Med |   {med:.1f}")
    print(f" Min |   {mn:.1f}")
    print(f" Max |   {mx:.1f}")
    print(f"\nBackend: VERDICT_MOE (VLLM_USE_VERDICT_MOE=1)")
    print(f"Prior VLLM_CUTLASS EP=4 MTP=3 baseline: ~129 tok/s median")
    print(f"Speedup vs VLLM_CUTLASS: {med/129:.2f}x (median)")

if __name__ == "__main__":
    main()
