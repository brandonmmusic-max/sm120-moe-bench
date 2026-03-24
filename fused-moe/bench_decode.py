#!/usr/bin/env python3
"""
Benchmark decode throughput against vLLM server.
Measures single-user decode tok/s with warmup.
"""
import json
import time
import sys
import urllib.request
import urllib.error

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9200"
MODEL = "qwen3.5-397b-nvfp4"
NUM_WARMUP = 3
NUM_RUNS = 5
OUTPUT_TOKENS = 512

PROMPT = """You are a Kentucky appellate attorney. Draft a brief analysis of whether
the doctrine of qualified immunity applies when a police officer conducts a
warrantless search of a vehicle during a routine traffic stop in Kentucky,
citing relevant Sixth Circuit and Kentucky Supreme Court precedent."""


def stream_completion(prompt: str, max_tokens: int) -> tuple[int, float]:
    """Send a streaming chat completion and count tokens/time."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    token_count = 0
    start = None

    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if start is None:
                        start = time.perf_counter()
                    token_count += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    elapsed = time.perf_counter() - start if start else 0.001
    return token_count, elapsed


def check_server():
    """Verify server is running."""
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"Server OK. Models: {models}")
            return True
    except Exception as e:
        print(f"Server not ready: {e}")
        return False


def main():
    if not check_server():
        print("Exiting.")
        sys.exit(1)

    print(f"\nBenchmark: {OUTPUT_TOKENS} output tokens, single user")
    print(f"Warmup: {NUM_WARMUP} runs, Benchmark: {NUM_RUNS} runs\n")

    # Warmup
    print("Warming up...")
    for i in range(NUM_WARMUP):
        tokens, elapsed = stream_completion(PROMPT, 256)
        tps = tokens / elapsed if elapsed > 0 else 0
        print(f"  Warmup {i+1}: {tps:.1f} tok/s ({tokens} tokens)")

    # Benchmark
    print(f"\nBenchmark ({NUM_RUNS} runs, {OUTPUT_TOKENS} tokens):")
    print(f" {'Run':>4} | {'Tok/s':>8} | {'Tokens':>7} | {'Time(s)':>8}")
    print("-" * 40)

    results = []
    for i in range(NUM_RUNS):
        tokens, elapsed = stream_completion(PROMPT, OUTPUT_TOKENS)
        tps = tokens / elapsed if elapsed > 0 else 0
        results.append(tps)
        print(f" {i+1:>4} | {tps:>8.1f} | {tokens:>7} | {elapsed:>8.2f}")

    print("-" * 40)
    avg = sum(results) / len(results)
    med = sorted(results)[len(results) // 2]
    print(f" {'Avg':>4} | {avg:>8.1f}")
    print(f" {'Med':>4} | {med:>8.1f}")
    print(f"\nBaseline (TP=4, no EP): ~172 tok/s")
    print(f"Target (EP=4):          ~160+ tok/s")


if __name__ == "__main__":
    main()
