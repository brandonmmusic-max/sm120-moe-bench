#!/usr/bin/env python3
"""
Vast.ai Response Generator for DFlash Training
================================================
Run on a Vast.ai instance with 8x B200 GPUs.

Setup:
  1. Rent 8x B200 on vast.ai
  2. SSH in
  3. pip install vllm datasets aiohttp huggingface_hub
  4. python vastai_generate_responses.py
  5. Download: scp -P <PORT> root@<IP>:/root/dflash_gen/responses.json .

Runs 4 model instances (TP=2 each, 2 GPUs per instance), generates 50K
target-model responses in parallel. Estimated time: ~20-30 min on 8x B200.
"""
import os, json, random, time, subprocess, sys, asyncio

# =============================================================================
# Config
# =============================================================================
MODEL_ID = "lukealonso/qwen3.5-397b-a17b-nvfp4"
NUM_RESPONSES = 50000
MAX_TOKENS = 512
TEMPERATURE = 0.7
CONCURRENT_PER_INSTANCE = 32
OUTPUT_DIR = "/root/dflash_gen"
PORTS = [8000, 8001, 8002, 8003]  # 4 instances, TP=2 each, 2 GPUs per instance

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run(cmd, **kw):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, **kw)

# =============================================================================
# Step 0: Install
# =============================================================================
print("Installing dependencies...")
run("pip install vllm datasets aiohttp huggingface_hub safetensors 2>&1 | tail -3")

# =============================================================================
# Step 1: Download model
# =============================================================================
print(f"\nDownloading {MODEL_ID}...")
from huggingface_hub import snapshot_download
model_path = snapshot_download(MODEL_ID, local_dir="/root/model")
print(f"Model at: {model_path}")

# =============================================================================
# Step 2: Launch 2 vLLM instances (TP=4 each)
# =============================================================================
print("\nLaunching vLLM instances...")
procs = []
for i, port in enumerate(PORTS):
    gpu_start = i * 2
    gpus = ",".join(str(g) for g in range(gpu_start, gpu_start + 2))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", "qwen3.5-397b-nvfp4",
        "--host", "0.0.0.0", "--port", str(port),
        "--trust-remote-code",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.90",
        "--kv-cache-dtype", "fp8_e4m3",
        "--max-model-len", "4096",
        "--max-num-seqs", "64",
        "--max-num-batched-tokens", "32768",
        "--enforce-eager",
    ]
    log = open(f"{OUTPUT_DIR}/server_{i}.log", "w")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    procs.append(p)
    print(f"  Instance {i}: GPUs {gpus}, port {port}, PID {p.pid}")

# =============================================================================
# Step 3: Wait for servers
# =============================================================================
import urllib.request
print("\nWaiting for servers to load...")
for port in PORTS:
    for attempt in range(180):
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            if r.status == 200:
                print(f"  Port {port} READY!")
                break
        except: pass
        time.sleep(10)
        if attempt % 6 == 5:
            print(f"  Port {port} loading... ({(attempt+1)*10}s)")
    else:
        print(f"  WARNING: Port {port} not ready after 30min, continuing anyway")

# =============================================================================
# Step 4: Load prompts
# =============================================================================
from datasets import load_dataset
print("\nLoading prompts...")
prompts = []
try:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    prompts.extend([x["question"] for x in ds])
    print(f"  gsm8k: {len(ds)}")
except: pass
try:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts.extend([x.get("instruction","") + ("\n"+x["input"] if x.get("input") else "") for x in ds])
    print(f"  alpaca: {len(ds)}")
except: pass
try:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    prompts.extend([x.get("question","") for x in ds][:5000])
    print(f"  MMLU-Pro: {min(len(ds),5000)}")
except: pass
try:
    ds = load_dataset("openai/openai_humaneval", split="test")
    prompts.extend([f"Write a solution:\n```python\n{x['prompt']}\n```" for x in ds])
    print(f"  humaneval: {len(ds)}")
except: pass
random.shuffle(prompts)
prompts = prompts[:int(NUM_RESPONSES * 1.3)]
print(f"Total: {len(prompts)} prompts")

# =============================================================================
# Step 5: Generate (async, both instances)
# =============================================================================
import aiohttp

resp_file = f"{OUTPUT_DIR}/responses.json"
existing = json.load(open(resp_file)) if os.path.exists(resp_file) else []
if len(existing) >= NUM_RESPONSES:
    print(f"Already have {len(existing)}, done!"); sys.exit(0)
if existing:
    print(f"Resuming from {len(existing)}")
    prompts = prompts[int(len(existing)/0.9):]

responses = list(existing)
apis = [f"http://localhost:{p}/v1/chat/completions" for p in PORTS]
t0 = time.time()

async def gen_one(session, prompt, sem, idx):
    api = apis[idx % len(apis)]
    async with sem:
        try:
            async with session.post(api, json={
                "model": "qwen3.5-397b-nvfp4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"].get("content") or ""
                    if len(content) > 20:
                        return prompt + "\n" + content
        except: pass
    return None

async def main():
    global responses
    total_conc = CONCURRENT_PER_INSTANCE * len(PORTS)
    sem = asyncio.Semaphore(total_conc)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=total_conc)) as session:
        for cs in range(0, len(prompts), 400):
            if len(responses) >= NUM_RESPONSES:
                print(f"[GEN] Hit {NUM_RESPONSES}!"); break
            chunk = prompts[cs:cs+400]
            results = await asyncio.gather(*[gen_one(session, p, sem, i) for i,p in enumerate(chunk)])
            new = sum(1 for r in results if r)
            for r in results:
                if r: responses.append(r)
            elapsed = time.time() - t0
            added = len(responses) - len(existing)
            rate = added / max(elapsed, 1)
            remaining = NUM_RESPONSES - len(responses)
            eta = remaining / max(rate, 0.01)
            print(f"[GEN] {len(responses)}/{NUM_RESPONSES} | +{new}/400 | "
                  f"{rate:.1f} resp/s | ETA: {eta/60:.0f}min", flush=True)
            json.dump(responses, open(resp_file, "w"))

asyncio.run(main())
json.dump(responses, open(resp_file, "w"))
print(f"\n{'='*60}")
print(f"DONE: {len(responses)} responses in {(time.time()-t0)/60:.0f}min")
print(f"File: {resp_file}")
print(f"Size: {os.path.getsize(resp_file)/1e6:.0f} MB")
print(f"\nDownload to your machine:")
print(f"  scp -P <PORT> root@<IP>:{resp_file} ~/nvfp4_v5_cache/responses.json")
print(f"{'='*60}")

# Cleanup
for p in procs:
    p.terminate()
