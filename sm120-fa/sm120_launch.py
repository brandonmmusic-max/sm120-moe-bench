"""SM120 attention backend launcher — pre-compiles kernels, then starts vLLM."""
import asyncio
import sys
import time
import os

print("[SM120] Pre-compiling SM120 kernels before starting vLLM...", flush=True)
t0 = time.time()

# Pre-compile decode kernel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import sm120_flash_decode_ext
    print(f"[SM120] Decode kernel compiled in {time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"[SM120] WARNING: Decode kernel compilation failed: {e}", flush=True)

# Pre-compile prefill kernel
t1 = time.time()
try:
    import sm120_flash_prefill_ext
    print(f"[SM120] Prefill kernel compiled in {time.time()-t1:.1f}s", flush=True)
except Exception as e:
    print(f"[SM120] WARNING: Prefill kernel compilation failed: {e}", flush=True)

print(f"[SM120] All kernels pre-compiled in {time.time()-t0:.1f}s total", flush=True)

# Register SM120 backend to override FLASHINFER before vLLM imports it
from vllm.v1.attention.backends.registry import (
    register_backend,
    AttentionBackendEnum,
)

register_backend(
    AttentionBackendEnum.FLASHINFER,
    "vllm.v1.attention.backends.sm120_vllm_backend.SM120FlashAttentionBackend",
)

# Now run the vLLM API server
from vllm.entrypoints.openai.api_server import (
    run_server,
    make_arg_parser,
    FlexibleArgumentParser,
)

if __name__ == "__main__":
    parser = make_arg_parser(FlexibleArgumentParser(
        description="vLLM OpenAI API server with SM120 attention backend"
    ))
    args = parser.parse_args()
    asyncio.run(run_server(args))
