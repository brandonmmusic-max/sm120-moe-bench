"""SM120 attention backend launcher — registers the backend override then starts vLLM."""
import asyncio

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
