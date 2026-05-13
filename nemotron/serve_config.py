"""
nemotron/serve_config.py

vLLM serving configuration for Nvidia Nemotron-3-Nano-30B.

Nemotron-3-Nano architecture (arXiv:2512.20848):
  - 30B total parameters, 3.5B active per token (MoE sparsity)
  - 23 Mamba-2 layers + 23 MoE layers (128 experts + 1 shared, 6 active)
  - 6 standard Attention layers
  - 1M context window (256k default in HuggingFace config)
  - Requires custom reasoning parser (nano_v3) for <think> block extraction

Available model variants on HuggingFace:
  - nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8   (official FP8, ~15GB VRAM)
  - stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ (community AWQ INT4, ~10GB VRAM)

VRAM requirements (approx):
  | Variant | VRAM  | GPU target              |
  |---------|-------|------------------------|
  | FP8     | ~15GB | RTX 5050 (16GB)        |
  | AWQ-4b  | ~10GB | RTX 5050 or RTX 4090   |
  | BF16    | ~60GB | Multi-GPU (2×RTX 5090) |
"""
from __future__ import annotations

import dataclasses
import os
from enum import Enum
from typing import List, Optional


class ModelVariant(str, Enum):
    """Available quantization variants."""
    FP8 = "fp8"
    AWQ = "awq"
    BF16 = "bf16"


@dataclasses.dataclass
class NemotronConfig:
    """
    Configuration for launching Nemotron-3-Nano via vLLM.

    Maps to `nemotron/serve_config.yaml` on disk (written by to_yaml()).
    Environment variables override dataclass defaults at runtime.

    Attributes
    ----------
    model_id         : HuggingFace model identifier.
    variant          : Quantization format (fp8, awq, bf16).
    served_model_name: Name exposed by the OpenAI-compatible API endpoint.
    api_base         : Full URL of the running vLLM instance.
    port             : Port for Nemotron vLLM instance.
    tensor_parallel  : Number of GPUs for tensor parallelism.
    max_model_len    : Context window length (tokens). Default 262144 (256k).
    max_num_seqs     : Max concurrent sequences; lower = less VRAM pressure.
    enable_thinking  : Enable Nemotron's internal <think> CoT reasoning.
    reasoning_budget : Max tokens for internal reasoning trace.
    kv_cache_dtype   : KV-cache quantization ("fp8" or "auto").
    tool_call_parser : vLLM tool parser name for structured tool calls.
    """
    # Model identity
    model_id: str = dataclasses.field(
        default_factory=lambda: os.getenv(
            "NEMOTRON_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
        )
    )
    variant: ModelVariant = ModelVariant.FP8

    # Endpoint
    served_model_name: str = "nemotron-nano"
    api_base: str = dataclasses.field(
        default_factory=lambda: os.getenv(
            "NEMOTRON_API_BASE", "http://localhost:8002/v1"   # separate port from Qwen @ 8001
        )
    )
    port: int = 8002

    # Serving
    tensor_parallel: int = dataclasses.field(
        default_factory=lambda: int(os.getenv("NEMOTRON_TP", "1"))
    )
    max_model_len: int = dataclasses.field(
        default_factory=lambda: int(os.getenv("NEMOTRON_MAX_LEN", "262144"))
    )
    max_num_seqs: int = 8
    kv_cache_dtype: str = "fp8"

    # Reasoning
    enable_thinking: bool = True
    reasoning_budget: int = 10_000
    tool_call_parser: str = "qwen3_coder"   # compatible parser available in vLLM >= 0.12

    @property
    def openai_api_key(self) -> str:
        return os.getenv("NEMOTRON_API_KEY", "sovereign-core-nemotron")

    def to_vllm_args(self) -> List[str]:
        """Generate the vLLM CLI argument list for this config."""
        args = [
            "vllm", "serve", self.model_id,
            "--served-model-name", self.served_model_name,
            "--port", str(self.port),
            "--max-num-seqs", str(self.max_num_seqs),
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--tool-call-parser", self.tool_call_parser,
            "--reasoning-parser-plugin", "nano_v3_reasoning_parser.py",
            "--reasoning-parser", "nano_v3",
            "--kv-cache-dtype", self.kv_cache_dtype,
        ]
        if self.variant == ModelVariant.AWQ:
            args += ["--quantization", "awq"]
        return args

    def to_yaml(self) -> str:
        """Render a human-readable YAML config for documentation."""
        lines = [
            "# nemotron/serve_config.yaml",
            "# Auto-generated — edit NemotronConfig and regenerate with:",
            "#   python -c \"from nemotron.serve_config import NemotronConfig; print(NemotronConfig().to_yaml())\"",
            "",
            "model:",
            f"  id: {self.model_id}",
            f"  variant: {self.variant.value}",
            f"  served_name: {self.served_model_name}",
            "",
            "serving:",
            f"  port: {self.port}",
            f"  tensor_parallel: {self.tensor_parallel}",
            f"  max_model_len: {self.max_model_len}",
            f"  max_num_seqs: {self.max_num_seqs}",
            f"  kv_cache_dtype: {self.kv_cache_dtype}",
            "",
            "reasoning:",
            f"  enable_thinking: {str(self.enable_thinking).lower()}",
            f"  reasoning_budget: {self.reasoning_budget}",
            "",
            "endpoint:",
            f"  api_base: {self.api_base}",
            "  api_key: $NEMOTRON_API_KEY  # set in .env",
        ]
        return "\n".join(lines)

    def launch_command(self) -> str:
        """Full shell command string for launching Nemotron via vLLM."""
        env = "VLLM_USE_FLASHINFER_MOE_FP8=1 \\\n"
        args = self.to_vllm_args()
        return env + " \\\n  ".join(args)


# ── Convenience constant ──────────────────────────────────────────────────────

VLLM_LAUNCH_TEMPLATE = NemotronConfig().launch_command()

# ── AWQ variant convenience ───────────────────────────────────────────────────

def nemotron_awq_config() -> NemotronConfig:
    """
    Config for community AWQ 4-bit variant (~10GB VRAM).
    Best for RTX 5050 (16GB) running alongside Qwen2.5-32B-AWQ on same machine.
    """
    return NemotronConfig(
        model_id="stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ",
        variant=ModelVariant.AWQ,
        kv_cache_dtype="auto",
    )
