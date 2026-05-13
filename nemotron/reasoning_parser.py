"""
nemotron/reasoning_parser.py

Nano-V3 reasoning parser for Nemotron-3-Nano's internal <think> CoT blocks.

Nemotron-3-Nano generates responses in the format:
    <think>
    [internal chain-of-thought reasoning — up to reasoning_budget tokens]
    </think>
    [final response visible to caller]

This module:
  1. Extracts and separates thinking from final response text
  2. Provides budget enforcement (truncate reasoning if it exceeds the limit)
  3. Is designed as a drop-in plugin for vLLM's `--reasoning-parser` system
     (compatible with the `nano_v3` parser spec)
  4. Also works standalone for post-processing API responses

Design note
-----------
The vLLM plugin version of this class is registered as `nano_v3` via the
`--reasoning-parser-plugin nano_v3_reasoning_parser.py` CLI flag.
The standalone version (used here) is used in benchmark.py and ab_router.py
without requiring a live vLLM instance.

Reference: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 model card
"""
from __future__ import annotations

import dataclasses
import re
import time
from typing import Optional, Tuple


# ── Data types ────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ReasoningTrace:
    """
    Parsed output from a Nemotron-3-Nano response.

    Attributes
    ----------
    thinking        : The internal CoT trace (None if not present / disabled).
    response        : The final response visible to the caller.
    thinking_tokens : Estimated thinking token count (4 chars ≈ 1 token).
    budget_exceeded : True if thinking was truncated to fit reasoning_budget.
    raw             : Original unparsed string.
    """
    thinking: Optional[str]
    response: str
    thinking_tokens: int = 0
    budget_exceeded: bool = False
    raw: str = ""

    @property
    def has_thinking(self) -> bool:
        return self.thinking is not None and len(self.thinking) > 0

    def summary(self) -> str:
        status = "✓" if not self.budget_exceeded else "⚠ budget exceeded"
        think_info = f"{self.thinking_tokens} thinking tokens — {status}" if self.has_thinking else "no CoT"
        return f"[Nemotron] {think_info} | {len(self.response)} chars response"


# ── Parser ────────────────────────────────────────────────────────────────────

# Compiled patterns for speed
_THINK_OPEN  = re.compile(r"<think>\s*", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"\s*</think>", re.IGNORECASE)
_THINK_FULL  = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


class NemotronReasoningParser:
    """
    Extracts and manages Nemotron-3-Nano's internal <think> CoT reasoning.

    Parameters
    ----------
    reasoning_budget : Max tokens allowed for internal thinking.
                       Approx estimate: 4 characters ≈ 1 token.
    strip_thinking   : If True, thinking is extracted but NOT included in
                       the returned response (callers only see final answer).
    """

    CHARS_PER_TOKEN = 4   # rough approximation for budget enforcement

    def __init__(self, reasoning_budget: int = 10_000, strip_thinking: bool = True):
        self.reasoning_budget = reasoning_budget
        self.strip_thinking = strip_thinking

    def parse(self, text: str) -> ReasoningTrace:
        """
        Parse a complete Nemotron response string.

        Args:
            text: Raw model output, potentially containing <think>...</think>.
        Returns:
            ReasoningTrace with thinking and response separated.
        """
        match = _THINK_FULL.search(text)
        if not match:
            return ReasoningTrace(
                thinking=None,
                response=text.strip(),
                thinking_tokens=0,
                raw=text,
            )

        raw_thinking = match.group(1).strip()
        # Everything after </think>
        after = text[match.end():].strip()
        # Budget enforcement
        budget_chars = self.reasoning_budget * self.CHARS_PER_TOKEN
        exceeded = len(raw_thinking) > budget_chars
        if exceeded:
            raw_thinking = raw_thinking[:budget_chars] + " [truncated]"

        thinking_tokens = len(raw_thinking) // self.CHARS_PER_TOKEN

        return ReasoningTrace(
            thinking=raw_thinking,
            response=after,
            thinking_tokens=thinking_tokens,
            budget_exceeded=exceeded,
            raw=text,
        )

    def parse_streaming(self, chunks: list[str]) -> ReasoningTrace:
        """
        Parse a response assembled from streaming chunks.
        Joins chunks and delegates to parse().
        """
        return self.parse("".join(chunks))

    def strip_think_tags(self, text: str) -> str:
        """Remove <think> blocks entirely, returning only the final response."""
        return _THINK_FULL.sub("", text).strip()

    def extract_thinking_only(self, text: str) -> Optional[str]:
        """Return only the thinking trace, or None if not present."""
        match = _THINK_FULL.search(text)
        return match.group(1).strip() if match else None

    def disable_thinking(self, messages: list[dict]) -> list[dict]:
        """
        Inject the Nemotron disable-thinking signal into a messages list.
        When thinking is disabled, the model responds directly without CoT.
        Achieved by appending a system instruction.
        """
        system_turn = {
            "role": "system",
            "content": "You are a helpful assistant. /no_think",
        }
        existing = [m for m in messages if m.get("role") != "system"]
        return [system_turn] + existing

    def set_budget(self, messages: list[dict], budget: int) -> list[dict]:
        """
        Inject a reasoning budget directive into a messages list.
        Tells Nemotron to limit its internal thinking to `budget` tokens.
        """
        directive = f"Think for at most {budget} tokens before answering."
        for m in messages:
            if m.get("role") == "system":
                m = dict(m)
                m["content"] = directive + "\n" + m["content"]
                return [m] + [x for x in messages if x is not m]
        return [{"role": "system", "content": directive}] + messages
