"""
RES-07: Self-Verification Loops — Upstream Error Filtering
Sovereign Core — gateway integration

Implementation path from issue:
  1. Add verification pass to every ARSO proposal
  2. After Synthetic Architect generates a fix, route through DeepSeek-Coder on Radeon
  3. DeepSeek verifies logical correctness before fix hits SUP-1 Stage I
  4. Filters bad proposals upstream → saves sandbox compute

Architecture:
  ARSO Proposal → [Verifier: DeepSeek-Coder @ Radeon:8002] → PASS/FAIL → SUP-1 Stage I
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger(__name__)

GATEWAY_URL = "http://localhost:8000"
VERIFIER_MODEL = "deepseek-coder-33b"   # Radeon 780M backend


class VerificationVerdict(str, Enum):
    PASS    = "PASS"     # Proposal is logically correct — forward to SUP-1
    PARTIAL = "PARTIAL"  # Proposal has minor issues — forward with warnings
    FAIL    = "FAIL"     # Proposal is flawed — reject, save sandbox compute


@dataclass
class VerificationResult:
    verdict: VerificationVerdict
    confidence: float           # 0.0–1.0
    issues: list[str]
    suggestions: list[str]
    raw_response: str
    proposal_id: Optional[str] = None

    @property
    def should_forward_to_sup1(self) -> bool:
        """Only forward PASS and PARTIAL (with warnings) to SUP-1 Stage I."""
        return self.verdict in (VerificationVerdict.PASS, VerificationVerdict.PARTIAL)


VERIFIER_SYSTEM_PROMPT = """You are a code and logic verifier for the Sovereign Core ARSO system.
Your role is to verify proposed fixes BEFORE they reach the sandbox (SUP-1 Stage I).
This saves compute by filtering bad proposals upstream.

For each proposal, check:
1. Logical correctness — does the fix actually solve the stated problem?
2. Type safety — are types compatible throughout?
3. Edge cases — are boundary conditions handled?
4. Side effects — could this fix break other components?
5. Performance — does this introduce regressions?

Respond ONLY in this exact format:
VERDICT: [PASS|PARTIAL|FAIL]
CONFIDENCE: [0.0-1.0]
ISSUES:
- <issue 1 or "none">
SUGGESTIONS:
- <suggestion 1 or "none">"""


def verify_proposal(
    proposal: str,
    bottleneck_context: str,
    proposal_id: Optional[str] = None,
    timeout: int = 60,
) -> VerificationResult:
    """
    Route a proposal through DeepSeek-Coder on Radeon for verification.
    Returns a VerificationResult with verdict and issues.

    Called by ARSO Orchestrator after Synthetic Architect generates a fix,
    BEFORE the fix is sent to SUP-1 Stage I sandbox.
    """
    user_message = (
        f"Bottleneck context:\n{bottleneck_context}\n\n"
        f"Proposed fix:\n{proposal}\n\n"
        "Verify this proposal."
    )

    payload = {
        "model": VERIFIER_MODEL,
        "messages": [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json=payload,
            params={"model_id": VERIFIER_MODEL},
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("Verifier request failed: %s", e)
        # On gateway failure, default to PARTIAL to avoid blocking the pipeline
        return VerificationResult(
            verdict=VerificationVerdict.PARTIAL,
            confidence=0.0,
            issues=[f"Verifier unreachable: {e}"],
            suggestions=["Check DeepSeek-Coder backend on Radeon (port 8002)"],
            raw_response="",
            proposal_id=proposal_id,
        )

    return _parse_verification_response(raw, proposal_id)


def _parse_verification_response(raw: str, proposal_id: Optional[str]) -> VerificationResult:
    """Parse the structured verifier response."""
    lines = raw.strip().split("\n")
    verdict = VerificationVerdict.PARTIAL
    confidence = 0.5
    issues: list[str] = []
    suggestions: list[str] = []
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            try:
                verdict = VerificationVerdict(v)
            except ValueError:
                verdict = VerificationVerdict.PARTIAL
        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
            except ValueError:
                confidence = 0.5
        elif line.startswith("ISSUES:"):
            section = "issues"
        elif line.startswith("SUGGESTIONS:"):
            section = "suggestions"
        elif line.startswith("- ") and section == "issues":
            item = line[2:].strip()
            if item.lower() != "none":
                issues.append(item)
        elif line.startswith("- ") and section == "suggestions":
            item = line[2:].strip()
            if item.lower() != "none":
                suggestions.append(item)

    return VerificationResult(
        verdict=verdict,
        confidence=confidence,
        issues=issues,
        suggestions=suggestions,
        raw_response=raw,
        proposal_id=proposal_id,
    )


class ARSOVerificationPipeline:
    """
    Wraps the ARSO proposal pipeline with upstream verification.

    Usage:
        pipeline = ARSOVerificationPipeline()
        result = pipeline.process(proposal, bottleneck_context)
        if result.should_forward_to_sup1:
            sup1.submit(proposal, warnings=result.issues)
        else:
            logger.info("Proposal %s rejected upstream — saved sandbox compute", proposal_id)
    """

    def __init__(self, strict_mode: bool = False) -> None:
        # strict_mode=True: reject PARTIAL as well (only PASS goes to SUP-1)
        self.strict_mode = strict_mode
        self.stats = {"total": 0, "passed": 0, "partial": 0, "rejected": 0, "compute_saved": 0}

    def process(
        self,
        proposal: str,
        bottleneck_context: str,
        proposal_id: Optional[str] = None,
    ) -> VerificationResult:
        self.stats["total"] += 1
        result = verify_proposal(proposal, bottleneck_context, proposal_id)

        if result.verdict == VerificationVerdict.PASS:
            self.stats["passed"] += 1
            logger.info("✓ Proposal %s PASSED verification (confidence=%.2f)", proposal_id, result.confidence)
        elif result.verdict == VerificationVerdict.PARTIAL:
            self.stats["partial"] += 1
            forward = not self.strict_mode
            logger.info(
                "%s Proposal %s PARTIAL (confidence=%.2f) — %s to SUP-1",
                "~" if forward else "✗",
                proposal_id, result.confidence,
                "forwarding" if forward else "rejecting (strict mode)",
            )
            if not forward:
                self.stats["compute_saved"] += 1
        else:
            self.stats["rejected"] += 1
            self.stats["compute_saved"] += 1
            logger.info(
                "✗ Proposal %s FAILED verification — rejected upstream, sandbox compute saved",
                proposal_id,
            )

        return result

    def report(self) -> dict:
        total = self.stats["total"] or 1
        return {
            **self.stats,
            "pass_rate": round(self.stats["passed"] / total, 3),
            "rejection_rate": round(self.stats["rejected"] / total, 3),
            "compute_saved_pct": round(self.stats["compute_saved"] / total * 100, 1),
        }
