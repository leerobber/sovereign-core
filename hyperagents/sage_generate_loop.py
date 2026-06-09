# Copyright (c) Meta Platforms, Inc. and affiliates.
# Sovereign Core adaptation — SAGE 4-agent co-evolution loop
#
# Maps generate_loop.py → SAGE (Self-Accelerating Generation Engine) architecture:
#   Agent 1: Proposer     — generates optimization candidates (ARSO proposals)
#   Agent 2: Critic       — adversarial review of proposals
#   Agent 3: Verifier     — logical/correctness verification (DeepSeek-Coder on Radeon)
#   Agent 4: Meta-Agent   — rewrites the improvement rules themselves (HyperAgents core)
#
# Entry point: run_sage_loop()

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Import local adapter — drop-in replacement for hyperagents agent/llm.py
try:
    from hyperagents.agent.llm_local import get_response_from_llm, LOCAL_PRIMARY_MODEL, LOCAL_CODER_MODEL
except ImportError:
    # Fallback: direct import when running from hyperagents root
    from agent.llm_local import get_response_from_llm, LOCAL_PRIMARY_MODEL, LOCAL_CODER_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model role assignments (maps to Sovereign Core backend routing)
# ---------------------------------------------------------------------------
PROPOSER_MODEL  = LOCAL_PRIMARY_MODEL   # RTX 5050 — Qwen2.5-32B-AWQ
CRITIC_MODEL    = LOCAL_PRIMARY_MODEL   # RTX 5050 — adversarial review
VERIFIER_MODEL  = LOCAL_CODER_MODEL     # Radeon 780M — DeepSeek-Coder logical check
META_MODEL      = LOCAL_PRIMARY_MODEL   # RTX 5050 — meta-agent self-rewrite

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SAGEProposal:
    generation: int
    proposer_output: str
    critic_feedback: str
    verifier_verdict: str          # "PASS" | "FAIL" | "PARTIAL"
    meta_rewrite: Optional[str]
    score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    accepted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SAGEArchive:
    """
    Pattern Memory — stores successful proposals as stepping stones (DGM-H concept).
    Every accepted proposal persists the full agent state so future bottlenecks
    can reconstruct the agent that produced a successful fix.
    """
    proposals: list[SAGEProposal] = field(default_factory=list)
    archive_path: Optional[Path] = None

    def add(self, proposal: SAGEProposal) -> None:
        self.proposals.append(proposal)
        if self.archive_path:
            self._persist()

    def best(self, n: int = 5) -> list[SAGEProposal]:
        return sorted(self.proposals, key=lambda p: p.score, reverse=True)[:n]

    def _persist(self) -> None:
        assert self.archive_path is not None
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.archive_path, "w") as f:
            json.dump([p.to_dict() for p in self.proposals], f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SAGEArchive":
        archive = cls(archive_path=path)
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            archive.proposals = [SAGEProposal(**r) for r in raw]
        return archive


# ---------------------------------------------------------------------------
# SAGE 4-agent co-evolution loop
# ---------------------------------------------------------------------------

PROPOSER_SYSTEM_PROMPT = """You are the Proposer agent in the SAGE co-evolution loop.
Your role: given a bottleneck description, generate a concrete optimization proposal.
Be specific — include code changes, config tweaks, or architectural modifications.
Format your output as:
PROPOSAL:
<your detailed proposal here>
RATIONALE:
<why this will work>"""

CRITIC_SYSTEM_PROMPT = """You are the Critic agent in the SAGE co-evolution loop.
Your role: adversarially review the Proposer's output.
Look for: logical flaws, missing edge cases, performance regressions, safety risks.
Format your output as:
VERDICT: [APPROVE | REJECT | REVISE]
ISSUES:
<list of specific issues>
SUGGESTED_FIXES:
<concrete fixes if REVISE>"""

VERIFIER_SYSTEM_PROMPT = """You are the Verifier agent (DeepSeek-Coder) in the SAGE loop.
Your role: verify logical correctness of the proposal after critic review.
Focus on: code correctness, type safety, algorithmic soundness.
Format your output as:
VERIFICATION: [PASS | FAIL | PARTIAL]
NOTES:
<specific correctness notes>"""

META_SYSTEM_PROMPT = """You are the Meta-Agent in the SAGE co-evolution loop.
Your role: rewrite the improvement rules themselves based on what worked and what didn't.
You have access to the archive of past proposals and their outcomes.
Output updated system prompts or generation strategies that will improve future cycles.
Format your output as:
META_UPDATE:
<updated strategy or prompt modifications>
REASONING:
<why these changes improve the loop>"""


def _agent_call(
    system_prompt: str,
    user_message: str,
    model: str,
    msg_history: Optional[list] = None,
    temperature: float = 0.3,
) -> tuple[str, list]:
    """Single agent turn — wraps get_response_from_llm with system prompt injection."""
    # Prepend system prompt as first user message if no history
    if not msg_history:
        msg_history = [{"role": "system", "text": system_prompt}]

    response, updated_history, _ = get_response_from_llm(
        msg=user_message,
        model=model,
        temperature=temperature,
        msg_history=msg_history,
    )
    return response, updated_history


def run_sage_loop(
    bottleneck_description: str,
    max_generations: int = 10,
    archive_path: Optional[str] = None,
    meta_rewrite_every: int = 3,
    score_threshold: float = 0.7,
) -> SAGEArchive:
    """
    SAGE 4-agent co-evolution loop.

    Args:
        bottleneck_description: Description of the system bottleneck to optimize.
        max_generations: Maximum number of proposal cycles.
        archive_path: Path to persist the archive (Pattern Memory).
        meta_rewrite_every: How often the meta-agent rewrites the loop rules.
        score_threshold: Minimum score to accept a proposal into the archive.

    Returns:
        SAGEArchive with all proposals and their outcomes.
    """
    archive = SAGEArchive(
        archive_path=Path(archive_path) if archive_path else None
    )

    # Load existing archive if available (DGM-H stepping stones)
    if archive.archive_path and archive.archive_path.exists():
        archive = SAGEArchive.load(archive.archive_path)
        logger.info("Loaded archive with %d prior proposals", len(archive.proposals))

    # Build context from best prior proposals (Pattern Memory lookup)
    prior_context = ""
    if archive.proposals:
        best_prior = archive.best(n=3)
        prior_context = "\n\nBest prior proposals (Pattern Memory):\n" + "\n---\n".join(
            f"Gen {p.generation}: {p.proposer_output[:300]}... [score={p.score:.2f}]"
            for p in best_prior
        )

    proposer_history: list = []
    meta_strategy: str = ""   # Updated by meta-agent each cycle

    for gen in range(1, max_generations + 1):
        logger.info("=== SAGE Generation %d / %d ===", gen, max_generations)

        # ── Agent 1: Proposer ────────────────────────────────────────────────
        proposer_input = (
            f"Bottleneck: {bottleneck_description}"
            f"{prior_context}"
            + (f"\n\nMeta-strategy update:\n{meta_strategy}" if meta_strategy else "")
        )
        proposal_text, proposer_history = _agent_call(
            system_prompt=PROPOSER_SYSTEM_PROMPT,
            user_message=proposer_input,
            model=PROPOSER_MODEL,
            msg_history=proposer_history if gen > 1 else None,
        )
        logger.info("Proposer output (%d chars)", len(proposal_text))

        # ── Agent 2: Critic ──────────────────────────────────────────────────
        critic_input = f"Review this proposal:\n\n{proposal_text}"
        critic_text, _ = _agent_call(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_message=critic_input,
            model=CRITIC_MODEL,
        )
        logger.info("Critic verdict captured")

        # ── Agent 3: Verifier (DeepSeek-Coder on Radeon) ────────────────────
        verifier_input = (
            f"Original proposal:\n{proposal_text}\n\n"
            f"Critic feedback:\n{critic_text}\n\n"
            "Verify logical correctness."
        )
        verifier_text, _ = _agent_call(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_message=verifier_input,
            model=VERIFIER_MODEL,
        )
        verifier_verdict = "PASS" if "PASS" in verifier_text.upper() else (
            "PARTIAL" if "PARTIAL" in verifier_text.upper() else "FAIL"
        )
        logger.info("Verifier: %s", verifier_verdict)

        # ── Agent 4: Meta-Agent (every N generations) ────────────────────────
        meta_text: Optional[str] = None
        if gen % meta_rewrite_every == 0:
            recent_proposals = archive.proposals[-meta_rewrite_every:] if archive.proposals else []
            meta_input = (
                f"Recent {len(recent_proposals)} proposals and outcomes:\n"
                + json.dumps([p.to_dict() for p in recent_proposals], indent=2)
                + "\n\nRewrite the improvement strategy to do better next cycle."
            )
            meta_text, _ = _agent_call(
                system_prompt=META_SYSTEM_PROMPT,
                user_message=meta_input,
                model=META_MODEL,
            )
            meta_strategy = meta_text
            logger.info("Meta-agent rewrote loop strategy (gen %d)", gen)

        # ── Score & archive ───────────────────────────────────────────────────
        # Scoring heuristic: PASS=1.0, PARTIAL=0.6, FAIL=0.2 × critic approval
        base_score = {"PASS": 1.0, "PARTIAL": 0.6, "FAIL": 0.2}[verifier_verdict]
        critic_approved = "APPROVE" in critic_text.upper()
        score = base_score * (1.0 if critic_approved else 0.5)

        proposal = SAGEProposal(
            generation=gen,
            proposer_output=proposal_text,
            critic_feedback=critic_text,
            verifier_verdict=verifier_verdict,
            meta_rewrite=meta_text,
            score=score,
            accepted=score >= score_threshold,
        )
        archive.add(proposal)

        logger.info(
            "Gen %d score=%.2f accepted=%s", gen, score, proposal.accepted
        )

        # Update prior context for next generation
        if proposal.accepted:
            prior_context = f"\n\nBest proposal so far (gen {gen}, score={score:.2f}):\n{proposal_text[:400]}"

    logger.info(
        "SAGE loop complete. %d / %d proposals accepted.",
        sum(1 for p in archive.proposals if p.accepted),
        len(archive.proposals),
    )
    return archive


# ---------------------------------------------------------------------------
# CLI entry point (mirrors generate_loop.py interface)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="SAGE 4-agent co-evolution loop")
    parser.add_argument("--bottleneck", required=True, help="Bottleneck description to optimize")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--archive-path", default="./sage_archive.json")
    parser.add_argument("--meta-rewrite-every", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    args = parser.parse_args()

    result = run_sage_loop(
        bottleneck_description=args.bottleneck,
        max_generations=args.max_generations,
        archive_path=args.archive_path,
        meta_rewrite_every=args.meta_rewrite_every,
        score_threshold=args.score_threshold,
    )

    best = result.best(n=3)
    print(f"\n{'='*60}")
    print(f"Top {len(best)} proposals:")
    for p in best:
        print(f"  Gen {p.generation}: score={p.score:.2f} verdict={p.verifier_verdict}")
        print(f"  {p.proposer_output[:200]}...")
