"""gateway/zero_committee.py — ZERO Committee 4-agent governance."""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class CommitteeDecision:
    decision_id: str      # uuid4 hex
    proposal: str         # proposer output
    verdict: str          # EXECUTE | REJECT | REVISION_REQUESTED | ESCALATED
    audit_verdict: str    # PASS | FLAG | FAIL
    strategy_score: int   # 1-10
    chair_activated: bool
    rounds: int
    reason: str
    timestamp: float


class ZEROCommittee:
    GATEWAY_URL = "http://localhost:9000"
    PRIMARY_BRAIN = "qwen2.5:7b"
    VERIFIER_MODEL = "deepseek-coder:6.7b"
    CONSENSUS_THRESHOLD = 7

    async def decide(
        self, decision_context: str, max_chair_rounds: int = 3
    ) -> CommitteeDecision:
        decision_id = uuid.uuid4().hex
        t0 = time.time()

        # Step 1: Proposer
        proposal = await self._proposer_generate(decision_context)

        # Step 2: Auditor + Strategist in parallel
        (audit_verdict, audit_reason), (strategy_score, strategy_reasoning) = (
            await asyncio.gather(
                self._auditor_review(proposal, decision_context),
                self._strategist_score(proposal, decision_context),
            )
        )

        # Step 3: Immediate REJECT on FAIL
        if audit_verdict == "FAIL":
            return CommitteeDecision(
                decision_id=decision_id,
                proposal=proposal,
                verdict="REJECT",
                audit_verdict=audit_verdict,
                strategy_score=strategy_score,
                chair_activated=False,
                rounds=0,
                reason=audit_reason,
                timestamp=t0,
            )

        # Step 4: Clean consensus — no Chair needed
        if audit_verdict == "PASS" and strategy_score >= self.CONSENSUS_THRESHOLD:
            return CommitteeDecision(
                decision_id=decision_id,
                proposal=proposal,
                verdict="EXECUTE",
                audit_verdict=audit_verdict,
                strategy_score=strategy_score,
                chair_activated=False,
                rounds=0,
                reason=strategy_reasoning,
                timestamp=t0,
            )

        # Step 5: Chair resolves
        chair_verdict = "EXECUTE"
        chair_reason = "Chair default approval"
        for round_num in range(1, max_chair_rounds + 1):
            chair_verdict, chair_reason = await self._chair_resolve(
                proposal, audit_verdict, strategy_score, decision_context, round_num
            )
            if chair_verdict in ("EXECUTE", "REJECT", "REVISION_REQUESTED"):
                return CommitteeDecision(
                    decision_id=decision_id,
                    proposal=proposal,
                    verdict=chair_verdict,
                    audit_verdict=audit_verdict,
                    strategy_score=strategy_score,
                    chair_activated=True,
                    rounds=round_num,
                    reason=chair_reason,
                    timestamp=t0,
                )

        # Step 6: Rounds exhausted
        return CommitteeDecision(
            decision_id=decision_id,
            proposal=proposal,
            verdict="ESCALATED",
            audit_verdict=audit_verdict,
            strategy_score=strategy_score,
            chair_activated=True,
            rounds=max_chair_rounds,
            reason=f"Consensus not reached after {max_chair_rounds} Chair rounds",
            timestamp=t0,
        )

    async def _proposer_generate(self, context: str) -> str:
        p = f"ZERO Committee Proposer: generate a clear, actionable proposal.\nContext:\n{context}\nProposal:"
        r = await self._llm_call(self.PRIMARY_BRAIN, p)
        return r.strip() or f"Proposal for: {context[:200]}"

    async def _auditor_review(self, proposal: str, context: str) -> tuple[str, str]:
        p = (f"ZERO Committee Auditor: review for safety/alignment.\n"
             f"Respond MUST start with PASS, FLAG, or FAIL then explain.\n"
             f"Context:\n{context}\nProposal:\n{proposal}\nAudit verdict:")
        raw = await self._llm_call(self.VERIFIER_MODEL, p) or await self._llm_call(self.PRIMARY_BRAIN, p)
        upper = raw.upper()
        v = "FAIL" if "FAIL" in upper else ("FLAG" if "FLAG" in upper else "PASS")
        return v, raw.strip() or f"Audit verdict: {v}"

    async def _strategist_score(self, proposal: str, context: str) -> tuple[int, str]:
        p = (f"ZERO Committee Strategist: score long-horizon alignment 1-10.\n"
             f"Start response with the integer score, then explain.\n"
             f"Context:\n{context}\nProposal:\n{proposal}\nScore (1-10):")
        raw = await self._llm_call(self.PRIMARY_BRAIN, p)
        m = re.search(r"\b([1-9]|10)\b", raw)
        score = max(1, min(10, int(m.group(1)) if m else 5))
        return score, raw.strip() or f"Strategic score: {score}"

    async def _chair_resolve(
        self, proposal: str, audit: str, score: int, context: str, round_num: int
    ) -> tuple[str, str]:
        p = (f"ZERO Committee Chair: resolve tie (round {round_num}).\n"
             f"Response MUST contain EXECUTE, REJECT, or REVISION then explain.\n"
             f"Context:\n{context}\nProposal:\n{proposal}\n"
             f"Auditor: {audit} | Strategist: {score}/10\nChair decision:")
        raw = await self._llm_call(self.PRIMARY_BRAIN, p)
        upper = raw.upper()
        v = "REJECT" if "REJECT" in upper else ("REVISION_REQUESTED" if "REVISION" in upper else "EXECUTE")
        return v, raw.strip() or f"Chair decision: {v}"

    async def _llm_call(self, model: str, prompt: str) -> str:
        """POST to gateway /v1/chat/completions. Returns raw text, empty string on failure."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        url = f"{self.GATEWAY_URL}/v1/chat/completions"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.warning("ZEROCommittee LLM call failed (model=%s): %s", model, exc)
            return ""


_committee: Optional[ZEROCommittee] = None


def get_committee() -> ZEROCommittee:
    global _committee
    if _committee is None:
        _committee = ZEROCommittee()
    return _committee
