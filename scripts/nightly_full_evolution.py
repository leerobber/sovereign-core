"""
Nightly Full Evolution Runner — calls the sovereign-core gateway HTTP API.

Phases:
  1. ARSO production cycle — all 7 specialist agents × N cycles each
  2. EnCompass backtracking summary (from local kairos module if available)
  3. Group evolution summary (from local kairos module if available)
  4. Final compound report

Run:
  python scripts/nightly_full_evolution.py
  python scripts/nightly_full_evolution.py --cycles 3 --budget 3600
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GATEWAY = "http://localhost:9000"


async def _wait_for_gateway(retries: int = 5, delay: float = 3.0) -> bool:
    for i in range(retries):
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{GATEWAY}/health")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        if i < retries - 1:
            await asyncio.sleep(delay)
    return False


async def run_nightly(cycles_per_agent: int = 2, time_budget_s: float = 3600.0) -> dict:
    started = datetime.datetime.utcnow().isoformat()
    sep = "=" * 60

    print(f"\n{sep}")
    print(f"  SOVEREIGN NIGHTLY EVOLUTION — {started}")
    print(f"  Gateway: {GATEWAY}  |  cycles/agent={cycles_per_agent}  |  budget={time_budget_s:.0f}s")
    print(f"{sep}\n")

    # ── Gateway health check ───────────────────────────────────────────────────
    print("[0/3] Checking gateway health...")
    alive = await _wait_for_gateway()
    if not alive:
        print(f"  ERROR: Gateway at {GATEWAY} not reachable. Is sovereign-core running?")
        print(f"  Start it with: START_ALL.bat  or  python -m uvicorn gateway.main:app --port 9000")
        sys.exit(1)
    print("  Gateway: ONLINE\n")

    # ── Phase 1: ARSO Production Cycle ────────────────────────────────────────
    print("[1/3] ARSO Production Cycle — 7 specialist agents in parallel...")
    async with httpx.AsyncClient(timeout=time_budget_s + 60) as client:
        resp = await client.post(
            f"{GATEWAY}/kairos/orchestrate",
            json={
                "cycles_per_agent": cycles_per_agent,
                "score_threshold": 0.6,
                "time_budget_s": time_budget_s,
            },
        )
        resp.raise_for_status()
        report = resp.json()

    print(f"  Run ID:       {report.get('run_id', 'N/A')}")
    print(f"  Wall clock:   {report.get('wall_clock_s', 0):.1f}s")
    print(f"  Agents:       {report.get('total_agents', 0)}")
    print(f"  SAGE cycles:  {report.get('total_cycles', 0)}")
    print(f"  Elites:       {report.get('elite_count', 0)}/{report.get('total_agents', 0)}")
    print(f"  Best score:   {report.get('best_score', 0):.4f}")
    print(f"  Status:       {report.get('status', 'unknown')}\n")

    # ── Phase 2: EnCompass backtracking (optional — local module) ─────────────
    print("[2/3] EnCompass backtracking stats...")
    encompass_stats = {}
    try:
        from kairos.encompass_backtrack import EnCompassBacktracker
        bt = EnCompassBacktracker()
        briefing = bt.meta_agent_briefing()
        encompass_stats = {"briefing": briefing}
        print(f"  {briefing[:200]}...\n" if len(briefing) > 200 else f"  {briefing}\n")
    except Exception as e:
        print(f"  EnCompass module unavailable: {e}\n")

    # ── Phase 3: Group evolution stats (optional — local module) ──────────────
    print("[3/3] Group evolution stats...")
    group_stats = {}
    try:
        from kairos.group_evolution import GroupEvolutionEngine
        ge = GroupEvolutionEngine()
        ci_report = ge.compound_intelligence_report()
        group_stats = {"compound_intelligence": ci_report}
        print(f"  {ci_report[:200]}...\n" if len(ci_report) > 200 else f"  {ci_report}\n")
    except Exception as e:
        print(f"  Group evolution module unavailable: {e}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    completed = datetime.datetime.utcnow().isoformat()
    total_elite = report.get("elite_count", 0)

    print(sep)
    print("  NIGHTLY EVOLUTION COMPLETE")
    print(f"  Started:      {started}")
    print(f"  Finished:     {completed}")
    print(f"  Total elites: {total_elite}")
    print(f"  Best score:   {report.get('best_score', 0):.4f}")
    print(f"  Run ID:       {report.get('run_id', 'N/A')}")
    print(sep)

    return {
        "started": started,
        "completed": completed,
        "arso_report": report,
        "encompass": encompass_stats,
        "group_evolution": group_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Sovereign nightly evolution runner")
    parser.add_argument("--cycles", type=int, default=2,
                        help="SAGE cycles per specialist agent (default: 2)")
    parser.add_argument("--budget", type=float, default=3600.0,
                        help="Wall-clock time budget in seconds (default: 3600)")
    args = parser.parse_args()

    result = asyncio.run(run_nightly(
        cycles_per_agent=args.cycles,
        time_budget_s=args.budget,
    ))

    # Save run log
    log_dir = ROOT / "data" / "nightly_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"nightly_{ts}.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
