"""
GH05T3 Nightly Full Evolution Runner
Phase 1: EnCompass KAIROS (10 cycles with backtracking)
Phase 2: Group Evolution (all 7 SAGE agents simultaneously)
Phase 3: Compound Intelligence Report

Run: python3 .agents/skills/nightly_full_evolution.py
"""
import sys, json, datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from systems.omega.kairos.encompass_backtrack import EnCompassBacktracker
from systems.omega.kairos.group_evolution import GroupEvolutionEngine
from skills.kairos_with_backtrack import run as run_encompass

def run_nightly():
    started = datetime.datetime.utcnow().isoformat()
    print(f"\n{'#'*60}")
    print(f"  GH05T3 NIGHTLY EVOLUTION — {started}")
    print(f"{'#'*60}\n")

    print(">>> PHASE 1: EnCompass KAIROS Backtracking (10 cycles)\n")
    encompass_summary = run_encompass(num_cycles=10)

    print("\n>>> PHASE 2: Group Evolution — All 7 SAGE Agents\n")
    ge = GroupEvolutionEngine()
    group_summary = ge.run_round()

    print("\n>>> PHASE 3: Compound Intelligence Report\n")
    print(ge.compound_intelligence_report())
    bt = EnCompassBacktracker()
    print(f"\n{bt.meta_agent_briefing()}")

    completed = datetime.datetime.utcnow().isoformat()
    elite_k = encompass_summary.get("elite_count", 0)
    elite_g = group_summary.get("elite_count", 0)

    print(f"\n{'#'*60}")
    print(f"  NIGHTLY EVOLUTION COMPLETE")
    print(f"  Started:  {started}")
    print(f"  Finished: {completed}")
    print(f"  KAIROS elite:   {elite_k}/10")
    print(f"  Group elite:    {elite_g}/7")
    print(f"  Total elite:    {elite_k + elite_g}")
    print(f"  Retry success:  {encompass_summary.get('retry_success_rate', 0)}%")
    print(f"{'#'*60}\n")

    return {"started": started, "completed": completed,
            "phase1_kairos": encompass_summary, "phase2_group": group_summary,
            "total_elite": elite_k + elite_g}

if __name__ == "__main__":
    run_nightly()
