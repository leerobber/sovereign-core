
"""Lightweight eval harness for gateway and agent wiring."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import aiohttp
import yaml

ROOT = Path(__file__).resolve().parents[1]


async def _check_http(name: str, url: str, expect_status: int = 200) -> dict:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                ok = r.status == expect_status
                return {"name": name, "ok": ok, "status": r.status, "url": url}
    except Exception as e:
        return {"name": name, "ok": False, "error": str(e)[:120], "url": url}


async def run_suite(suite_path: Path) -> dict[str, Any]:
    with open(suite_path) as f:
        suite = yaml.safe_load(f) or {}
    checks = suite.get("checks", [])
    results = await asyncio.gather(*[
        _check_http(c["name"], c["url"], c.get("expect_status", 200))
        for c in checks
    ])
    passed = sum(1 for r in results if r.get("ok"))
    return {"suite": suite.get("name", suite_path.stem), "passed": passed, "total": len(results), "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    args = ap.parse_args()
    suite = Path(args.suite)
    if not suite.is_absolute():
        suite = ROOT / "eval" / "suites" / suite.name
    report = asyncio.run(run_suite(suite))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
