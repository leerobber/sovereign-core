#!/usr/bin/env python3
"""
scripts/sovereign.py — Sovereign Core Unified CLI
Inspect, control, and evolve the entire platform from the terminal.

Usage:
  python scripts/sovereign.py status
  python scripts/sovereign.py backends
  python scripts/sovereign.py evolve --cycles 5
  python scripts/sovereign.py ledger tail --n 20
  python scripts/sovereign.py bench --model nemotron-3-nano
  python scripts/sovereign.py kairos <agent_id>
  python scripts/sovereign.py auction bid <resource> <votes>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from typing import Any

GATEWAY = "http://localhost:8000"

COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "blue":   "\033[94m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "reset":  "\033[0m",
}


def c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def _get(path: str) -> Any:
    url = f"{GATEWAY}{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        print(c("red", f"✗ Gateway unreachable at {GATEWAY}: {e}"))
        print(c("dim", "  Make sure `gateway.main` is running on localhost:8000"))
        sys.exit(1)


def _post(path: str, body: dict) -> Any:
    url = f"{GATEWAY}{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        print(c("red", f"✗ Request failed: {e}"))
        sys.exit(1)


# ── Command handlers ───────────────────────────────────────────────────────────

def cmd_status(args):
    data = _get("/status/")
    uptime = data.get("uptime_s", 0)
    h, rem = divmod(int(uptime), 3600)
    m, s = divmod(rem, 60)

    print(c("bold", "\n╔══════════════════════════════════════╗"))
    print(c("bold",   "║  Sovereign Core — System Status       ║"))
    print(c("bold",   "╚══════════════════════════════════════╝"))
    print(f"  Uptime:   {h:02d}:{m:02d}:{s:02d}")
    print(f"  Version:  {data.get('version','?')}")
    print(f"  Time:     {time.strftime('%Y-%m-%d %H:%M:%S')}")

    backends = data.get("backends", [])
    print(c("bold", f"\n  GPU Backends ({len(backends)}):"))
    for b in backends:
        symbol = c("green", "●") if b.get("healthy") else c("red", "●")
        lat_raw = b.get("latency_ms")
        lat = f"{lat_raw:.1f}ms" if lat_raw is not None else "—"
        url = b.get("meta", {}).get("url", "")
        device = b.get("meta", {}).get("device_type", "")
        print(f"    {symbol} {b['name']:20s} {lat:>8s}  {device:10s}  {url}")

    auction = data.get("auction", {})
    if auction and "error" not in auction:
        print(c("bold", "\n  Auction (Vickrey-Quadratic):"))
        for k, v in auction.items():
            print(f"    {k}: {v}")

    print()


def cmd_backends(args):
    data = _get("/status/backends")
    print(c("bold", "\n╔══ GPU Backend Details ══╗\n"))
    for name, info in data.items():
        status = c("green", "HEALTHY") if info.get("healthy") else c("red", "DOWN")
        lat_raw = info.get("latency_ms")
        lat = f"{lat_raw:.2f}ms" if lat_raw is not None else "—"
        print(f"  {c('bold', name)}")
        print(f"    Status:      {status}")
        print(f"    Latency EMA: {lat}")
        print(f"    URL:         {info.get('url','?')}")
        print(f"    Device:      {info.get('device_type','?')}")
        print()


def cmd_evolve(args):
    cycles = args.cycles
    print(c("cyan", f"⟳ Running {cycles} KAIROS ARSO evolution cycle(s)...\n"))
    total_start = time.time()
    for i in range(cycles):
        t0 = time.time()
        result = _post("/kairos/evolve", {"cycles": 1})
        elapsed = time.time() - t0
        agent_id = result.get("agent_id", "?")
        gen = result.get("generation", "?")
        score = result.get("score", 0.0)
        verdict = result.get("verification_verdict", "?")
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  [{i+1:02d}/{cycles:02d}] agent={c('cyan', agent_id)}  gen={gen}  "
              f"score=[{bar}] {score:.3f}  verdict={verdict}  ({elapsed:.1f}s)")

    total = time.time() - total_start
    print(c("green", f"\n✓ {cycles} cycle(s) complete in {total:.1f}s"))


def cmd_kairos(args):
    agent_id = args.agent_id
    data = _get(f"/status/kairos/{agent_id}")
    if "error" in data:
        print(c("red", f"✗ {data['error']}"))
        return
    print(json.dumps(data, indent=2))


def cmd_ledger(args):
    n = args.n
    data = _get(f"/ledger/tail?n={n}")
    entries = data.get("entries", [])
    print(c("bold", f"\n╔══ Aegis-Vault — Last {len(entries)} Entries ══╗\n"))
    for e in entries:
        ts_str = time.strftime("%H:%M:%S", time.localtime(e.get("timestamp", 0)))
        op = e.get("operation_type", "?")
        backend = e.get("backend_id", "?")
        trust = e.get("trust_score", 0.0)
        integrity = c("green", "✓") if e.get("integrity_ok") else c("red", "✗")
        print(f"  {ts_str}  {integrity}  {op:30s}  {backend:15s}  trust={trust:.3f}")
    print()


def cmd_bench(args):
    model = args.model
    print(c("cyan", f"⟳ Benchmarking model: {model}"))
    t0 = time.time()
    result = _post("/benchmark/run", {"model_id": model})
    elapsed = time.time() - t0
    print(json.dumps(result, indent=2))
    print(c("dim", f"  (completed in {elapsed:.1f}s)"))


def cmd_auction_bid(args):
    resource = args.resource
    votes = args.votes
    print(c("cyan", f"⟳ Bidding {votes} votes for {resource}..."))
    result = _post("/auction/bid", {"resource_type": resource, "votes": votes})
    if result.get("won"):
        print(c("green", f"✓ Won! Payment: {result.get('payment_credits', 0)} credits"))
    else:
        print(c("yellow", f"✗ Lost auction. Winner: {result.get('winner', '?')}"))
    print(json.dumps(result, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="sovereign",
        description="Sovereign Core CLI — control the autonomous agent platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sovereign status
  sovereign evolve --cycles 10
  sovereign ledger tail --n 25
  sovereign bench --model nemotron-3-nano
  sovereign kairos <agent-uuid>
        """,
    )
    parser.add_argument(
        "--gateway", default=GATEWAY,
        help=f"Gateway URL (default: {GATEWAY})"
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # status
    sub.add_parser("status", help="Full system snapshot")

    # backends
    sub.add_parser("backends", help="GPU backend health details")

    # evolve
    evolve_p = sub.add_parser("evolve", help="Run KAIROS ARSO evolution cycles")
    evolve_p.add_argument("--cycles", type=int, default=1, metavar="N")

    # kairos
    kairos_p = sub.add_parser("kairos", help="Inspect a KAIROS agent")
    kairos_p.add_argument("agent_id")

    # ledger
    ledger_p = sub.add_parser("ledger", help="Aegis-Vault ledger operations")
    ledger_p.add_argument("action", choices=["tail"])
    ledger_p.add_argument("--n", type=int, default=10, metavar="N")

    # bench
    bench_p = sub.add_parser("bench", help="Run throughput benchmark on a model")
    bench_p.add_argument("--model", default="default", metavar="MODEL_ID")

    # auction
    auction_p = sub.add_parser("auction", help="Auction operations")
    auction_sub = auction_p.add_subparsers(dest="auction_action")
    bid_p = auction_sub.add_parser("bid", help="Bid on a resource")
    bid_p.add_argument("resource", help="Resource type (e.g. gpu_vram, cpu_threads)")
    bid_p.add_argument("votes", type=int, help="Number of votes to cast")

    args = parser.parse_args()

    # Override gateway URL if provided
    global GATEWAY
    GATEWAY = args.gateway

    dispatch = {
        "status": cmd_status,
        "backends": cmd_backends,
        "evolve": cmd_evolve,
        "kairos": cmd_kairos,
        "ledger": cmd_ledger,
        "bench": cmd_bench,
    }

    if args.command == "auction":
        if args.auction_action == "bid":
            cmd_auction_bid(args)
        else:
            auction_p.print_help()
    elif args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
