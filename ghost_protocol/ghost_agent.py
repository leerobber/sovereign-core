"""
GH05T3 Ghost Agent - Sovereign Core Companion
Usage: python ghost_agent.py --pair-code <6-digit>
       python ghost_agent.py --serve
       python ghost_agent.py --status
"""
from __future__ import annotations
import argparse, asyncio, hashlib, hmac, json, logging, os, subprocess, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GHOST_DIR      = Path(__file__).parent
SESSION_FILE   = GHOST_DIR / "ghost_sessions.json"
LOG_FILE       = GHOST_DIR / "ghost_agent.log"
GATEWAY_URL    = os.environ.get("SOVEREIGN_GATEWAY", "http://127.0.0.1:9000")
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
COMPANION_PORT = int(os.environ.get("GHOST_PORT", "8006"))
_SESSION_SECRET = os.environ.get("GHOST_SESSION_SECRET", "GH05T3_SOVEREIGN_2026")
MODEL_SLOTS = {"proposer": "qwen2.5:7b", "verifier": "deepseek-coder:6.7b", "critic": "llama3.1"}

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [GHOST] %(levelname)-8s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(str(LOG_FILE), encoding="utf-8")])
log = logging.getLogger("ghost_agent")

def free_port(port: int):
    """Kill whatever process is holding this port before we bind."""
    try:
        result = subprocess.run(f'netstat -aon | findstr :{port}',
            shell=True, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[-1].isdigit():
                pid = int(parts[-1])
                if pid > 4:  # never kill system processes
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                    log.info(f"Freed port {port} — killed PID {pid}")
    except Exception as e:
        log.warning(f"Could not free port {port}: {e}")

def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try: return loop.run_until_complete(coro)
    finally: loop.close()

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, Dict] = {}
        if SESSION_FILE.exists():
            try: self._sessions = json.loads(SESSION_FILE.read_text())
            except: pass

    def _save(self):
        try: SESSION_FILE.write_text(json.dumps(self._sessions, indent=2))
        except: pass

    def _sign(self, p: str) -> str:
        return hmac.new(_SESSION_SECRET.encode(), p.encode(), hashlib.sha256).hexdigest()

    def create_session(self, pair_code: str, client_id: str = "") -> Dict:
        sid = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps({"session_id": sid, "pair_code": pair_code,
                              "client_id": client_id, "issued_at": now}, sort_keys=True)
        token = f"{sid}.{self._sign(payload)}"
        s = {"session_id": sid, "token": token, "pair_code": pair_code,
             "client_id": client_id, "issued_at": now,
             "expires_at": time.time() + 86400, "active": True}
        self._sessions[sid] = s; self._save()
        log.info(f"Session: {sid[:8]}... code={pair_code}")
        return s

    def list_active(self) -> List[Dict]:
        now = time.time()
        return [{"session_id": s["session_id"][:8]+"...", "pair_code": s["pair_code"],
                 "issued_at": s["issued_at"]}
                for s in self._sessions.values()
                if s.get("active") and now < s.get("expires_at", 0)]

class PairCodeValidator:
    def __init__(self): self._authorized: Dict[str, float] = {}

    def authorize(self, code: str, ttl: int = 600):
        if not (len(code) == 6 and code.isdigit()): raise ValueError(f"Must be 6 digits: '{code}'")
        self._authorized[code] = time.time() + ttl

    def validate(self, code: str) -> Tuple[bool, str]:
        if not code: return False, "EMPTY"
        if not (len(code) == 6 and code.isdigit()): return False, "INVALID_FORMAT"
        expiry = self._authorized.get(code)
        if expiry:
            if time.time() > expiry: return False, "EXPIRED"
            return True, "OK"
        if os.environ.get("GHOST_DEV_MODE", "true").lower() == "true": return True, "OK_DEV"
        return False, "NOT_AUTHORIZED"

async def check_gateway() -> Dict:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{GATEWAY_URL}/health")
            return {"ok": r.status_code == 200, "url": GATEWAY_URL}
    except Exception as e: return {"ok": False, "url": GATEWAY_URL, "error": str(e)}

async def check_ollama() -> Dict:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return {"ok": True, "url": OLLAMA_URL, "models": models, "count": len(models)}
            return {"ok": False, "url": OLLAMA_URL, "error": f"HTTP {r.status_code}"}
    except Exception as e: return {"ok": False, "url": OLLAMA_URL, "error": str(e)}

async def check_both(): return await asyncio.gather(check_gateway(), check_ollama())

async def resolve_slots(available: List[str]) -> Dict[str, str]:
    resolved = {}
    for slot, preferred in MODEL_SLOTS.items():
        if preferred in available: resolved[slot] = preferred; continue
        matches = [m for m in available if m.startswith(preferred.split(":")[0])]
        resolved[slot] = matches[0] if matches else preferred
    return resolved

session_mgr    = SessionManager()
code_validator = PairCodeValidator()

try:
    import httpx
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI(title="GH05T3 Ghost Agent", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.post("/pair")
    async def pair(request: Request):
        body  = await request.json()
        code  = str(body.get("pair_code", "")).strip()
        client = str(body.get("client_id", "emergent")).strip()
        valid, reason = code_validator.validate(code)
        if not valid: raise HTTPException(401, f"Invalid pair code: {reason}")
        gw, ol = await asyncio.gather(check_gateway(), check_ollama())
        available = ol.get("models", [])
        slots = await resolve_slots(available)
        session = session_mgr.create_session(code, client)
        return JSONResponse({"status": "PAIRED", "session_id": session["session_id"],
            "token": session["token"], "node_id": "TatorTot", "node_type": "sovereign_core",
            "backends": {"gateway": {"url": GATEWAY_URL, "health": gw["ok"]},
                         "ollama": {"url": OLLAMA_URL, "health": ol["ok"], "model_count": ol.get("count",0)}},
            "models": {"slots": slots, "available": available},
            "capabilities": {"inference": gw["ok"], "ollama_direct": ol["ok"],
                             "kairos": True, "ghost_recall": True},
            "companion_version": "1.0.0"})

    @app.post("/v1/chat/completions")
    async def proxy(request: Request, x_ghost_token: Optional[str] = Header(None)):
        body = await request.json()
        try:
            async with httpx.AsyncClient(timeout=120.0) as c:
                r = await c.post(f"{GATEWAY_URL}/v1/chat/completions", json=body)
                return JSONResponse(r.json(), status_code=r.status_code)
        except Exception as e: raise HTTPException(502, f"Gateway unreachable: {e}")

    @app.get("/status")
    async def status():
        gw, ol = await asyncio.gather(check_gateway(), check_ollama())
        return JSONResponse({"ghost_agent": "UP", "gateway": gw, "ollama": ol,
            "active_sessions": session_mgr.list_active(),
            "timestamp": datetime.now(timezone.utc).isoformat()})

    @app.get("/models")
    async def get_models():
        ol = await check_ollama()
        available = ol.get("models", [])
        return JSONResponse({"available": available, "slots": await resolve_slots(available)})

    FASTAPI_OK = True
except ImportError as e:
    FASTAPI_OK = False
    log.warning(f"FastAPI unavailable: {e}")

def cli_pair(code: str):
    print(f"\n{'='*60}\n  GH05T3 - Pairing\n{'='*60}")
    print(f"  Code: {code} | Gateway: {GATEWAY_URL} | Ollama: {OLLAMA_URL}\n{'-'*60}")
    try: code_validator.authorize(code)
    except ValueError as e: print(f"\n  FAIL: {e}"); sys.exit(1)
    session = session_mgr.create_session(code, "cli")
    gw, ol  = run_async(check_both())
    available = ol.get("models", [])
    slots = run_async(resolve_slots(available))
    print(f"\n  Gateway : {'OK' if gw['ok'] else 'OFFLINE - '+str(gw.get('error',''))}")
    print(f"  Ollama  : {'OK' if ol['ok'] else 'OFFLINE - '+str(ol.get('error',''))}")
    print(f"  Models  : {len(available)}\n")
    for slot, model in slots.items():
        print(f"  [{slot:10s}] {model:40s} {'OK' if model in available else 'NEEDS PULL'}")
    needs = [m for m in MODEL_SLOTS.values() if m not in available]
    if needs:
        print(f"\n  Pull missing:"); [print(f"    ollama pull {m}") for m in needs]
    print(f"\n  Session: {session['session_id'][:16]}...")
    print(f"  Token  : {session['token'][:32]}...")
    print(f"\n  PAIRED - GH05T3 ready on TatorTot\n{'='*60}\n")

def cli_status():
    gw, ol = run_async(check_both())
    sessions = session_mgr.list_active()
    print(f"\n{'='*60}\n  GH05T3 Status\n{'='*60}")
    print(f"  Gateway : {'OK' if gw['ok'] else 'OFFLINE'} ({GATEWAY_URL})")
    print(f"  Ollama  : {'OK' if ol['ok'] else 'OFFLINE'} ({OLLAMA_URL})")
    print(f"  Models  : {ol.get('count',0)} | Sessions: {len(sessions)}")
    for s in sessions: print(f"    - {s['session_id']} | {s['issued_at'][:19]}")
    print(f"{'='*60}\n")

def cli_serve():
    if not FASTAPI_OK: print("ERROR: pip install fastapi uvicorn httpx"); sys.exit(1)
    free_port(COMPANION_PORT)
    time.sleep(1)
    print(f"\n  GH05T3 Companion starting on port {COMPANION_PORT}")
    print(f"  Gateway: {GATEWAY_URL} | Ollama: {OLLAMA_URL}\n")
    uvicorn.run(app, host="127.0.0.1", port=COMPANION_PORT, log_level="info")

def main():
    parser = argparse.ArgumentParser(description="GH05T3 Ghost Agent")
    parser.add_argument("--pair-code",   metavar="CODE")
    parser.add_argument("--status",      action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--serve",       action="store_true")
    parser.add_argument("--gateway",     default=None)
    parser.add_argument("--ollama",      default=None)
    args = parser.parse_args()
    global GATEWAY_URL, OLLAMA_URL
    if args.gateway: GATEWAY_URL = args.gateway
    if args.ollama:  OLLAMA_URL  = args.ollama
    ran = False
    if args.pair_code:
        cli_pair(args.pair_code); ran = True
    if args.status:
        cli_status(); ran = True
    if args.list_models:
        ol = run_async(check_ollama())
        print("\n  Models:"); [print(f"  {m}") for m in ol.get("models",[])]; print()
        ran = True
    if args.serve:
        cli_serve(); ran = True
    if not ran: parser.print_help()

if __name__ == "__main__":
    main()
