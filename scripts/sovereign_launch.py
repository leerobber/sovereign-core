"""
scripts/sovereign_launch.py — Windows-native Sovereign Core launcher v2
Adds: watchdog auto-restart, .env loading, model verification, DB init
Run: python scripts/sovereign_launch.py
"""
import subprocess, sys, os, time, json, urllib.request, urllib.error
import threading, signal, sqlite3
from pathlib import Path

# ── Load .env before anything else ───────────────────────────────────────────
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# ── Colors ────────────────────────────────────────────────────────────────────
os.system("")
R='\033[91m'; G='\033[92m'; Y='\033[93m'; C='\033[96m'; M='\033[95m'
B='\033[1m'; D='\033[2m'; X='\033[0m'; W='\033[97m'

BANNER = f"""
{M}{B}  ╔═══════════════════════════════════════════════════════════╗{X}
{M}{B}  ║       SOVEREIGN CORE — HETEROGENEOUS COMPUTE GATEWAY      ║{X}
{M}{B}  ║              Self-Improving Agent Infrastructure           ║{X}
{M}{B}  ╚═══════════════════════════════════════════════════════════╝{X}
{D}  RTX 5050 · Radeon 780M · Ryzen 7  |  KAIROS · SAGE · Iron Dome{X}
{D}  Built by Robert 'Terry' Lee Jr.  |  SovereignNation LLC{X}
"""

# ── Backend config ────────────────────────────────────────────────────────────
BACKENDS = [
    {
        "name": "rtx5050",
        "port": 8001,
        "label": "RTX 5050    (NVIDIA 8GB)",
        "model": "gemma3:12b",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
    },
    {
        "name": "radeon780m",
        "port": 8002,
        "label": "Radeon 780M (AMD 4GB)  ",
        "model": "qwen2.5:7b",
        "env": {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"},
    },
    {
        "name": "ryzen7cpu",
        "port": 8003,
        "label": "Ryzen 7     (CPU)      ",
        "model": "llama3.2:3b",
        "env": {"CUDA_VISIBLE_DEVICES": "-1", "OLLAMA_NO_GPU": "1"},
    },
]

GATEWAY_PORT  = int(os.environ.get("GATEWAY_PORT", "8080"))
OLLAMA_BIN    = "ollama"
LOG_DIR       = Path("logs")
DATA_DIR      = Path("data")
DB_PATH       = DATA_DIR / "sovereign.db"

procs: list[dict] = []  # {name, port, proc, log_file, env}
_running = True

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(sym, color, msg): print(f"  {color}{sym}{X} {msg}", flush=True)
def ok(msg):    log("✅", G, msg)
def warn(msg):  log("⚠ ", Y, msg)
def err(msg):   log("✗ ", R, msg)
def info(msg):  log("→ ", C, msg)
def head(msg):  print(f"\n{B}── {msg} ──{X}", flush=True)

def probe(port, path="/api/tags", timeout=2):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout)
        return True
    except: return False

def wait_for(port, path="/api/tags", max_wait=25, label=""):
    for i in range(max_wait):
        if probe(port, path): return True
        time.sleep(1)
        if i > 0 and i % 5 == 0: info(f"  waiting for {label or port}... {i}s")
    return False

# ── Step 0: Init directories + DB ────────────────────────────────────────────
def init_dirs():
    head("Initializing Directories & Database")
    for d in ["data/kairos/agents", "data/ledger", "data/pattern_memory", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Touch sovereign.db so the gateway finds it on boot
    if not DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
        ok(f"Created {DB_PATH}")
    else:
        ok(f"Database exists: {DB_PATH} ({DB_PATH.stat().st_size // 1024}KB)")

# ── Step 1: Check Ollama ──────────────────────────────────────────────────────
def check_ollama():
    head("Checking Ollama")
    try:
        r = subprocess.run([OLLAMA_BIN, "--version"], capture_output=True, text=True)
        ok(f"Ollama: {r.stdout.strip()}")
    except FileNotFoundError:
        err("Ollama not found — install from https://ollama.ai")
        sys.exit(1)

# ── Step 2: Pull missing models ───────────────────────────────────────────────
def ensure_models():
    head("Verifying Models")
    # Get all installed models from default Ollama (11434)
    installed = set()
    try:
        data = json.loads(urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags", timeout=5).read())
        installed = {m["name"].split(":")[0] for m in data.get("models", [])}
    except: pass

    required = {b["model"] for b in BACKENDS}
    for model in sorted(required):
        prefix = model.split(":")[0]
        if prefix in installed or model.split(":")[0] in installed:
            ok(f"{model} ✓ installed")
        else:
            warn(f"{model} not found — pulling now...")
            subprocess.run([OLLAMA_BIN, "pull", model])

# ── Step 3: Start backends ────────────────────────────────────────────────────
def start_backend(b: dict) -> dict | None:
    port, name, label = b["port"], b["name"], b["label"]
    if probe(port):
        ok(f"{label} :{port} already running")
        return None

    info(f"Starting {label} on :{port}...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
    env.update(b["env"])

    LOG_DIR.mkdir(exist_ok=True)
    lf = open(LOG_DIR / f"ollama_{name}.log", "w")
    p = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        env=env, stdout=lf, stderr=lf,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=="win32" else 0
    )
    rec = {"name": name, "port": port, "label": label,
           "proc": p, "log_file": lf, "env": b["env"], "model": b["model"]}

    if wait_for(port, label=label):
        ok(f"{label} :{port} started (pid {p.pid})")
        return rec
    else:
        warn(f"{label} :{port} slow — check logs/ollama_{name}.log")
        return rec

def start_backends():
    head("Starting Ollama Backends")
    for b in BACKENDS:
        rec = start_backend(b)
        if rec: procs.append(rec)

# ── Step 4: Start gateway ─────────────────────────────────────────────────────
def start_gateway():
    head("Starting Sovereign Gateway")
    if probe(GATEWAY_PORT, "/health"):
        ok(f"Gateway already running on :{GATEWAY_PORT}")
        return

    lf = open(LOG_DIR / "gateway.log", "w")
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "gateway.main:app",
         "--host", "127.0.0.1", "--port", str(GATEWAY_PORT), "--log-level", "info"],
        stdout=lf, stderr=lf,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=="win32" else 0
    )
    procs.append({"name": "gateway", "port": GATEWAY_PORT, "label": "Gateway",
                  "proc": p, "log_file": lf, "env": {}, "model": None})

    if wait_for(GATEWAY_PORT, "/health", label="gateway"):
        ok(f"Gateway online :{GATEWAY_PORT} (pid {p.pid})")
    else:
        warn(f"Gateway slow — check logs/gateway.log")
        info(f"  Manual: python -m uvicorn gateway.main:app --host 127.0.0.1 --port {GATEWAY_PORT}")

# ── Step 5: Status board ──────────────────────────────────────────────────────
def status_board():
    print(f"\n{B}{'═'*60}{X}")
    print(f"{B}  SOVEREIGN CORE — SYSTEM STATUS{X}")
    print(f"{B}{'═'*60}{X}")
    checks = [
        (8001, "/api/tags", "RTX 5050    (gemma3:12b)       "),
        (8002, "/api/tags", "Radeon 780M (qwen2.5:7b)       "),
        (8003, "/api/tags", "Ryzen 7     (llama3.2:3b)      "),
        (GATEWAY_PORT, "/health", "Sovereign Gateway              "),
    ]
    all_up = True
    for port, path, label in checks:
        up = probe(port, path)
        sym = f"{G}●{X}" if up else f"{R}○{X}"
        st  = f"{G}ONLINE {X}" if up else f"{R}OFFLINE{X}"
        print(f"  {sym} {label} :{port}  {st}")
        if not up: all_up = False

    # Gateway routes
    print(f"\n{C}  Endpoints:{X}")
    print(f"  {D}Health    → http://127.0.0.1:{GATEWAY_PORT}/health{X}")
    print(f"  {D}Models    → http://127.0.0.1:{GATEWAY_PORT}/v1/models{X}")
    print(f"  {D}KAIROS    → http://127.0.0.1:{GATEWAY_PORT}/kairos/sage{X}")
    print(f"  {D}Dashboard → http://127.0.0.1:{GATEWAY_PORT}/dashboard{X}")
    print(f"  {D}Metrics   → http://127.0.0.1:{GATEWAY_PORT}/metrics{X}")
    print(f"  {D}WebSocket → ws://127.0.0.1:{GATEWAY_PORT}/ws/events{X}")
    print(f"\n  {D}Logs: ./logs/   |   DB: {DB_PATH}{X}")

    if all_up:
        print(f"\n{G}{B}  ⚡ Sovereign Core FULLY OPERATIONAL. KAIROS ready.{X}")
    else:
        print(f"\n{Y}  Some services offline — check logs/ for details.{X}")
    print(f"\n{D}  Watchdog active — crashed backends auto-restart.{X}")
    print(f"{D}  Press Ctrl+C to stop all services.{X}\n")

# ── Watchdog ──────────────────────────────────────────────────────────────────
def watchdog():
    """Monitor and auto-restart crashed processes."""
    OLLAMA_BACKENDS = [p for p in procs if p["name"] != "gateway"]
    while _running:
        time.sleep(10)
        for rec in OLLAMA_BACKENDS:
            p = rec["proc"]
            if p and p.poll() is not None:  # process died
                warn(f"Backend {rec['name']} crashed (exit {p.returncode}) — restarting...")
                env = os.environ.copy()
                env["OLLAMA_HOST"] = f"0.0.0.0:{rec['port']}"
                env.update(rec["env"])
                lf = open(LOG_DIR / f"ollama_{rec['name']}.log", "a")
                new_p = subprocess.Popen(
                    [OLLAMA_BIN, "serve"], env=env, stdout=lf, stderr=lf,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform=="win32" else 0
                )
                rec["proc"] = new_p
                rec["log_file"] = lf
                if wait_for(rec["port"], label=rec["name"]):
                    ok(f"Backend {rec['name']} restarted (pid {new_p.pid})")
                else:
                    err(f"Backend {rec['name']} failed to restart — check logs")

# ── Cleanup ───────────────────────────────────────────────────────────────────
def cleanup(sig=None, frame=None):
    global _running
    _running = False
    print(f"\n{Y}Stopping all Sovereign Core services...{X}", flush=True)
    for rec in procs:
        try:
            rec["proc"].terminate()
            rec["proc"].wait(timeout=5)
        except: pass
        try: rec["log_file"].close()
        except: pass
    print(f"{G}All services stopped. Goodbye.{X}\n")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(BANNER)
    init_dirs()
    check_ollama()
    ensure_models()
    start_backends()
    start_gateway()
    status_board()

    # Start watchdog in background
    wt = threading.Thread(target=watchdog, daemon=True)
    wt.start()

    # Block main thread
    try:
        while _running: time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
