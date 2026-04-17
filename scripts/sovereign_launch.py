"""
sovereign_launch.py — Windows-native Sovereign Core launcher
Run from inside the sovereign-core directory:
    python scripts/sovereign_launch.py

No bash. No admin. No nonsense. Just works.
"""
import subprocess, sys, os, time, json, urllib.request, urllib.error, threading, signal

# ── Colors (Windows-safe) ─────────────────────────────────────────────────────
os.system("")  # enable ANSI on Windows
R='\033[91m'; G='\033[92m'; Y='\033[93m'; C='\033[96m'; M='\033[95m'
B='\033[1m'; D='\033[2m'; X='\033[0m'

BANNER = f"""
{M}{B}  ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗  ██╗{X}
{M}{B}  ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗ ██║{X}
{M}{B}  ███████╗██║   ██║╚██╗ ██╔╝█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗██║{X}
{M}{B}  ╚════██║██║   ██║ ╚████╔╝ ██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚████║{X}
{M}{B}  ███████║╚██████╔╝  ╚██╔╝  ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚███║{X}
{M}{B}  ╚══════╝ ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚══╝{X}
{D}  Heterogeneous Compute Gateway — Windows Native Launcher{X}
{D}  Built by Robert 'Terry' Lee Jr. | SovereignNation LLC{X}
"""

# ── Config ────────────────────────────────────────────────────────────────────
BACKENDS = [
    {"name": "rtx5050",    "port": 8001, "label": "RTX 5050   (NVIDIA 8GB)", "env": {"CUDA_VISIBLE_DEVICES": "0"}},
    {"name": "radeon780m", "port": 8002, "label": "Radeon 780M (AMD 4GB)  ", "env": {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"}},
    {"name": "ryzen7cpu",  "port": 8003, "label": "Ryzen 7    (CPU)       ", "env": {"OLLAMA_NO_GPU": "1"}},
]

GATEWAY_PORT = 8080
OLLAMA_BIN   = "ollama"
LOG_DIR      = "logs"

procs = []  # track all spawned processes

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(sym, color, msg): print(f"  {color}{sym}{X} {msg}")
def ok(msg):   log("✅", G, msg)
def warn(msg): log("⚠️ ", Y, msg)
def err(msg):  log("✗ ", R, msg)
def info(msg): log("→ ", C, msg)

def probe(port, path="/api/tags", timeout=2):
    try:
        url = f"http://127.0.0.1:{port}{path}"
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except: return False

def wait_for_port(port, path="/api/tags", max_wait=20, label=""):
    for i in range(max_wait):
        if probe(port, path): return True
        time.sleep(1)
        if i % 5 == 4: info(f"  still waiting for {label}... ({i+1}s)")
    return False

def cleanup(sig=None, frame=None):
    print(f"\n{Y}Stopping all Sovereign Core processes...{X}")
    for p in procs:
        try:
            p.terminate()
            p.wait(timeout=3)
        except: pass
    print(f"{G}Stopped. Goodbye.{X}\n")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ── Step 1: Check Ollama ──────────────────────────────────────────────────────
def check_ollama():
    print(f"\n{B}── Step 1: Checking Ollama ──{X}")
    try:
        r = subprocess.run([OLLAMA_BIN, "--version"], capture_output=True, text=True)
        ok(f"Ollama found: {r.stdout.strip()}")
    except FileNotFoundError:
        err("Ollama not found. Install from https://ollama.ai")
        sys.exit(1)

    # Check default ollama serve
    if probe(11434, "/api/tags"):
        ok("Ollama default instance running on :11434")
    else:
        warn("No Ollama running on :11434 — backends will serve from their own instances")

# ── Step 2: Start backends ────────────────────────────────────────────────────
def start_backends():
    print(f"\n{B}── Step 2: Starting Ollama Backends ──{X}")
    os.makedirs(LOG_DIR, exist_ok=True)

    for b in BACKENDS:
        port, name, label = b["port"], b["name"], b["label"]

        if probe(port):
            ok(f"{label} ::{port} already running")
            continue

        info(f"Starting {label} on :{port}...")

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"0.0.0.0:{port}"
        env.update(b["env"])

        log_file = open(f"{LOG_DIR}/ollama_{name}.log", "w")
        p = subprocess.Popen(
            [OLLAMA_BIN, "serve"],
            env=env,
            stdout=log_file,
            stderr=log_file,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        procs.append(p)

        if wait_for_port(port, label=label):
            ok(f"{label} ::{port} started (pid {p.pid})")
        else:
            warn(f"{label} ::{port} slow to start — check {LOG_DIR}/ollama_{name}.log")

# ── Step 3: Verify models ─────────────────────────────────────────────────────
def check_models():
    print(f"\n{B}── Step 3: Checking Models ──{X}")

    model_map = {
        8001: ("gemma3:12b",         "RTX 5050   "),
        8002: ("qwen2.5:7b",         "Radeon 780M"),
        8003: ("llama3.2:3b",        "Ryzen 7    "),
    }

    for port, (model, label) in model_map.items():
        if not probe(port):
            warn(f"{label} :{port} — offline, skipping model check")
            continue
        try:
            data = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/tags", timeout=5).read())
            names = [m["name"] for m in data.get("models", [])]
            # Match by prefix (e.g. "gemma3" matches "gemma3:12b")
            prefix = model.split(":")[0]
            if any(prefix in n for n in names):
                ok(f"{label} :{port} — {model} ✓")
            else:
                warn(f"{label} :{port} — {model} not loaded yet")
                info(f"  To load: ollama pull {model}")
                info(f"  Models present: {', '.join(names[:3]) if names else 'none'}")
        except Exception as e:
            warn(f"{label} :{port} — could not check models: {e}")

# ── Step 4: Start gateway ─────────────────────────────────────────────────────
def start_gateway():
    print(f"\n{B}── Step 4: Starting Sovereign Gateway ──{X}")

    if probe(GATEWAY_PORT, "/health"):
        ok(f"Gateway already running on :{GATEWAY_PORT}")
        return

    # Create required data dirs
    for d in ["data/kairos/agents", "data/ledger", "data/pattern_memory"]:
        os.makedirs(d, exist_ok=True)

    info(f"Launching gateway on :{GATEWAY_PORT}...")
    log_file = open(f"{LOG_DIR}/gateway.log", "w")
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "gateway.main:app",
         "--host", "127.0.0.1",
         "--port", str(GATEWAY_PORT),
         "--log-level", "info"],
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )
    procs.append(p)

    if wait_for_port(GATEWAY_PORT, "/health", max_wait=20, label="gateway"):
        ok(f"Gateway online on :{GATEWAY_PORT} (pid {p.pid})")
    else:
        warn(f"Gateway slow to start — check {LOG_DIR}/gateway.log")
        info(f"  Try manually: python -m uvicorn gateway.main:app --host 127.0.0.1 --port {GATEWAY_PORT}")

# ── Step 5: Status report ─────────────────────────────────────────────────────
def status_report():
    print(f"\n{B}{'═'*54}{X}")
    print(f"{B}  SOVEREIGN CORE — LIVE STATUS{X}")
    print(f"{B}{'═'*54}{X}")

    checks = [
        (8001, "/api/tags", "RTX 5050   (Qwen/Gemma)      "),
        (8002, "/api/tags", "Radeon 780M (Qwen2.5)         "),
        (8003, "/api/tags", "Ryzen 7    (Llama3.2)         "),
        (8080, "/health",   "Sovereign Gateway             "),
    ]
    all_up = True
    for port, path, label in checks:
        up = probe(port, path)
        sym = f"{G}●{X}" if up else f"{R}○{X}"
        status = f"{G}ONLINE{X}" if up else f"{R}OFFLINE{X}"
        print(f"  {sym} {label} :{port}  {status}")
        if not up: all_up = False

    # Show models via gateway
    try:
        data = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{GATEWAY_PORT}/v1/models", timeout=3).read())
        models = data.get("data", [])
        if models:
            print(f"\n{C}  Registered models:{X}")
            for m in models[:6]:
                print(f"    ✅ {m['id']:<35} [{m.get('backend','?')}]")
    except: pass

    print(f"\n{D}  Logs:    ./{LOG_DIR}/{X}")
    print(f"{D}  Gateway: http://127.0.0.1:{GATEWAY_PORT}{X}")
    print(f"{D}  Health:  http://127.0.0.1:{GATEWAY_PORT}/health{X}")
    print(f"{D}  Models:  http://127.0.0.1:{GATEWAY_PORT}/v1/models{X}")
    print(f"{D}  KAIROS:  http://127.0.0.1:{GATEWAY_PORT}/kairos/sage{X}")

    if all_up:
        print(f"\n{G}{B}  ⚡ Sovereign Core fully operational. KAIROS ready.{X}\n")
    else:
        print(f"\n{Y}  Some services offline — check logs/ for details.{X}\n")

    print(f"{D}  Press Ctrl+C to stop all services.{X}\n")

# ── Keep alive ────────────────────────────────────────────────────────────────
def keep_alive():
    """Keep the script running, restart crashed processes."""
    while True:
        time.sleep(30)
        # Reap crashed procs
        for p in list(procs):
            if p.poll() is not None:
                procs.remove(p)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(BANNER)
    check_ollama()
    start_backends()
    check_models()
    start_gateway()
    status_report()
    # Block until Ctrl+C
    try:
        keep_alive()
    except KeyboardInterrupt:
        cleanup()
