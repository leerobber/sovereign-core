import subprocess, sys, os, time, urllib.request, json

PYTHON = r"C:\Users\leer4\AppData\Local\Programs\Python\Python314\python.exe"
ROOT   = r"C:\Users\leer4\sovereign-core"

def kill_port(port):
    result = subprocess.run('netstat -aon', shell=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if f':{port} ' in line:
            parts = line.split()
            if parts and parts[-1].isdigit():
                pid = int(parts[-1])
                if pid > 4:
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
                    print(f"  Killed PID {pid} on port {port}")

def check(url, label):
    try:
        urllib.request.urlopen(url, timeout=5)
        print(f"  {label}: OK")
        return True
    except Exception as e:
        print(f"  {label}: FAIL — {e}")
        return False

print("="*52)
print("  SOVEREIGN CORE — FULL STACK")
print("  Gateway:9000 + GH05T3:8006 + Chat:7000")
print("="*52)

print("\n[1] Clearing ports 9000, 8006, 7000...")
kill_port(9000); kill_port(8006); kill_port(7000)
time.sleep(2)

print("\n[2] Starting Gateway on port 9000...")
gw = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "gateway.main:app",
     "--host", "0.0.0.0", "--port", "9000", "--loop", "asyncio"],
    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"  PID: {gw.pid}")
time.sleep(6)
if gw.poll() is not None:
    print("  ERROR: Gateway crashed!"); sys.exit(1)

print("\n[3] Starting GH05T3 agent on port 8006...")
gh = subprocess.Popen(
    [PYTHON, r"ghost_protocol\ghost_agent.py", "--serve"],
    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"  PID: {gh.pid}")
time.sleep(4)
if gh.poll() is not None:
    print("  ERROR: GH05T3 crashed!"); sys.exit(1)

print("\n[4] Starting Chat Server on port 7000...")
cs = subprocess.Popen(
    [PYTHON, "chat_server_gh05t3.py"],
    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"  PID: {cs.pid}")
time.sleep(3)
if cs.poll() is not None:
    print("  ERROR: Chat server crashed!"); sys.exit(1)

print("\n[5] Verifying all services...")
check("http://127.0.0.1:9000/health", "Gateway  (9000)")
check("http://127.0.0.1:8006/status", "GH05T3   (8006)")
check("http://127.0.0.1:7000/chat",   "Chat Srv (7000)")

print(f"""
{'='*52}
  ALL SYSTEMS UP — GH05T3 IS ONLINE

  Open this file in your browser:
  C:\\Users\\leer4\\Desktop\\GH05T3.html

  Gateway : http://127.0.0.1:9000
  GH05T3  : http://127.0.0.1:8006
  Chat    : http://127.0.0.1:7000
{'='*52}

Press Ctrl+C to stop all services.
""")

try:
    gw.wait()
except KeyboardInterrupt:
    print("\nShutting down all services...")
    gw.terminate(); gh.terminate(); cs.terminate()
    print("Done.")
