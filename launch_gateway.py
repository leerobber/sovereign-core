import subprocess, sys, os, time

os.chdir(r"C:\Users\leer4\sovereign-core")
log_path = r"C:\Users\leer4\sovereign-core\gateway_run.log"
pid_path = r"C:\Users\leer4\sovereign-core\gateway.pid"

# Kill whatever is on port 9000 before we try to bind
print("Clearing port 9000...")
result = subprocess.run(
    'netstat -aon | findstr ":9000 "',
    shell=True, capture_output=True, text=True
)
for line in result.stdout.splitlines():
    parts = line.split()
    if parts and parts[-1].isdigit():
        pid = int(parts[-1])
        if pid > 4:
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
            print(f"  Killed PID {pid} on port 9000")

# Also kill any previous gateway PID we saved
if os.path.exists(pid_path):
    try:
        old_pid = int(open(pid_path).read().strip())
        subprocess.run(f"taskkill /F /PID {old_pid}", shell=True, capture_output=True)
        print(f"  Killed previous gateway PID {old_pid}")
    except:
        pass

time.sleep(2)

cmd = [sys.executable, "-m", "uvicorn", "gateway.main:app",
       "--host", "0.0.0.0", "--port", "9000", "--loop", "asyncio"]

print(f"Starting gateway on port 9000...")
print(f"Log: {log_path}")

with open(log_path, "w") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                            cwd=r"C:\Users\leer4\sovereign-core")
    print(f"Gateway PID: {proc.pid}")
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    time.sleep(5)
    if proc.poll() is None:
        print("Gateway is running!")
    else:
        print(f"Gateway exited with code: {proc.returncode}")
        with open(log_path) as f:
            print(f.read())
