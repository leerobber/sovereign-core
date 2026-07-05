@echo off
setlocal
set AETHYRO_SKIP_LICENSE=1
set GH05T3=C:\Users\leer4\GH05T3
set PY=%GH05T3%\backend\.venv\Scripts\python.exe
cd /d %GH05T3%\backend

powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://localhost:8001/api/health' -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul
if errorlevel 1 (
  echo Starting GH05T3 backend :8001 ...
  start "gh05t3-backend" /MIN "%PY%" -m uvicorn server:app --host 0.0.0.0 --port 8001
)

powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://localhost:8002/health' -TimeoutSec 2).StatusCode } catch { 0 }" | findstr 200 >nul
if errorlevel 1 (
  echo Starting GH05T3 gateway :8002 ...
  start "gh05t3-gateway" /MIN "%PY%" -m uvicorn gateway_v3:app --host 0.0.0.0 --port 8002
)

echo GH05T3 launch commands sent. Wait ~15s then check http://localhost:8002/health
endlocal