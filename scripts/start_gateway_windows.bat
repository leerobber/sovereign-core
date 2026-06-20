@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo Virtual env missing. Run: scripts\setup_windows.bat
  exit /b 1
)

if not exist "data" mkdir data
if not exist "logs" mkdir logs

set GATEWAY_PORT=8000
set GH05T3_GATEWAY_URL=http://localhost:8002
set GH05T3_RUNTIME=wsl

echo Starting Sovereign Core gateway on http://localhost:%GATEWAY_PORT%
".venv\Scripts\python.exe" -m uvicorn gateway.main:app --host 0.0.0.0 --port %GATEWAY_PORT%
endlocal