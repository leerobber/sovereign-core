@echo off
setlocal
cd /d "%~dp0.."

echo === Sovereign Core — Windows setup ===
echo.

where py >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.11+ from python.org
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    py -3 -m venv .venv
  )
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q

echo.
echo Done. Start the gateway with:
echo   scripts\start_gateway_windows.bat
echo Or in PowerShell:
echo   .\.venv\Scripts\Activate.ps1
echo   uvicorn gateway.main:app --host 0.0.0.0 --port 8000
endlocal