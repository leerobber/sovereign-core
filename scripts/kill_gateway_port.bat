@echo off
setlocal
set PORT=%~1
if "%PORT%"=="" set PORT=8000

echo Clearing port %PORT% (python/uvicorn only)...

for /f "tokens=5" %%A in ('netstat -ano 2^>nul ^| findstr "0.0.0.0:%PORT% " ^| findstr LISTENING') do (
    if not "%%A"=="0" (
        tasklist /FI "PID eq %%A" 2>nul | findstr /i "python.exe uvicorn" >nul && (
            echo   Killing PID %%A on 0.0.0.0:%PORT%
            taskkill /F /PID %%A >nul 2>&1
        )
    )
)

timeout /t 1 /nobreak >nul
endlocal