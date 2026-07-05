@echo off
REM Run this file as Administrator (right-click - Run as administrator)
echo Removing stale WSL portproxy on localhost:8000...
netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=8000
echo.
echo Remaining port proxies:
netsh interface portproxy show all
echo.
echo Done. Restart: scripts\start_gateway_windows.bat
pause