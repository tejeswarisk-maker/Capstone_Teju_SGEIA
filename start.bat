@echo off
echo Starting SGEIA Dashboard...
echo.
echo Backend + Frontend will be available at: http://127.0.0.1:8000/
echo.
cd /d %~dp0
call venv\Scripts\activate.bat
start "" "http://127.0.0.1:8000/"
timeout /t 2 /nobreak >nul
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
