@echo off
cd /d %~dp0
call venv\Scripts\activate.bat

REM Kill any existing streamlit processes
taskkill /f /im "streamlit.exe" >nul 2>&1

REM Start FastAPI backend (backend + frontend in one)
start "SGEIA Backend" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && uvicorn src.api.main:app --host 127.0.0.1 --port 8000"

REM Wait for backend, then open browser directly to port 8000
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000/"
