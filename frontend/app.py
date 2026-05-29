"""
app.py — Starts the SGEIA backend and opens the dashboard in the browser.
Run from project root: python frontend/app.py
"""
import subprocess
import sys
import time
import webbrowser
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Starting SGEIA backend...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app",
     "--host", "127.0.0.1", "--port", "8000"],
)

time.sleep(3)
print("Opening dashboard at http://127.0.0.1:8000/")
webbrowser.open("http://127.0.0.1:8000/")

try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    print("\nSGEIA stopped.")
