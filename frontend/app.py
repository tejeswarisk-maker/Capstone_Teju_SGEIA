"""
app.py — Opens the SGEIA dashboard in the browser.
Run after the backend is already started:
  python frontend/app.py
"""
import webbrowser
import time

print("Opening dashboard at http://127.0.0.1:8000/")
print("Make sure backend is running: uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload")

time.sleep(1)
webbrowser.open("http://127.0.0.1:8000/")
