"""
start_prefect.py — Render-compatible Prefect server launcher.
Reads the PORT env var that Render injects and passes it to prefect server start.
"""
import os
import subprocess
import sys

port = os.environ.get("PORT", "8080")

# Also set the API URL env var so the UI connects correctly
api_url = os.environ.get("PREFECT_UI_API_URL", f"http://0.0.0.0:{port}/api")
os.environ["PREFECT_UI_API_URL"] = api_url

cmd = [
    sys.executable, "-m", "prefect", "server", "start",
    "--host", "0.0.0.0",
    "--port", str(port),
]

print(f"Starting Prefect server on port {port} …")
subprocess.run(cmd)
