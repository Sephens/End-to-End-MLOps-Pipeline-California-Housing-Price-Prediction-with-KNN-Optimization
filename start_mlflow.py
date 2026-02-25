"""
start_mlflow.py — Render-compatible MLflow server launcher.
Reads the PORT env var that Render injects and passes it to mlflow server.
"""
import os
import subprocess
import sys

port = os.environ.get("PORT", "8080")
host = "housing-mlflow.onrender.com"

cmd = [
    sys.executable, "-m", "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", str(port),
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "./mlartifacts",
    "--allowed-hosts", host,
    "--cors-allowed-origins", f"https://{host}",
    "--workers", "1",
]

print(f"Starting MLflow on port {port} ...")
subprocess.run(cmd)