#!/usr/bin/env python3
"""
Script to start a local MLflow tracking server.

Usage:
    python start_mlflow_server.py [--port 5000] [--host 127.0.0.1]
"""

import argparse
import subprocess
import sys
import os

def start_mlflow_server(host="127.0.0.1", port=5000, backend_store_uri=None):
    """Start MLflow tracking server."""
    
    # Default backend store URI to local directory
    if backend_store_uri is None:
        backend_store_uri = f"file://{os.path.abspath('./mlruns')}"
    
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store_uri,
    ]
    
    print(f"Starting MLflow server on {host}:{port}")
    print(f"Backend store: {backend_store_uri}")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop the server")
    print(f"Once started, open http://{host}:{port} in your browser")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nMLflow server stopped.")
    except FileNotFoundError:
        print("Error: MLflow not found. Please install it with: pip install mlflow")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Start MLflow tracking server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the server to")
    parser.add_argument("--backend-store-uri", help="Backend store URI (default: local file store)")
    
    args = parser.parse_args()
    
    start_mlflow_server(
        host=args.host,
        port=args.port,
        backend_store_uri=args.backend_store_uri
    )

if __name__ == "__main__":
    main()
