#!/usr/bin/env python3
"""
Script to run both Streamlit app and FastAPI server simultaneously
Usage: python run_servers.py
"""
import subprocess
import sys
import os
import signal
import time
from threading import Thread

def run_streamlit():
    """Run Streamlit app on port 8503"""
    os.environ['STREAMLIT_SERVER_PORT'] = '8503'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])

def run_fastapi():
    """Run FastAPI server on port 8000"""
    subprocess.run([sys.executable, '-m', 'uvicorn', 'api_server:app', '--reload', '--host', '0.0.0.0', '--port', '8000'])

def main():
    print("🚀 Starting Sentiment Analysis Servers...")
    print("📊 Streamlit App: http://localhost:8503")
    print("🔌 FastAPI Server: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop both servers\n")

    # Start both servers in separate threads
    streamlit_thread = Thread(target=run_streamlit, daemon=True)
    fastapi_thread = Thread(target=run_fastapi, daemon=True)

    try:
        streamlit_thread.start()
        time.sleep(2)  # Give Streamlit time to start
        fastapi_thread.start()

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()