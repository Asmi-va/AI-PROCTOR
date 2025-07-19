#!/usr/bin/env python3
"""
AI Interview Proctoring Backend Startup Script
This script starts both the Python FastAPI backend and Node.js backend
"""

import subprocess
import sys
import time
import os
import shutil  # ✅ NEW: Needed for safe npm detection
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python dependencies
    try:
        import fastapi
        import uvicorn
        import torch
        import cv2
        import ultralytics
        print("✅ Python dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Please install dependencies with: pip install -r ai-backend/model/requirements.txt")
        return False
    
    # Check if Node.js is installed
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js is installed: {result.stdout.strip()}")
        else:
            print("❌ Node.js is not installed")
            return False
    except FileNotFoundError:
        print("❌ Node.js is not installed")
        return False
    
    return True

def install_node_dependencies():
    """Install Node.js dependencies if needed"""
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return False

    package_json = backend_dir / "package.json"
    if not package_json.exists():
        print("❌ package.json not found in backend directory")
        return False

    print(f"📦 Installing Node.js dependencies from: {backend_dir.resolve()}")

    # ✅ Get full path to npm binary
    npm_path = shutil.which("npm")
    if not npm_path:
        print("❌ npm command not found in PATH")
        return False

    try:
        subprocess.run([npm_path, 'install'], cwd=backend_dir, check=True)
        print("✅ Node.js dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Node.js dependencies: {e}")
        return False

def start_backends():
    """Start both backends"""
    print(" Starting AI Interview Proctoring backends...")

    # Start Node.js backend
    print(" Starting Node.js backend (port 3001)...")
    node_process = subprocess.Popen(
        ['node', 'server.js'],
        cwd='backend',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait a moment for Node.js to start
    time.sleep(2)

    # Start Python FastAPI backend
    print(" Starting Python FastAPI backend (port 8000)...")
    python_process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'main_api:app', '--host', '0.0.0.0', '--port', '8000', '--reload'],
        cwd='ai-backend/model',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print("\n" + "="*60)
    print(" Backends are starting up!")
    print("="*60)
    print(" Frontend URLs:")
    print("   • Candidate Interface: http://localhost:3001/ai-interview-proctoring/")
    print("   • Interviewer Live View: http://localhost:3001/ai-interview-proctoring/interviewer_live.html")
    print("   • Interviewer Upload: http://localhost:3001/ai-interview-proctoring/interviewer_upload.html")
    print("   • Interviewer Logs: http://localhost:3001/ai-interview-proctoring/interviewer.html")
    print("\n🔧 Backend URLs:")
    print("   • Python API: http://localhost:8000")
    print("   • API Documentation: http://localhost:8000/docs")
    print("   • Node.js API: http://localhost:3001")
    print("\n💡 Press Ctrl+C to stop all backends")
    print("="*60)

    try:
        # Wait for both processes
        node_process.wait()
        python_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping backends...")
        node_process.terminate()
        python_process.terminate()

        # Wait for processes to terminate
        try:
            node_process.wait(timeout=5)
            python_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing processes...")
            node_process.kill()
            python_process.kill()

        print("✅ Backends stopped")

def main():
    """Main function"""
    print("🤖 AI Interview Proctoring Backend Startup")
    print("="*50)

    # Check if we're in the right directory
    if not Path("ai-backend").exists() or not Path("backend").exists():
        print("❌ Please run this script from the project root directory")
        print("   (where ai-backend/ and backend/ folders are located)")
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    # Install Node.js dependencies
    if not install_node_dependencies():
        return 1

    # Start backends
    start_backends()
    return 0

if __name__ == "__main__":
    sys.exit(main())
