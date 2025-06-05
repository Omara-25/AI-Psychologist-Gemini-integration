#!/usr/bin/env python3
"""
Force install FastRTC with all dependencies
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Force Installing FastRTC Dependencies")
    print("=" * 50)
    
    commands = [
        ("pip uninstall fastrtc aiortc -y", "Removing existing packages"),
        ("pip install --upgrade pip setuptools wheel", "Upgrading build tools"),
        ("pip install --no-cache-dir numpy==1.26.4", "Installing NumPy"),
        ("pip install --no-cache-dir scipy==1.11.4", "Installing SciPy"),
        ("pip install --no-cache-dir av==10.0.0", "Installing PyAV"),
        ("pip install --no-cache-dir aiortc==1.5.0", "Installing aiortc"),
        ("pip install --no-cache-dir fastrtc==0.0.24", "Installing FastRTC"),
    ]
    
    success_count = 0
    for cmd, desc in commands:
        if run_command(cmd, desc):
            success_count += 1
        else:
            print(f"⚠️  Continuing despite failure in: {desc}")
    
    print(f"\n📊 Installation Summary: {success_count}/{len(commands)} successful")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_commands = [
        ("python -c 'import aiortc; print(f\"aiortc: {aiortc.__version__}\")'", "Testing aiortc"),
        ("python -c 'import fastrtc; print(f\"fastrtc: {fastrtc.__version__}\")'", "Testing fastrtc"),
        ("python -c 'from fastrtc import AsyncStreamHandler, Stream; print(\"FastRTC components OK\")'", "Testing FastRTC components"),
    ]
    
    for cmd, desc in test_commands:
        run_command(cmd, desc)
    
    print("\n✅ Force installation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
