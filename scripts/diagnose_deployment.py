#!/usr/bin/env python3
"""
Comprehensive deployment diagnostics for FastRTC issues
"""

import sys
import subprocess
import os
import platform

def run_command(cmd):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def check_system_info():
    """Check system information"""
    print("🖥️  System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
def check_system_dependencies():
    """Check if required system packages are installed"""
    print("\n🔧 System Dependencies:")
    
    # Check for required system libraries
    system_deps = [
        "pkg-config --version",
        "ffmpeg -version",
        "which gcc",
        "which g++",
        "which make"
    ]
    
    for cmd in system_deps:
        code, stdout, stderr = run_command(cmd)
        if code == 0:
            print(f"✅ {cmd.split()[0]}: Available")
        else:
            print(f"❌ {cmd.split()[0]}: Missing or failed")

def check_python_packages():
    """Check Python package installation"""
    print("\n📦 Python Packages:")
    
    packages = [
        "fastapi",
        "uvicorn", 
        "google-genai",
        "numpy",
        "scipy",
        "av",
        "aiortc",
        "fastrtc"
    ]
    
    for package in packages:
        try:
            __import__(package)
            # Get version if possible
            try:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                print(f"✅ {package}: {version}")
            except:
                print(f"✅ {package}: installed")
        except ImportError as e:
            print(f"❌ {package}: {e}")

def test_fastrtc_specific():
    """Test FastRTC specific functionality"""
    print("\n🎯 FastRTC Specific Tests:")
    
    try:
        import fastrtc
        print(f"✅ FastRTC import: Success (version: {getattr(fastrtc, '__version__', 'unknown')})")
        
        # Test specific imports that were failing
        try:
            from fastrtc import AsyncStreamHandler, Stream, get_cloudflare_turn_credentials_async
            print("✅ FastRTC core components: Success")
        except ImportError as e:
            print(f"❌ FastRTC core components: {e}")
            
        try:
            from fastrtc.tracks import EmitType, StreamHandler
            print("✅ FastRTC tracks module: Success")
        except ImportError as e:
            print(f"❌ FastRTC tracks module: {e}")
            
    except ImportError as e:
        print(f"❌ FastRTC import: {e}")

def test_aiortc_specific():
    """Test aiortc specific functionality"""
    print("\n🎯 aiortc Specific Tests:")
    
    try:
        import aiortc
        print(f"✅ aiortc import: Success (version: {getattr(aiortc, '__version__', 'unknown')})")
        
        # Test specific components FastRTC needs
        try:
            from aiortc import AudioStreamTrack, VideoStreamTrack, RTCPeerConnection
            print("✅ aiortc core components: Success")
        except ImportError as e:
            print(f"❌ aiortc core components: {e}")
            
        # Test the specific import that was failing
        try:
            from aiortc.mediastreams import AudioStreamTrack
            print("✅ aiortc AudioStreamTrack: Success")
        except ImportError as e:
            print(f"❌ aiortc AudioStreamTrack: {e}")
            
    except ImportError as e:
        print(f"❌ aiortc import: {e}")

def check_environment_variables():
    """Check required environment variables"""
    print("\n🔑 Environment Variables:")
    
    required_vars = ["GEMINI_API_KEY", "PORT"]
    optional_vars = ["GOOGLE_CLOUD_PROJECT", "USE_VERTEX_AI"]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if "KEY" in var or "SECRET" in var:
                print(f"✅ {var}: [REDACTED]")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"ℹ️  {var}: {value}")
        else:
            print(f"ℹ️  {var}: Not set (optional)")

def suggest_fixes():
    """Suggest potential fixes based on findings"""
    print("\n🔧 Suggested Fixes:")
    print("1. Try installing with specific versions:")
    print("   pip uninstall fastrtc aiortc -y")
    print("   pip install aiortc==1.5.0")
    print("   pip install fastrtc==0.0.24")
    print()
    print("2. Install system dependencies:")
    print("   apt-get update && apt-get install -y build-essential pkg-config")
    print("   apt-get install -y portaudio19-dev libasound2-dev libpulse-dev")
    print("   apt-get install -y ffmpeg libavcodec-dev libavformat-dev")
    print()
    print("3. Alternative: Use text-only mode")
    print("   Set FASTRTC_AVAILABLE=False in environment")

def main():
    print("🔍 Railway Deployment Diagnostics")
    print("=" * 60)
    
    check_system_info()
    check_system_dependencies()
    check_python_packages()
    test_aiortc_specific()
    test_fastrtc_specific()
    check_environment_variables()
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("📊 Diagnostic Complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
