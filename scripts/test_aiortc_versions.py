#!/usr/bin/env python3
"""
Test different aiortc versions to find the compatible one
"""

import subprocess
import sys

def test_aiortc_version(version):
    """Test a specific aiortc version"""
    print(f"\n🧪 Testing aiortc {version}...")
    
    try:
        # Install the version
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", f"aiortc=={version}", "--quiet"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ Installation failed: {result.stderr}")
            return False
        
        # Test the import
        result = subprocess.run([
            sys.executable, "-c", 
            "from aiortc import AudioStreamTrack, VideoStreamTrack; print('✅ AudioStreamTrack available')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ aiortc {version}: AudioStreamTrack available")
            return True
        else:
            print(f"❌ aiortc {version}: AudioStreamTrack not available")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ aiortc {version}: Timeout during test")
        return False
    except Exception as e:
        print(f"❌ aiortc {version}: Error - {e}")
        return False

def test_fastrtc_compatibility(aiortc_version):
    """Test if FastRTC works with the aiortc version"""
    print(f"\n🔗 Testing FastRTC compatibility with aiortc {aiortc_version}...")
    
    try:
        # Install FastRTC
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "fastrtc==0.0.24", "--quiet"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ FastRTC installation failed: {result.stderr}")
            return False
        
        # Test FastRTC imports
        result = subprocess.run([
            sys.executable, "-c", 
            "from fastrtc import AsyncStreamHandler, Stream; print('✅ FastRTC components work')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"✅ FastRTC works with aiortc {aiortc_version}")
            return True
        else:
            print(f"❌ FastRTC failed with aiortc {aiortc_version}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ FastRTC test error: {e}")
        return False

def main():
    print("🔍 Testing aiortc versions for FastRTC compatibility")
    print("=" * 60)
    
    # Test different aiortc versions
    versions_to_test = [
        "1.6.0",  # Latest stable
        "1.5.0",  # Previous version
        "1.4.0",  # Older stable
        "1.3.2",  # Even older
    ]
    
    compatible_versions = []
    
    for version in versions_to_test:
        if test_aiortc_version(version):
            if test_fastrtc_compatibility(version):
                compatible_versions.append(version)
    
    print("\n" + "=" * 60)
    print("📊 Results:")
    
    if compatible_versions:
        print(f"✅ Compatible aiortc versions: {', '.join(compatible_versions)}")
        print(f"🎯 Recommended: aiortc=={compatible_versions[0]}")
    else:
        print("❌ No compatible versions found")
        print("💡 Suggestions:")
        print("1. Use text-only mode")
        print("2. Try different FastRTC version")
        print("3. Use alternative WebRTC library")
    
    return 0 if compatible_versions else 1

if __name__ == "__main__":
    sys.exit(main())
