#!/usr/bin/env python3
"""
Script to test FastRTC and aiortc compatibility
"""

import sys
import subprocess
import importlib

def test_import(module_name):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}: Import successful")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"❌ {module_name}: Unexpected error - {e}")
        return False

def test_fastrtc_components():
    """Test specific FastRTC components"""
    try:
        from fastrtc import AsyncStreamHandler, Stream, get_cloudflare_turn_credentials_async
        print("✅ FastRTC core components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ FastRTC components failed: {e}")
        return False

def test_aiortc_components():
    """Test specific aiortc components that FastRTC needs"""
    try:
        from aiortc import AudioStreamTrack, VideoStreamTrack, RTCPeerConnection
        print("✅ aiortc core components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ aiortc components failed: {e}")
        return False

def get_package_version(package_name):
    """Get the version of an installed package"""
    try:
        result = subprocess.run([sys.executable, "-c", 
                               f"import {package_name}; print({package_name}.__version__)"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown"
    except Exception:
        return "unknown"

def main():
    print("🔍 Testing FastRTC and aiortc compatibility")
    print("=" * 50)
    
    # Test basic imports
    print("\n📦 Testing basic imports:")
    aiortc_ok = test_import("aiortc")
    fastrtc_ok = test_import("fastrtc")
    
    # Get versions
    print("\n📋 Package versions:")
    aiortc_version = get_package_version("aiortc")
    fastrtc_version = get_package_version("fastrtc")
    print(f"aiortc: {aiortc_version}")
    print(f"fastrtc: {fastrtc_version}")
    
    # Test specific components
    print("\n🔧 Testing specific components:")
    if aiortc_ok:
        aiortc_components_ok = test_aiortc_components()
    else:
        aiortc_components_ok = False
        
    if fastrtc_ok:
        fastrtc_components_ok = test_fastrtc_components()
    else:
        fastrtc_components_ok = False
    
    # Test compatibility
    print("\n🔗 Testing compatibility:")
    if aiortc_ok and fastrtc_ok:
        try:
            # This is the specific import that was failing
            from fastrtc.tracks import EmitType, StreamHandler
            print("✅ FastRTC tracks module imported successfully")
            compatibility_ok = True
        except ImportError as e:
            print(f"❌ FastRTC tracks import failed: {e}")
            compatibility_ok = False
    else:
        compatibility_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"aiortc import: {'✅' if aiortc_ok else '❌'}")
    print(f"fastrtc import: {'✅' if fastrtc_ok else '❌'}")
    print(f"aiortc components: {'✅' if aiortc_components_ok else '❌'}")
    print(f"fastrtc components: {'✅' if fastrtc_components_ok else '❌'}")
    print(f"Compatibility: {'✅' if compatibility_ok else '❌'}")
    
    if compatibility_ok:
        print("\n✅ All tests passed! Voice features should work.")
    else:
        print("\n⚠️  Compatibility issues detected. Voice features may be limited.")
        print("\n🔧 Recommended fixes:")
        print("1. Try: pip install aiortc==1.5.0 fastrtc==0.0.24")
        print("2. Or use the fallback text-only mode")
    
    return 0 if compatibility_ok else 1

if __name__ == "__main__":
    sys.exit(main())
