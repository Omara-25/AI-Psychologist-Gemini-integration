#!/usr/bin/env python3
"""
Verify installation after deployment
"""

import sys
import traceback

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import fastapi
        import uvicorn
        import google.genai
        print("✅ Core imports: OK")
        return True
    except ImportError as e:
        print(f"❌ Core imports failed: {e}")
        return False

def test_audio_imports():
    """Test audio-related imports"""
    print("🧪 Testing audio imports...")
    
    try:
        import numpy
        import scipy
        import av
        print("✅ Audio/video imports: OK")
        return True
    except ImportError as e:
        print(f"❌ Audio/video imports failed: {e}")
        return False

def test_aiortc():
    """Test aiortc import and components"""
    print("🧪 Testing aiortc...")
    
    try:
        import aiortc
        print(f"✅ aiortc import: OK (version: {getattr(aiortc, '__version__', 'unknown')})")
        
        # Test specific components
        from aiortc import AudioStreamTrack, VideoStreamTrack, RTCPeerConnection
        print("✅ aiortc components: OK")
        return True
    except ImportError as e:
        print(f"❌ aiortc failed: {e}")
        return False

def test_fastrtc():
    """Test FastRTC import and components"""
    print("🧪 Testing FastRTC...")
    
    try:
        import fastrtc
        print(f"✅ FastRTC import: OK (version: {getattr(fastrtc, '__version__', 'unknown')})")
        
        # Test core components
        from fastrtc import AsyncStreamHandler, Stream
        print("✅ FastRTC core components: OK")
        
        # Test tracks module (this is where it usually fails)
        from fastrtc.tracks import EmitType, StreamHandler
        print("✅ FastRTC tracks module: OK")
        
        return True
    except ImportError as e:
        print(f"❌ FastRTC failed: {e}")
        print(f"   Full error: {traceback.format_exc()}")
        return False

def main():
    print("🔍 Post-Installation Verification")
    print("=" * 50)
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("Audio imports", test_audio_imports),
        ("aiortc", test_aiortc),
        ("FastRTC", test_fastrtc),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    # Overall assessment
    critical_tests = ["Basic imports", "Audio imports"]
    critical_passed = all(results[test] for test in critical_tests if test in results)
    
    voice_tests = ["aiortc", "FastRTC"]
    voice_passed = all(results[test] for test in voice_tests if test in results)
    
    print(f"\n🎯 Assessment:")
    if critical_passed:
        print("✅ Application will start successfully")
        print("✅ Text chat will work")
        
        if voice_passed:
            print("✅ Voice chat will work")
        else:
            print("⚠️  Voice chat will be disabled (fallback mode)")
    else:
        print("❌ Application may have startup issues")
    
    return 0 if critical_passed else 1

if __name__ == "__main__":
    sys.exit(main())
