#!/usr/bin/env python3
"""
Simple test script to verify basic functionality
"""

def test_imports():
    """Test all imports with detailed error reporting"""
    
    tests = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("google.genai", "Google Gemini AI"),
        ("numpy", "NumPy arrays"),
        ("scipy", "SciPy scientific computing"),
        ("av", "PyAV audio/video"),
        ("aiortc", "WebRTC library"),
        ("fastrtc", "FastRTC wrapper"),
    ]
    
    results = {}
    
    for module, description in tests:
        try:
            __import__(module)
            results[module] = {"status": "✅", "error": None}
            print(f"✅ {module}: {description}")
        except ImportError as e:
            results[module] = {"status": "❌", "error": str(e)}
            print(f"❌ {module}: {e}")
    
    # Test FastRTC components specifically
    if results["fastrtc"]["status"] == "✅":
        try:
            from fastrtc import AsyncStreamHandler, Stream
            print("✅ FastRTC components: Available")
            results["fastrtc_components"] = {"status": "✅", "error": None}
        except ImportError as e:
            print(f"❌ FastRTC components: {e}")
            results["fastrtc_components"] = {"status": "❌", "error": str(e)}
    
    return results

def main():
    print("🧪 Simple Import Test")
    print("=" * 30)
    
    results = test_imports()
    
    print("\n📊 Results:")
    
    # Check critical components
    critical = ["fastapi", "uvicorn", "google.genai"]
    critical_ok = all(results[mod]["status"] == "✅" for mod in critical)
    
    # Check voice components
    voice = ["aiortc", "fastrtc"]
    voice_ok = all(results[mod]["status"] == "✅" for mod in voice if mod in results)
    
    print(f"Critical components: {'✅ OK' if critical_ok else '❌ FAILED'}")
    print(f"Voice components: {'✅ OK' if voice_ok else '❌ FAILED'}")
    
    if critical_ok:
        print("\n🎉 Application should start successfully!")
        if voice_ok:
            print("🎤 Voice features should work!")
        else:
            print("💬 Text chat will work (voice features disabled)")
    else:
        print("\n⚠️  Application may have issues starting")
    
    return 0

if __name__ == "__main__":
    exit(main())
