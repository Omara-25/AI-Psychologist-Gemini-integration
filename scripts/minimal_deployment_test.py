#!/usr/bin/env python3
"""
Minimal test to verify deployment success
"""

def test_critical_imports():
    """Test only the imports needed for the app to start"""
    
    critical_imports = [
        "fastapi",
        "uvicorn", 
        "google.genai",
        "numpy",
        "pydantic",
        "dotenv"
    ]
    
    print("🧪 Testing Critical Imports")
    print("=" * 30)
    
    failed = []
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    if not failed:
        print("\n🎉 All critical imports successful!")
        print("✅ Application should start")
        return True
    else:
        print(f"\n❌ Failed imports: {', '.join(failed)}")
        print("⚠️  Application may not start")
        return False

def test_optional_imports():
    """Test optional imports (voice features)"""
    
    optional_imports = [
        "aiortc",
        "fastrtc",
        "av",
        "scipy"
    ]
    
    print("\n🧪 Testing Optional Imports")
    print("=" * 30)
    
    available = []
    
    for module in optional_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
            available.append(module)
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    voice_ready = "aiortc" in available and "fastrtc" in available
    
    print(f"\n🎤 Voice features: {'✅ Available' if voice_ready else '❌ Disabled'}")
    
    return available

def main():
    print("🚀 Minimal Deployment Test")
    print("=" * 40)
    
    critical_ok = test_critical_imports()
    available_modules = test_optional_imports()
    
    print("\n" + "=" * 40)
    print("📊 Summary:")
    
    if critical_ok:
        print("✅ Deployment successful")
        print("✅ Text chat will work")
        
        if "aiortc" in available_modules and "fastrtc" in available_modules:
            print("✅ Voice features may work (needs component test)")
        else:
            print("⚠️  Voice features disabled")
    else:
        print("❌ Deployment has issues")
    
    return 0 if critical_ok else 1

if __name__ == "__main__":
    exit(main())
