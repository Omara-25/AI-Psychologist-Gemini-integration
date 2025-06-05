#!/usr/bin/env python3
"""
Test FastRTC components after deployment (not during build)
"""

import sys
import traceback

def test_fastrtc_components():
    """Test FastRTC components with detailed error reporting"""
    print("🧪 Testing FastRTC Components")
    print("=" * 40)
    
    # Test basic FastRTC import
    try:
        import fastrtc
        version = getattr(fastrtc, '__version__', 'unknown')
        print(f"✅ FastRTC import: OK (version: {version})")
    except ImportError as e:
        print(f"❌ FastRTC import failed: {e}")
        return False
    
    # Test core components
    try:
        from fastrtc import AsyncStreamHandler, Stream
        print("✅ FastRTC core components: OK")
    except ImportError as e:
        print(f"❌ FastRTC core components failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False
    
    # Test tracks module (this is where it usually fails)
    try:
        from fastrtc.tracks import EmitType, StreamHandler
        print("✅ FastRTC tracks module: OK")
    except ImportError as e:
        print(f"❌ FastRTC tracks module failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False
    
    # Test get_cloudflare_turn_credentials_async
    try:
        from fastrtc import get_cloudflare_turn_credentials_async
        print("✅ FastRTC TURN credentials: OK")
    except ImportError as e:
        print(f"❌ FastRTC TURN credentials failed: {e}")
        return False
    
    print("\n🎉 All FastRTC components working!")
    return True

def test_aiortc_components():
    """Test aiortc components that FastRTC needs"""
    print("\n🧪 Testing aiortc Components")
    print("=" * 40)
    
    try:
        import aiortc
        version = getattr(aiortc, '__version__', 'unknown')
        print(f"✅ aiortc import: OK (version: {version})")
    except ImportError as e:
        print(f"❌ aiortc import failed: {e}")
        return False
    
    # Test specific components FastRTC needs
    components = [
        ("AudioStreamTrack", "aiortc.AudioStreamTrack"),
        ("VideoStreamTrack", "aiortc.VideoStreamTrack"),
        ("RTCPeerConnection", "aiortc.RTCPeerConnection"),
    ]
    
    for component_name, import_path in components:
        try:
            module_path, class_name = import_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {component_name}: OK")
        except (ImportError, AttributeError) as e:
            print(f"❌ {component_name}: {e}")
            return False
    
    return True

def test_compatibility():
    """Test if FastRTC and aiortc work together"""
    print("\n🔗 Testing FastRTC + aiortc Compatibility")
    print("=" * 40)
    
    if not test_aiortc_components():
        print("❌ aiortc components not available")
        return False
    
    if not test_fastrtc_components():
        print("❌ FastRTC components not available")
        return False
    
    print("✅ FastRTC and aiortc are compatible!")
    return True

def main():
    print("🔍 Post-Deployment FastRTC Test")
    print("=" * 50)
    
    compatibility_ok = test_compatibility()
    
    print("\n" + "=" * 50)
    print("📊 Final Assessment:")
    
    if compatibility_ok:
        print("✅ Voice features should work!")
        print("🎤 Real-time voice chat available")
        print("💬 Text chat available")
    else:
        print("⚠️  Voice features will be disabled")
        print("💬 Text chat will work perfectly")
        print("🔧 Application will use fallback mode")
    
    print("\n📝 Recommendations:")
    if compatibility_ok:
        print("• Test voice chat functionality")
        print("• Check WebRTC connection")
        print("• Verify microphone permissions")
    else:
        print("• Use text chat mode")
        print("• Consider alternative voice implementation")
        print("• Check deployment logs for details")
    
    return 0 if compatibility_ok else 1

if __name__ == "__main__":
    sys.exit(main())
