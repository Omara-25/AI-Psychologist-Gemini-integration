#!/usr/bin/env python3
"""
Check which version of aiortc is actually installed
"""

def check_aiortc_version():
    """Check aiortc version and available components"""
    
    print("🔍 Checking aiortc Installation")
    print("=" * 35)
    
    try:
        import aiortc
        version = getattr(aiortc, '__version__', 'unknown')
        print(f"✅ aiortc version: {version}")
        
        # Check if AudioStreamTrack is available
        try:
            from aiortc import AudioStreamTrack
            print("✅ AudioStreamTrack: Available")
        except ImportError as e:
            print(f"❌ AudioStreamTrack: Not available - {e}")
            return False
        
        # Check other components
        components = [
            "VideoStreamTrack",
            "RTCPeerConnection", 
            "RTCDataChannel"
        ]
        
        for component in components:
            try:
                getattr(aiortc, component)
                print(f"✅ {component}: Available")
            except AttributeError:
                print(f"⚠️  {component}: Not found")
        
        return True
        
    except ImportError as e:
        print(f"❌ aiortc not installed: {e}")
        return False

def check_fastrtc_compatibility():
    """Check if FastRTC can work with current aiortc"""
    
    print("\n🔗 Checking FastRTC Compatibility")
    print("=" * 35)
    
    if not check_aiortc_version():
        return False
    
    try:
        import fastrtc
        version = getattr(fastrtc, '__version__', 'unknown')
        print(f"✅ fastrtc version: {version}")
        
        # Try to import the problematic components
        try:
            from fastrtc import AsyncStreamHandler, Stream
            print("✅ FastRTC core components: OK")
        except ImportError as e:
            print(f"❌ FastRTC core components: {e}")
            return False
        
        # Try the tracks module that usually fails
        try:
            from fastrtc.tracks import EmitType, StreamHandler
            print("✅ FastRTC tracks module: OK")
        except ImportError as e:
            print(f"❌ FastRTC tracks module: {e}")
            return False
        
        print("\n🎉 FastRTC is fully compatible!")
        return True
        
    except ImportError as e:
        print(f"❌ FastRTC not installed: {e}")
        return False

def main():
    print("🧪 aiortc Version Check")
    print("=" * 40)
    
    compatible = check_fastrtc_compatibility()
    
    print("\n" + "=" * 40)
    print("📊 Assessment:")
    
    if compatible:
        print("✅ Voice features should work!")
        print("🎤 Real-time voice chat available")
    else:
        print("⚠️  Voice features will be disabled")
        print("💬 Text chat will work perfectly")
        print("\n🔧 Possible solutions:")
        print("• Force rebuild with cache break")
        print("• Try different aiortc version")
        print("• Use text-only mode")
    
    return 0 if compatible else 1

if __name__ == "__main__":
    exit(main())
