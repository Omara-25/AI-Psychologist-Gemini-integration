# Quick test script to verify the Stream initialization fix
# Save this as test_streams.py and run it to verify the fix

import os
os.environ["GEMINI_API_KEY"] = "test_key"  # Set a dummy key for testing

try:
    from fastrtc import Stream, AsyncStreamHandler, AsyncAudioVideoStreamHandler
    print("✅ FastRTC imports successful")
    
    # Test the handler classes
    class TestAudioHandler(AsyncStreamHandler):
        def __init__(self):
            super().__init__("mono", 24000, 16000)
        def copy(self):
            return TestAudioHandler()
        async def start_up(self):
            pass
        async def receive(self, frame):
            pass
        async def emit(self):
            return None
        def shutdown(self):
            pass
    
    class TestVideoHandler(AsyncAudioVideoStreamHandler):
        def __init__(self):
            super().__init__("mono", 24000, 16000)
        def copy(self):
            return TestVideoHandler()
        async def start_up(self):
            pass
        async def receive(self, frame):
            pass
        async def emit(self):
            return None
        async def video_receive(self, frame):
            pass
        async def video_emit(self):
            return None
        async def shutdown(self):
            pass
    
    # Test audio stream
    print("Testing audio stream...")
    audio_stream = Stream(
        modality="audio",
        mode="send-receive",
        handler=TestAudioHandler(),
        concurrency_limit=5,
        time_limit=300,
    )
    print("✅ Audio stream created successfully")
    
    # Test video stream with send-receive mode
    print("Testing video stream...")
    video_stream = Stream(
        modality="audio-video", 
        mode="send-receive",  # Using send-receive instead of send-only
        handler=TestVideoHandler(),
        concurrency_limit=3,
        time_limit=300,
    )
    print("✅ Video stream created successfully")
    print("🎉 All streams initialized correctly!")
    
except ImportError as e:
    print(f"❌ FastRTC import error: {e}")
    print("📝 This is expected if FastRTC is not installed")
except Exception as e:
    print(f"❌ Stream creation error: {e}")
    print("🔧 This indicates the configuration issue")