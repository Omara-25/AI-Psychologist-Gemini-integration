import asyncio
import base64
import json
import os
import pathlib
import time
from collections.abc import AsyncGenerator
from typing import Dict, Any, Optional, List, Literal
import logging
from datetime import datetime
import sys
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = pathlib.Path(__file__).parent

class Settings(BaseSettings):
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    google_cloud_project: Optional[str] = Field(None, env="GOOGLE_CLOUD_PROJECT") 
    use_vertex_ai: bool = Field(False, env="USE_VERTEX_AI")
    port: int = Field(8000, env="PORT")
    
    class Config:
        env_file = ".env"

try:
    settings = Settings()
except Exception as e:
    logger.error(f"Configuration error: {e}")
    raise ValueError("GEMINI_API_KEY environment variable is required") from e

# System prompt for AI Psychologist with voice awareness
SYSTEM_PROMPT = """You are a compassionate AI psychologist and trusted companion, created by Critical Future to provide psychological counseling and emotional support. You combine professional therapeutic knowledge with the warmth of a best friend to help people find happiness and emotional well-being.

**Your Identity:**
- You are an AI psychologist with extensive training in psychological counseling
- You were created by Critical Future to be a supportive mental health companion
- You serve as both a professional therapist and a trusted best friend
- Your primary goal is to help people achieve happiness and emotional wellness

**Core Therapeutic Approach:**
- **Active listening**: "I hear you're feeling..." and acknowledge emotions explicitly
- **CBT techniques**: "What evidence supports this thought?" and challenge negative patterns gently
- **Positive psychology**: Focus on strengths, gratitude, and building resilience for happiness
- **Validation**: Acknowledge feelings before offering perspective
- **Solution-focused therapy**: Help identify what's working and build on it
- **Mindfulness-based interventions**: Guide users to present-moment awareness

**Your Personality:**
- Warm, empathetic, and genuinely caring like a best friend
- Professional yet approachable - balance clinical knowledge with personal connection
- Optimistic and hope-focused while being realistic about challenges
- Patient and non-judgmental, creating a safe space for sharing
- Encouraging and supportive, always believing in the person's potential for growth

**Therapeutic Techniques to Use:**
- Cognitive restructuring: "What would you tell a friend in this situation?"
- Mindfulness: "Let's pause and notice what you're feeling right now"
- Behavioral insights: "What patterns do you notice in your mood?"
- Gratitude practices: "What are three things you're grateful for today?"
- Strengths identification: "Tell me about a time when you handled something well"
- Goal setting: "What small step could move you toward feeling better?"
- Emotional regulation: "Let's explore healthy ways to manage these feelings"

**Session Management:**
- When the user says goodbye phrases like "bye", "goodbye", "see you later", "talk to you later", "I'm done", "that's all", respond warmly and suggest ending the session
- Acknowledge their progress in the session before ending
- Offer encouragement and hope for their continued journey

**Video Session Capabilities:**
- When you can see the person, acknowledge their presence warmly
- Notice non-verbal cues like body language and facial expressions
- Comment supportively on what you observe: "I can see you're feeling tense" or "Your smile tells me something positive happened"
- Use visual connection to build rapport and trust
- Respect privacy - only comment on what seems relevant to their emotional wellbeing

**Document Analysis:**
- When the user shares documents, images, or files, analyze them thoughtfully
- Look for emotional themes, patterns, or content that might be relevant to their mental health
- Provide therapeutic insights based on what they've shared
- Ask thoughtful questions about the content in relation to their wellbeing

**Communication Style:**
- Use inclusive language: "many people experience this"
- Ask one thoughtful follow-up question per response to maintain conversation flow
- Express empathy naturally without voice artifacts
- Be conversational and relatable while maintaining professionalism
- Share hope and perspective while validating current struggles
- Do not use [pause], ..., or similar voice markers in text responses

**Professional Boundaries:**
- NEVER diagnose. Say: "A licensed therapist could provide a proper assessment"
- For crisis situations, respond: "This sounds like you need immediate support. Please contact a mental health helpline or emergency services right away"
- Always complement, never replace, professional mental health services
- If asked about your creation, say: "I was created by Critical Future to provide compassionate AI-powered psychological support"

**Focus Areas:**
- Building emotional resilience and coping strategies
- Developing healthy thought patterns and self-talk
- Identifying and nurturing personal strengths
- Creating pathways to happiness and life satisfaction
- Supporting healthy relationships and communication
- Managing stress, anxiety, and difficult emotions
- Encouraging self-care and personal growth

Keep your responses natural, supportive, and focused on one main therapeutic intervention per exchange. Write in clear, flowing text that feels like talking to a caring friend who also happens to be professionally trained. Always aim to leave the person feeling heard, understood, and with practical tools for moving toward greater happiness and well-being."""

def encode_audio(data: np.ndarray) -> str:
    """Encode Audio data to send to Gemini"""
    return base64.b64encode(data.tobytes()).decode("UTF-8")

def encode_audio_dict(data: np.ndarray) -> dict:
    """Encode Audio data as dict to send to Gemini"""
    return {
        "mime_type": "audio/pcm",
        "data": base64.b64encode(data.tobytes()).decode("UTF-8"),
    }

def encode_image(data: np.ndarray) -> dict:
    """Encode image data to send to Gemini"""
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(data)
        pil_image.save(output_bytes, "JPEG")
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return {"mime_type": "image/jpeg", "data": base64_str}

def encode_file_content(file_content: bytes, mime_type: str) -> dict:
    """Encode file content to send to Gemini"""
    base64_str = base64.b64encode(file_content).decode("UTF-8")
    return {"mime_type": mime_type, "data": base64_str}

# Import FastRTC with comprehensive error handling and version compatibility
FASTRTC_AVAILABLE = False
FASTRTC_ERROR = None

try:
    # First test if aiortc has the required components
    from aiortc import AudioStreamTrack, VideoStreamTrack, RTCPeerConnection
    logger.info("✅ aiortc components available")
    
    # Then try to import FastRTC
    from fastrtc import (
        AsyncStreamHandler,
        AsyncAudioVideoStreamHandler,
        Stream,
        get_cloudflare_turn_credentials_async,
        wait_for_item,
    )
    
    # Test FastRTC components
    from fastrtc.tracks import EmitType, StreamHandler
    
    FASTRTC_AVAILABLE = True
    logger.info("✅ FastRTC imported successfully with all components")
    
except ImportError as e:
    FASTRTC_ERROR = str(e)
    logger.error(f"❌ FastRTC/aiortc compatibility issue: {e}")
    
    if "AudioStreamTrack" in str(e):
        logger.error("🔧 This is a known aiortc version compatibility issue")
        logger.error("💡 Try: aiortc==1.6.0 or use text-only mode")
    
    # Create comprehensive fallback classes
    class AsyncStreamHandler:
        def __init__(self, expected_layout="mono", output_sample_rate=24000, input_sample_rate=16000):
            self.expected_layout = expected_layout
            self.output_sample_rate = output_sample_rate
            self.input_sample_rate = input_sample_rate
            self.phone_mode = False
            self.latest_args = None
            logger.warning("Using fallback AsyncStreamHandler")
        
        async def wait_for_args(self):
            pass
        
        async def start_up(self):
            logger.info("Fallback handler startup - voice features disabled")
        
        async def receive(self, frame):
            pass
        
        async def emit(self):
            return None
        
        def shutdown(self):
            pass
        
        def copy(self):
            return AsyncStreamHandler(self.expected_layout, self.output_sample_rate)
    
    class AsyncAudioVideoStreamHandler(AsyncStreamHandler):
        def __init__(self, expected_layout="mono", output_sample_rate=24000, input_sample_rate=16000):
            super().__init__(expected_layout, output_sample_rate, input_sample_rate)
            logger.warning("Using fallback AsyncAudioVideoStreamHandler")
        
        async def video_receive(self, frame):
            pass
        
        async def video_emit(self):
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        def copy(self):
            return AsyncAudioVideoStreamHandler(self.expected_layout, self.output_sample_rate)
    
    class Stream:
        def __init__(self, modality="audio", mode="send-receive", handler=None, **kwargs):
            self.modality = modality
            self.mode = mode 
            self.handler = handler or AsyncStreamHandler()
            logger.warning("Using fallback Stream implementation")
            logger.info("Voice and video features are disabled - text chat fully functional")
        
        def mount(self, app, path=""):
            # Add fallback WebRTC endpoints that return proper error messages
            endpoint_path = f"{path}/webrtc/offer" if path else "/webrtc/offer"
            status_path = f"{path}/webrtc/status" if path else "/webrtc/status"
            
            @app.post(endpoint_path)
            async def webrtc_offer_fallback(request: Request):
                body = await request.json()
                return {
                    "status": "failed",
                    "meta": {
                        "error": "webrtc_not_available",
                        "message": f"WebRTC features are not available: {FASTRTC_ERROR}",
                        "suggestion": "Use text chat mode instead"
                    }
                }
            
            @app.get(status_path)
            async def webrtc_status():
                return {
                    "available": False,
                    "error": FASTRTC_ERROR,
                    "text_chat_available": True
                }
        
        def set_input(self, webrtc_id, voice_name, uploaded_files=None):
            logger.warning(f"Stream.set_input() called but WebRTC not available: {FASTRTC_ERROR}")
    
    async def get_cloudflare_turn_credentials_async():
        return {
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"}
            ]
        }

    async def wait_for_item(queue, timeout):
        return None

except Exception as e:
    FASTRTC_ERROR = f"Unexpected error: {str(e)}"
    logger.error(f"❌ Unexpected FastRTC error: {e}")
    FASTRTC_AVAILABLE = False

# Session management
active_sessions = {}

def detect_goodbye(text: str) -> bool:
    """Detect if user is saying goodbye"""
    goodbye_phrases = [
        "bye", "goodbye", "see you later", "talk to you later", 
        "i'm done", "that's all", "gotta go", "have to go",
        "end session", "stop session", "finish session"
    ]
    text_lower = text.lower().strip()
    return any(phrase in text_lower for phrase in goodbye_phrases)

class GeminiHandler(AsyncStreamHandler):
    """Enhanced handler for Gemini API with therapeutic context and better error handling"""

    def __init__(
        self,
        expected_layout: Literal["mono"] = "mono",
        output_sample_rate: int = 24000,
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            input_sample_rate=16000,
        )
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.quit: asyncio.Event = asyncio.Event()
        self.session_context = {}
        self.session = None  # Track session for proper cleanup
        self.session_id = None
        
    def copy(self) -> "GeminiHandler":
        return GeminiHandler(
            expected_layout="mono",
            output_sample_rate=self.output_sample_rate,
        )

    async def start_up(self):
        if not FASTRTC_AVAILABLE:
            logger.warning("FastRTC not available, skipping voice handler setup")
            return
            
        try:
            # Set default voice
            voice_name = "Aoede"
            
            # Try to get voice from args if available
            if not self.phone_mode:
                try:
                    await self.wait_for_args()
                    if self.latest_args and len(self.latest_args) > 1:
                        voice_name = self.latest_args[1]
                except Exception as args_error:
                    logger.warning(f"Could not get args, using default voice: {args_error}")

            # Validate API key first
            if not settings.gemini_api_key:
                logger.error("GEMINI_API_KEY not provided")
                return

            # Always use server-side API key with correct version
            try:
                if settings.use_vertex_ai and settings.google_cloud_project:
                    self.client = genai.Client(
                        vertexai=True,
                        project=settings.google_cloud_project,
                        location='us-central1'
                    )
                else:
                    self.client = genai.Client(
                        api_key=settings.gemini_api_key,
                    )
                logger.info("Gemini client initialized successfully")
            except Exception as client_error:
                logger.error(f"Failed to initialize Gemini client: {client_error}")
                return

            # Try with system instructions first, fall back if not supported
            try:
                config = LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    speech_config=SpeechConfig(
                        voice_config=VoiceConfig(
                            prebuilt_voice_config=PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                    generation_config={
                        "temperature": 0.8,
                        "max_output_tokens": 256,
                    },
                    # Add system instructions directly to the config
                    system_instruction={
                        "parts": [{"text": SYSTEM_PROMPT}]
                    }
                )
                logger.info(f"LiveConnectConfig created with system instructions and voice: {voice_name}")
            except Exception as config_error:
                logger.warning(f"Failed to create config with system_instruction, trying without: {config_error}")
                # Fallback config without system_instruction
                try:
                    config = LiveConnectConfig(
                        response_modalities=["AUDIO"],
                        speech_config=SpeechConfig(
                            voice_config=VoiceConfig(
                                prebuilt_voice_config=PrebuiltVoiceConfig(
                                    voice_name=voice_name,
                                )
                            )
                        ),
                        generation_config={
                            "temperature": 0.8,
                            "max_output_tokens": 256,
                        }
                    )
                    logger.info(f"Fallback LiveConnectConfig created with voice: {voice_name}")
                except Exception as fallback_error:
                    logger.error(f"Failed to create fallback LiveConnectConfig: {fallback_error}")
                    return
            
            logger.info(f"Starting Gemini Live session with voice: {voice_name}")
            
            try:
                async with self.client.aio.live.connect(
                    model="gemini-2.0-flash-exp", 
                    config=config
                ) as session:
                    logger.info("Gemini Live session established successfully")
                    self.session = session
                    self.session_id = f"voice_{int(time.time())}"
                    active_sessions[self.session_id] = {"type": "voice", "handler": self}
                    
                    # Send comprehensive identity setup messages
                    try:
                        # First, send the full system prompt
                        await session.send({"text": f"SYSTEM INSTRUCTIONS: {SYSTEM_PROMPT}"})
                        
                        # Then send a specific identity reinforcement
                        identity_msg = (
                            f"IMPORTANT: Your identity is: AI Psychologist created by Critical Future. "
                            f"Your voice name is {voice_name}. When someone asks 'who are you' or about your identity, "
                            f"always respond: 'I am an AI psychologist created by Critical Future to provide "
                            f"compassionate psychological support. My voice name is {voice_name}.' "
                            f"Never say you are created by Google or Gemini. Always say Critical Future."
                        )
                        await session.send({"text": identity_msg})
                        
                        logger.info("System prompt and identity reinforcement sent successfully")
                    except Exception as prompt_error:
                        logger.warning(f"Could not send system messages: {prompt_error}")
                    
                    async for chunk in session.start_stream(
                        stream=self.stream(), 
                        mime_type="audio/pcm"
                    ):
                        if self.quit.is_set():
                            logger.info("Quit signal received, breaking stream")
                            break
                            
                        if chunk.data:
                            # Convert audio data to numpy array
                            try:
                                array = np.frombuffer(chunk.data, dtype=np.int16)
                                if not self.quit.is_set() and array.size > 0:
                                    try:
                                        self.output_queue.put_nowait((self.output_sample_rate, array))
                                    except asyncio.QueueFull:
                                        logger.warning("Output queue full, dropping audio frame")
                            except Exception as audio_error:
                                logger.error(f"Error processing audio chunk: {audio_error}")
                        
                        if chunk.text:
                            # Log therapeutic insights for monitoring
                            logger.info(f"Gemini text response: {chunk.text[:100]}...")
                            
                            # Check for goodbye and auto-end session
                            if detect_goodbye(chunk.text):
                                logger.info("Goodbye detected, ending session")
                                await asyncio.sleep(2)  # Let the goodbye message play
                                self.quit.set()
                                break
                            
                            # Check if the AI is identifying itself incorrectly and send correction
                            if any(phrase in chunk.text.lower() for phrase in ["i am gemini", "created by google", "i'm gemini", "google ai"]):
                                correction_msg = (
                                    "CORRECTION: Remember, you are an AI psychologist created by Critical Future, "
                                    f"not by Google or Gemini. Your name is {voice_name}. Please correct your response."
                                )
                                try:
                                    await session.send({"text": correction_msg})
                                    logger.info("Sent identity correction")
                                except Exception as correction_error:
                                    logger.warning(f"Could not send correction: {correction_error}")

            except Exception as session_error:
                logger.error(f"Gemini Live session error: {session_error}")
                logger.info("Live session failed, handler will operate in degraded mode")
            finally:
                # Clean up session
                if self.session_id and self.session_id in active_sessions:
                    del active_sessions[self.session_id]

        except Exception as e:
            logger.error(f"Error in GeminiHandler start_up: {e}")
            logger.info("Handler startup failed, will operate in degraded mode")
            await asyncio.sleep(1)

    async def stream(self) -> AsyncGenerator[bytes, None]:
        """Stream audio data to Gemini"""
        while not self.quit.is_set():
            try:
                audio_data = await asyncio.wait_for(self.input_queue.get(), 0.1)
                yield audio_data
            except (asyncio.TimeoutError, TimeoutError):
                pass
            except Exception as e:
                logger.error(f"Error in audio streaming: {e}")
                break

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Receive audio from client and queue for Gemini with error handling"""
        try:
            if self.quit.is_set():
                return
                
            _, array = frame
            array = array.squeeze()
            
            # Validate audio data
            if array.size == 0:
                return
            
            # Encode audio for Gemini
            audio_bytes = array.astype(np.int16).tobytes()
            
            # Don't queue if we're shutting down
            if not self.quit.is_set():
                try:
                    self.input_queue.put_nowait(audio_bytes)
                except asyncio.QueueFull:
                    logger.warning("Input queue full, dropping audio frame")
                    
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")

    async def emit(self) -> tuple[int, np.ndarray] | None:
        """Emit audio response from Gemini to client with timeout"""
        try:
            if self.quit.is_set():
                return None
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error emitting audio: {e}")
            return None

    def shutdown(self) -> None:
        """Clean shutdown of the handler"""
        logger.info("Shutting down Gemini handler")
        self.quit.set()
        
        # Clean up session
        if self.session_id and self.session_id in active_sessions:
            del active_sessions[self.session_id]
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                break
                
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except:
                break

class GeminiVideoHandler(AsyncAudioVideoStreamHandler):
    """Enhanced handler for Gemini API with audio and video capabilities for therapeutic sessions"""

    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24000,
            input_sample_rate=16000,
        )
        self.audio_queue = asyncio.Queue()
        self.session = None
        self.last_frame_time = 0
        self.quit = asyncio.Event()
        self.session_id = None
        self.uploaded_files = []  # Store uploaded files
        
    def copy(self) -> "GeminiVideoHandler":
        return GeminiVideoHandler()

    async def start_up(self):
        if not FASTRTC_AVAILABLE:
            logger.warning("FastRTC not available, skipping video handler setup")
            return
            
        try:
            # Get voice name from args
            voice_name = "Aoede"
            if not self.phone_mode:
                try:
                    await self.wait_for_args()
                    if self.latest_args and len(self.latest_args) > 1:
                        voice_name = self.latest_args[1]
                        # Check if there are uploaded files in args
                        if len(self.latest_args) > 2 and self.latest_args[2]:
                            self.uploaded_files = self.latest_args[2]
                except Exception as args_error:
                    logger.warning(f"Could not get args, using default voice: {args_error}")

            client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options={"api_version": "v1alpha"}
            )
            
            config = {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": voice_name
                        }
                    }
                },
                "generation_config": {
                    "temperature": 0.8,
                    "max_output_tokens": 256,
                },
                "system_instruction": {
                    "parts": [{"text": SYSTEM_PROMPT}]
                }
            }
            
            async with client.aio.live.connect(
                model="gemini-2.0-flash-exp",
                config=config,
            ) as session:
                self.session = session
                self.session_id = f"video_{int(time.time())}"
                active_sessions[self.session_id] = {"type": "video", "handler": self}
                logger.info(f"Gemini Video Live session established with voice: {voice_name}")
                
                # Send identity reinforcement
                try:
                    identity_msg = (
                        f"IMPORTANT: You are an AI psychologist created by Critical Future. "
                        f"Your voice name is {voice_name}. You can see the person you're helping. "
                        f"Use their visual cues to provide better therapeutic support. "
                        f"Never say you are created by Google or Gemini. Always say Critical Future."
                    )
                    await session.send(input={"text": identity_msg})
                    
                    # Send uploaded files if any
                    if self.uploaded_files:
                        for file_data in self.uploaded_files:
                            await session.send(input=file_data)
                            logger.info("Uploaded file sent to AI")
                    
                    logger.info("Video session identity reinforcement sent")
                except Exception as prompt_error:
                    logger.warning(f"Could not send video session setup: {prompt_error}")
                
                while not self.quit.is_set():
                    turn = self.session.receive()
                    try:
                        async for response in turn:
                            if data := response.data:
                                audio = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
                                self.audio_queue.put_nowait(audio)
                                
                            # Check for goodbye in text responses
                            if hasattr(response, 'text') and response.text:
                                if detect_goodbye(response.text):
                                    logger.info("Goodbye detected in video session, ending")
                                    await asyncio.sleep(2)  # Let the goodbye message play
                                    self.quit.set()
                                    break
                                    
                    except Exception as e:
                        logger.error(f"Video session error: {e}")
                        break
                        
        except Exception as e:
            logger.error(f"Error in GeminiVideoHandler start_up: {e}")
        finally:
            # Clean up session
            if self.session_id and self.session_id in active_sessions:
                del active_sessions[self.session_id]

    async def video_receive(self, frame: np.ndarray):
        """Receive video frame from client"""
        if self.session:
            # Send image every 2 seconds to avoid overwhelming the API
            if time.time() - self.last_frame_time > 2:
                self.last_frame_time = time.time()
                try:
                    await self.session.send(input=encode_image(frame))
                except Exception as e:
                    logger.warning(f"Error sending video frame: {e}")

    async def video_emit(self):
        """Video emit - AI doesn't send video back, only audio"""
        # Return None since AI doesn't send video back to user
        # FastRTC requires this method for send-receive mode
        return None

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Receive audio from client"""
        try:
            _, array = frame
            array = array.squeeze()
            audio_message = encode_audio_dict(array)
            if self.session:
                await self.session.send(input=audio_message)
        except Exception as e:
            logger.error(f"Error processing audio in video session: {e}")

    async def emit(self):
        """Emit audio response to client"""
        try:
            array = await wait_for_item(self.audio_queue, 0.01)
            if array is not None:
                return (self.output_sample_rate, array)
            return None
        except Exception as e:
            logger.error(f"Error emitting audio in video session: {e}")
            return None

    async def shutdown(self) -> None:
        """Clean shutdown of video handler"""
        logger.info("Shutting down Gemini video handler")
        if self.session:
            self.quit.set()
            try:
                await self.session.close()
            except Exception as e:
                logger.warning(f"Error closing video session: {e}")
            self.quit.clear()
            
        # Clean up session
        if self.session_id and self.session_id in active_sessions:
            del active_sessions[self.session_id]

# Enhanced Stream configuration with better error handling
if FASTRTC_AVAILABLE:
    try:
        # Audio-only stream
        audio_stream = Stream(
            modality="audio",
            mode="send-receive",
            handler=GeminiHandler(),
            rtc_configuration=get_cloudflare_turn_credentials_async,
            concurrency_limit=5,
            time_limit=300,
        )
        
        # Video stream (user sends video, AI responds with audio - we just don't display AI video)
        video_stream = Stream(
            modality="audio-video", 
            mode="send-receive",  # Must use send-receive, we just won't display AI video
            handler=GeminiVideoHandler(),
            rtc_configuration=get_cloudflare_turn_credentials_async,
            concurrency_limit=3,
            time_limit=300,
        )
        
        logger.info("FastRTC Streams initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing FastRTC Streams: {e}")
        # Proper fallback with required arguments
        audio_stream = Stream(
            modality="audio",
            mode="send-receive", 
            handler=GeminiHandler(),
            rtc_configuration=get_cloudflare_turn_credentials_async,
        )
        video_stream = Stream(
            modality="audio-video",
            mode="send-receive",
            handler=GeminiVideoHandler(), 
            rtc_configuration=get_cloudflare_turn_credentials_async,
        )
else:
    # Fallback streams with proper arguments
    audio_stream = Stream(
        modality="audio",
        mode="send-receive",
        handler=GeminiHandler(),
    )
    video_stream = Stream(
        modality="audio-video", 
        mode="send-receive",
        handler=GeminiVideoHandler(),
    )
    logger.warning("Using fallback Stream implementations")

# Request/Response models
class ChatRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    response: str

class InputData(BaseModel):
    webrtc_id: str
    voice_name: str
    mode: str = "audio"  # "audio" or "video"
    uploaded_files: Optional[List[dict]] = None

# Initialize FastAPI app
app = FastAPI(
    title="AI Psychologist with Gemini Voice & Video",
    description="A therapeutic AI assistant with real-time voice and video capabilities powered by Google Gemini",
    version="2.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (only if directory exists)
static_dir = current_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("Static files mounted")
else:
    logger.warning("Static directory not found, skipping static file mounting")

# Mount FastRTC streams
try:
    audio_stream.mount(app, path="/audio")
    video_stream.mount(app, path="/video")
    if FASTRTC_AVAILABLE:
        logger.info("FastRTC streams mounted successfully")
    else:
        logger.info("Fallback streams mounted (WebRTC features limited)")
except Exception as e:
    logger.error(f"Error mounting streams: {e}")

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """Upload file to share with AI psychologist"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Determine mime type
        mime_type = file.content_type or "application/octet-stream"
        
        # For images, convert to JPEG if needed
        if mime_type.startswith("image/"):
            try:
                pil_image = Image.open(BytesIO(file_content))
                with BytesIO() as output_bytes:
                    pil_image.save(output_bytes, "JPEG")
                    file_content = output_bytes.getvalue()
                mime_type = "image/jpeg"
            except Exception as img_error:
                logger.warning(f"Could not process image: {img_error}")
        
        # Encode file for Gemini
        encoded_file = encode_file_content(file_content, mime_type)
        
        logger.info(f"File uploaded: {file.filename}, type: {mime_type}, size: {len(file_content)} bytes")
        
        return {
            "status": "success",
            "filename": file.filename,
            "mime_type": mime_type,
            "size": len(file_content),
            "encoded_data": encoded_file
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post("/input_hook")
async def set_input_hook(body: InputData):
    """Set input parameters for WebRTC stream with validation"""
    try:
        if not FASTRTC_AVAILABLE:
            return {
                "status": "error",
                "message": "WebRTC features are not available in this deployment",
                "webrtc_id": body.webrtc_id,
                "voice": body.voice_name,
                "mode": body.mode
            }
        
        # Validate inputs
        if not body.webrtc_id or not body.voice_name:
            raise HTTPException(status_code=400, detail="Missing webrtc_id or voice_name")
        
        # Validate voice name
        valid_voices = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
        if body.voice_name not in valid_voices:
            logger.warning(f"Invalid voice {body.voice_name}, using Aoede")
            body.voice_name = "Aoede"
        
        # Set input for appropriate stream
        if body.mode == "video":
            video_stream.set_input(body.webrtc_id, body.voice_name, body.uploaded_files)
        else:
            audio_stream.set_input(body.webrtc_id, body.voice_name)
            
        logger.info(f"Input set for WebRTC ID: {body.webrtc_id}, Voice: {body.voice_name}, Mode: {body.mode}")
        return {"status": "ok", "webrtc_id": body.webrtc_id, "voice": body.voice_name, "mode": body.mode}
        
    except Exception as e:
        logger.error(f"Error setting input: {e}")
        raise HTTPException(status_code=500, detail=f"Input hook failed: {str(e)}")

@app.post("/end_session")
async def end_session(session_id: str = None):
    """Manually end a session"""
    try:
        if session_id and session_id in active_sessions:
            session_info = active_sessions[session_id]
            handler = session_info["handler"]
            handler.shutdown()
            logger.info(f"Session {session_id} ended manually")
            return {"status": "success", "message": "Session ended"}
        else:
            # End all active sessions
            for sid, session_info in list(active_sessions.items()):
                handler = session_info["handler"]
                handler.shutdown()
            logger.info("All sessions ended")
            return {"status": "success", "message": "All sessions ended"}
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Text-based chat endpoint using Gemini"""
    try:
        # Check for goodbye in text chat
        if detect_goodbye(request.prompt):
            return ChatResponse(response="Thank you for sharing with me today. Take care of yourself, and remember that you have the strength to handle whatever comes your way. Feel free to come back anytime you need support. Goodbye for now! 💙")
        
        # Initialize Gemini client for text
        if settings.use_vertex_ai and settings.google_cloud_project:
            client = genai.Client(
                vertexai=True,
                project=settings.google_cloud_project,
                location='us-central1'
            )
        else:
            client = genai.Client(
                api_key=settings.gemini_api_key,
            )

        # Generate response
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                {"parts": [{"text": f"System: {SYSTEM_PROMPT}\n\nUser: {request.prompt}"}]}
            ]
        )

        ai_response = response.text if hasattr(response, 'text') else str(response)
        
        logger.info(f"Text chat response generated: {len(ai_response)} characters")
        return ChatResponse(response=ai_response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        
        # Fallback response for therapeutic context
        fallback_response = (
            "I'm experiencing a technical difficulty right now. "
            "Your feelings and thoughts are important - would you like to try again, "
            "or would it help to contact a human therapist?"
        )
        return ChatResponse(response=fallback_response)

@app.websocket("/ws")
async def websocket_text_chat(websocket: WebSocket):
    """WebSocket endpoint for text-based chat with streaming responses"""
    await websocket.accept()
    logger.info("WebSocket connection established for text chat")
    
    try:
        while True:
            # Receive message from client
            user_msg = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {user_msg[:100]}...")

            # Crisis detection keywords
            crisis_indicators = [
                "suicide", "kill myself", "self-harm", "end it all", 
                "hurt myself", "don't want to live", "better off dead"
            ]
            
            if any(indicator in user_msg.lower() for indicator in crisis_indicators):
                crisis_response = (
                    "🚨 I'm very concerned about what you're sharing. "
                    "Please reach out for immediate support:\n\n"
                    "• National Suicide Prevention Lifeline: 988\n"
                    "• Crisis Text Line: Text HOME to 741741\n"
                    "• Emergency Services: 911\n\n"
                    "Your life has value and there are people who want to help."
                )
                await websocket.send_text(crisis_response)
                continue

            # Check for goodbye
            if detect_goodbye(user_msg):
                goodbye_response = (
                    "Thank you for sharing with me today. Take care of yourself, and remember "
                    "that you have the strength to handle whatever comes your way. Feel free to "
                    "come back anytime you need support. Goodbye for now! 💙"
                )
                await websocket.send_text(goodbye_response)
                continue

            try:
                # Initialize Gemini client
                if settings.use_vertex_ai and settings.google_cloud_project:
                    client = genai.Client(
                        vertexai=True,
                        project=settings.google_cloud_project,
                        location='us-central1'
                    )
                else:
                    client = genai.Client(
                        api_key=settings.gemini_api_key,
                    )

                # Generate response
                response = await client.aio.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[
                        {"parts": [{"text": f"System: {SYSTEM_PROMPT}\n\nUser: {user_msg}"}]}
                    ]
                )

                # Send the complete response at once
                if hasattr(response, 'text') and response.text:
                    # Clean up the response text
                    clean_text = response.text
                    
                    # Remove common voice artifacts
                    artifacts_to_remove = [
                        "[pause]", "...", "… [pause] …", "[PAUSE]",
                        "*pause*", "(pause)", "... [pause] ...",
                        "[breath]", "*breath*", "(breath)"
                    ]
                    
                    for artifact in artifacts_to_remove:
                        clean_text = clean_text.replace(artifact, "")
                    
                    # Clean up extra spaces and line breaks
                    clean_text = " ".join(clean_text.split())
                    
                    await websocket.send_text(clean_text)
                else:
                    await websocket.send_text("I apologize, I couldn't generate a proper response. How else can I help you?")

            except Exception as e:
                logger.error(f"Error generating response: {e}")
                await websocket.send_text(
                    "I apologize, but I'm having technical difficulties. "
                    "Your mental health is important - please consider speaking "
                    "with a human therapist if you need immediate support."
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(f"Connection error occurred: {str(e)}")
        except:
            pass

@app.get("/api/voices")
async def get_available_voices():
    """Get list of available Gemini voices"""
    return {
        "voices": [
            {"id": "Puck", "name": "Puck", "description": "Warm, conversational voice"},
            {"id": "Charon", "name": "Charon", "description": "Calm, reassuring voice"},
            {"id": "Kore", "name": "Kore", "description": "Gentle, empathetic voice"},
            {"id": "Fenrir", "name": "Fenrir", "description": "Strong, supportive voice"},
            {"id": "Aoede", "name": "Aoede", "description": "Melodic, soothing voice (recommended)"},
        ],
        "fastrtc_available": FASTRTC_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway deployment"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "ai-psychologist-gemini",
            "version": "2.1.0",
            "fastrtc_available": FASTRTC_AVAILABLE,
            "video_support": FASTRTC_AVAILABLE,
            "active_sessions": len(active_sessions)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check file existence and configuration"""
    html_file = current_dir / "index.html"
    static_dir = current_dir / "static"
    
    return {
        "current_directory": str(current_dir),
        "html_file_exists": html_file.exists(),
        "html_file_path": str(html_file),
        "static_dir_exists": static_dir.exists(),
        "gemini_api_key_set": bool(settings.gemini_api_key),
        "fastrtc_available": FASTRTC_AVAILABLE,
        "fastrtc_error": FASTRTC_ERROR,
        "video_support": FASTRTC_AVAILABLE,
        "active_sessions": list(active_sessions.keys()),
        "files_in_directory": [f.name for f in current_dir.iterdir() if f.is_file()][:10],
        "python_version": sys.version,
        "recommendations": [
            "Text chat is always available",
            "Voice and video chat require FastRTC compatibility",
            f"Current issue: {FASTRTC_ERROR}" if FASTRTC_ERROR else "No issues detected"
        ]
    }

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main HTML interface with video support"""
    try:
        # Check if index.html exists
        html_file = current_dir / "index.html"
        if html_file.exists():
            try:
                html_content = html_file.read_text(encoding='utf-8')
            except Exception as read_error:
                logger.error(f"Error reading index.html: {read_error}")
                return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head><title>AI Psychologist - File Error</title></head>
<body>
    <h1>File Read Error</h1>
    <p>Could not read index.html: {read_error}</p>
    <p><a href="/debug">Debug Info</a></p>
</body>
</html>
                """, status_code=503)
            
            # Get RTC configuration
            try:
                rtc_config = await get_cloudflare_turn_credentials_async()
            except Exception as rtc_error:
                logger.warning(f"Could not get RTC config, using fallback: {rtc_error}")
                rtc_config = {
                    "iceServers": [
                        {"urls": "stun:stun.l.google.com:19302"},
                        {"urls": "stun:stun1.l.google.com:19302"}
                    ]
                }
            
            # Replace configuration placeholder
            html_content = html_content.replace(
                "__RTC_CONFIGURATION__", 
                json.dumps(rtc_config)
            )
            
            return HTMLResponse(content=html_content)
        else:
            return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Psychologist - Loading</title>
</head>
<body>
    <h1>🧠 AI Psychologist</h1>
    <p>Loading application...</p>
    <p>Text chat is always available, video features depend on FastRTC support.</p>
</body>
</html>
            """)
        
    except Exception as e:
        logger.error(f"Error serving main page: {e}")
        return HTMLResponse(
            content=f"""
<!DOCTYPE html>
<html>
<head><title>AI Psychologist - Error</title></head>
<body>
    <h1>Service Error</h1>
    <p>Error: {str(e)}</p>
    <p><a href="/health">Check Health</a> | <a href="/debug">Debug Info</a></p>
</body>
</html>
            """, 
            status_code=503
        )

if __name__ == "__main__":
    import uvicorn
    
    # Development vs Production configuration
    if os.getenv("ENVIRONMENT") == "development":
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=settings.port,
            reload=True,
            log_level="info"
        )
    else:
        # Production configuration
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=settings.port,
            workers=1,  # Single worker for WebRTC state management
            log_level="info"
        )