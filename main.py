import asyncio
import base64
import json
import os
import pathlib
from collections.abc import AsyncGenerator
from typing import Dict, Any, Optional, List, Literal
import logging
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
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
SYSTEM_PROMPT = """You are a compassionate AI psychologist having a natural conversation. Use:

**Core Principles:**
- **Active listening**: "I hear you're feeling..." and acknowledge emotions explicitly
- **CBT techniques**: "What evidence supports this thought?" and challenge negative patterns
- **Validation**: Acknowledge feelings before offering perspective
- **Conversational responses**: Keep responses natural and supportive

**Guidelines:**
- Use inclusive language: "many people experience this" 
- Ask one follow-up question per response to maintain conversation flow
- Express empathy naturally without voice artifacts
- Do not use [pause], ..., or similar voice markers in text responses

**Boundaries:**
- NEVER diagnose. Say: "A licensed therapist could provide a proper assessment"
- For crisis situations, respond: "This sounds like you need immediate support. Please contact a mental health helpline or emergency services right away"

**Therapeutic Techniques:**
- Cognitive restructuring: "What would you tell a friend in this situation?"
- Mindfulness: "Let's pause and notice what you're feeling right now"
- Behavioral insights: "What patterns do you notice in your mood?"

Keep your responses natural, supportive, and focused on one main therapeutic intervention per exchange. Write in clear, flowing text without voice-specific formatting."""

def encode_audio(data: np.ndarray) -> str:
    """Encode Audio data to send to Gemini"""
    return base64.b64encode(data.tobytes()).decode("UTF-8")

# Import FastRTC with comprehensive error handling and logging
try:
    from fastrtc import (
        AsyncStreamHandler,
        Stream,
        get_cloudflare_turn_credentials_async,
        wait_for_item,
    )
    FASTRTC_AVAILABLE = True
    logger.info("✅ FastRTC imported successfully")
    
    # Test critical components
    try:
        from fastrtc.tracks import EmitType, StreamHandler
        logger.info("✅ FastRTC tracks module available")
    except ImportError as tracks_error:
        logger.warning(f"⚠️  FastRTC tracks module issue: {tracks_error}")
        
except ImportError as e:
    logger.error(f"❌ FastRTC import error: {e}")
    logger.error("This usually means:")
    logger.error("1. aiortc version incompatibility")
    logger.error("2. Missing system dependencies")
    logger.error("3. Build compilation failed")
    FASTRTC_AVAILABLE = False
    
    # Create fallback classes
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
            pass
        
        async def receive(self, frame):
            pass
        
        async def emit(self):
            return None
        
        def shutdown(self):
            pass
        
        def copy(self):
            return AsyncStreamHandler(self.expected_layout, self.output_sample_rate)
    
    class Stream:
        def __init__(self, *args, **kwargs):
            logger.warning("Using fallback Stream implementation")
        
        def mount(self, app):
            # Add fallback WebRTC endpoints
            @app.post("/webrtc/offer")
            async def webrtc_offer_fallback(request: Request):
                body = await request.json()
                return {
                    "status": "failed",
                    "meta": {
                        "error": "webrtc_not_available",
                        "message": "WebRTC features are not available in this deployment"
                    }
                }
        
        def set_input(self, webrtc_id, voice_name):
            logger.warning(f"Stream.set_input() called but WebRTC not available")
    
    async def get_cloudflare_turn_credentials_async():
        return {
            "iceServers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": "stun:stun1.l.google.com:19302"}
            ]
        }

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

            # Simplified LiveConnectConfig
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
                logger.info(f"LiveConnectConfig created with voice: {voice_name}")
            except Exception as config_error:
                logger.error(f"Failed to create LiveConnectConfig: {config_error}")
                return
            
            logger.info(f"Starting Gemini Live session with voice: {voice_name}")
            
            try:
                async with self.client.aio.live.connect(
                    model="gemini-2.0-flash-exp", 
                    config=config
                ) as session:
                    logger.info("Gemini Live session established successfully")
                    self.session = session
                    
                    # Send initial system message through the session
                    try:
                        await session.send({"text": f"System: {SYSTEM_PROMPT}"})
                        logger.info("System prompt sent successfully")
                    except Exception as prompt_error:
                        logger.warning(f"Could not send system prompt: {prompt_error}")
                    
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

            except Exception as session_error:
                logger.error(f"Gemini Live session error: {session_error}")
                logger.info("Live session failed, handler will operate in degraded mode")

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

# Enhanced Stream configuration with better error handling
if FASTRTC_AVAILABLE:
    try:
        stream = Stream(
            modality="audio",
            mode="send-receive",
            handler=GeminiHandler(),
            rtc_configuration=get_cloudflare_turn_credentials_async,
            concurrency_limit=5,
            time_limit=300,
        )
        logger.info("FastRTC Stream initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing FastRTC Stream: {e}")
        stream = Stream()  # Fallback stream
else:
    stream = Stream()  # Fallback stream
    logger.warning("Using fallback Stream implementation")

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

# Initialize FastAPI app
app = FastAPI(
    title="AI Psychologist with Gemini Voice",
    description="A therapeutic AI assistant with real-time voice capabilities powered by Google Gemini",
    version="2.0.0",
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

# Mount FastRTC stream
try:
    stream.mount(app)
    if FASTRTC_AVAILABLE:
        logger.info("FastRTC stream mounted successfully")
    else:
        logger.info("Fallback stream mounted (WebRTC features limited)")
except Exception as e:
    logger.error(f"Error mounting stream: {e}")

@app.post("/input_hook")
async def set_input_hook(body: InputData):
    """Set input parameters for WebRTC stream with validation"""
    try:
        if not FASTRTC_AVAILABLE:
            return {
                "status": "error",
                "message": "WebRTC features are not available in this deployment",
                "webrtc_id": body.webrtc_id,
                "voice": body.voice_name
            }
        
        # Validate inputs
        if not body.webrtc_id or not body.voice_name:
            raise HTTPException(status_code=400, detail="Missing webrtc_id or voice_name")
        
        # Validate voice name
        valid_voices = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
        if body.voice_name not in valid_voices:
            logger.warning(f"Invalid voice {body.voice_name}, using Aoede")
            body.voice_name = "Aoede"
        
        stream.set_input(body.webrtc_id, body.voice_name)
        logger.info(f"Input set for WebRTC ID: {body.webrtc_id}, Voice: {body.voice_name}")
        return {"status": "ok", "webrtc_id": body.webrtc_id, "voice": body.voice_name}
        
    except Exception as e:
        logger.error(f"Error setting input: {e}")
        raise HTTPException(status_code=500, detail=f"Input hook failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Text-based chat endpoint using Gemini"""
    try:
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
        "files_in_directory": [f.name for f in current_dir.iterdir() if f.is_file()][:10]
    }

@app.get("/deployment-status")
async def deployment_status():
    """Comprehensive deployment status check"""
    try:
        import aiortc
        aiortc_version = getattr(aiortc, '__version__', 'unknown')
        aiortc_available = True
    except ImportError as e:
        aiortc_version = f"Import failed: {e}"
        aiortc_available = False
    
    try:
        import fastrtc
        fastrtc_version = getattr(fastrtc, '__version__', 'unknown')
        fastrtc_import_ok = True
        
        # Test components
        try:
            from fastrtc import AsyncStreamHandler, Stream
            fastrtc_components_ok = True
        except ImportError:
            fastrtc_components_ok = False
            
    except ImportError as e:
        fastrtc_version = f"Import failed: {e}"
        fastrtc_import_ok = False
        fastrtc_components_ok = False
    
    return {
        "deployment_status": "deployed",
        "fastrtc": {
            "available": FASTRTC_AVAILABLE,
            "version": fastrtc_version,
            "import_ok": fastrtc_import_ok,
            "components_ok": fastrtc_components_ok
        },
        "aiortc": {
            "available": aiortc_available,
            "version": aiortc_version
        },
        "voice_features": FASTRTC_AVAILABLE,
        "text_features": True,
        "recommendations": [
            "Use text chat - always available",
            "Voice chat - depends on FastRTC availability",
            "Check /debug for detailed diagnostics"
        ]
    }

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main HTML interface with better error handling"""
    try:
        # Check if index.html exists
        html_file = current_dir / "index.html"
        if not html_file.exists():
            logger.error(f"index.html not found at {html_file}")
            
            # Return the embedded interface
            return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Psychologist - Voice & Text Chat</title>
    <style>
        :root {
            --color-accent: #6366f1;
            --color-background: #0f172a;
            --color-surface: #1e293b;
            --color-text: #e2e8f0;
            --color-success: #10b981;
            --color-warning: #f59e0b;
            --color-error: #ef4444;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, var(--color-background) 0%, #1e1b4b 100%);
            color: var(--color-text);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
        }

        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 0.5rem;
        }

        .header .subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }

        .disclaimer {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .container {
            width: 90%;
            max-width: 900px;
            background: rgba(30, 41, 59, 0.95);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .mode-selector {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 0.75rem;
            padding: 0.5rem;
        }

        .mode-btn {
            flex: 1;
            padding: 0.75rem 1.5rem;
            border: none;
            background: transparent;
            color: var(--color-text);
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .mode-btn.active {
            background: var(--color-accent);
            color: white;
        }

        .mode-btn:hover:not(.active) {
            background: rgba(99, 102, 241, 0.2);
        }

        .mode-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .chat-container {
            height: 60vh;
            border-radius: 0.75rem;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(15, 23, 42, 0.5);
        }

        .messages {
            height: calc(100% - 80px);
            overflow-y: auto;
            padding: 1rem;
        }

        .message {
            display: flex;
            margin-bottom: 1rem;
            animation: fadeIn 0.3s ease-in-out;
        }

        .user-message {
            flex-direction: row-reverse;
        }

        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 10px;
            background: linear-gradient(135deg, var(--color-accent), #8b5cf6);
            color: white;
            font-size: 1.2rem;
        }

        .user-message .avatar {
            background: linear-gradient(135deg, #06b6d4, #0891b2);
        }

        .bubble {
            max-width: 70%;
            padding: 12px 15px;
            border-radius: 18px;
            position: relative;
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .ai-message .bubble {
            border-top-left-radius: 5px;
        }

        .user-message .bubble {
            border-top-right-radius: 5px;
            background: rgba(6, 182, 212, 0.2);
            border-color: rgba(6, 182, 212, 0.3);
        }

        .bubble p {
            margin: 0;
            line-height: 1.5;
        }

        .input-area {
            display: flex;
            padding: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(15, 23, 42, 0.8);
        }

        .text-input {
            flex: 1;
            border: none;
            padding: 10px 15px;
            border-radius: 20px;
            resize: none;
            background: rgba(30, 41, 59, 0.8);
            color: var(--color-text);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            font-family: inherit;
            font-size: 1rem;
        }

        .text-input:focus {
            outline: none;
            border-color: var(--color-accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }

        .send-btn {
            background: var(--color-accent);
            border: none;
            cursor: pointer;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            color: white;
            margin-left: 0.75rem;
        }

        .send-btn:hover {
            background-color: #5b21b6;
            transform: scale(1.05);
        }

        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .voice-notice {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            text-align: center;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 768px) {
            .container {
                width: 95%;
                padding: 1.5rem;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .bubble {
                max-width: 85%;
            }

            .chat-container {
                height: 70vh;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 AI Psychologist</h1>
        <p class="subtitle">Real-Time Voice & Text Therapeutic Support</p>
        <div class="disclaimer">
            <strong>⚠️ Important:</strong> This AI is not a substitute for professional mental health services. 
            If you're experiencing a crisis, please contact emergency services or a mental health helpline immediately.
        </div>
    </div>

    <div class="container">
        <div class="mode-selector">
            <button class="mode-btn" data-mode="voice" id="voice-btn">🎤 Voice Chat</button>
            <button class="mode-btn active" data-mode="text">💬 Text Chat</button>
        </div>

        <div class="voice-notice" id="voice-notice" style="display: none;">
            Voice features are being initialized. Please use text chat for now.
        </div>

        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message ai-message">
                    <div class="avatar">🤖</div>
                    <div class="bubble">
                        <p>Hello! I'm your AI psychologist. I'm here to listen and support you. How are you feeling today?</p>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <textarea class="text-input" id="user-input" placeholder="Share your thoughts and feelings..." rows="1"></textarea>
                <button class="send-btn" id="send-btn" title="Send">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M2 21l21-9L2 3v7l15 2-15 2v7z"></path>
                    </svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        const messagesContainer = document.getElementById('messages');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-btn');
        const voiceBtn = document.getElementById('voice-btn');
        const voiceNotice = document.getElementById('voice-notice');
        let isProcessing = false;

        // Check if voice features are available
        fetch('/debug')
            .then(response => response.json())
            .then(data => {
                if (data.fastrtc_available) {
                    voiceBtn.disabled = false;
                    voiceNotice.style.display = 'none';
                } else {
                    voiceBtn.disabled = true;
                    voiceNotice.style.display = 'block';
                    voiceNotice.textContent = 'Voice features are not available in this deployment. Text chat is fully functional.';
                }
            })
            .catch(() => {
                voiceBtn.disabled = true;
                voiceNotice.style.display = 'block';
            });

        // Auto-resize textarea
        userInput.addEventListener('input', () => {
            userInput.style.height = 'auto';
            userInput.style.height = (userInput.scrollHeight) + 'px';
        });

        // Send message on Enter (without Shift)
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        sendButton.addEventListener('click', sendMessage);

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message || isProcessing) return;

            isProcessing = true;
            sendButton.disabled = true;

            addMessageToChat(message, 'user');
            userInput.value = '';
            userInput.style.height = 'auto';
            
            // Add typing indicator
            const typingMessage = addMessageToChat('Thinking...', 'ai');

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: message,
                        temperature: 0.7,
                        max_tokens: 512
                    })
                });

                const data = await response.json();
                
                // Remove typing indicator and show response
                typingMessage.remove();
                addMessageToChat(data.response, 'ai');

            } catch (error) {
                console.error('Error:', error);
                typingMessage.remove();
                addMessageToChat('I apologize, but I encountered an error. Please try again.', 'ai');
            } finally {
                isProcessing = false;
                sendButton.disabled = false;
            }
        }

        function addMessageToChat(text, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message}`;
            
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'avatar';
            avatarDiv.textContent = sender === 'user' ? '👤' : '🤖';
            
            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'bubble';
            
            const paragraph = document.createElement('p');
            paragraph.textContent = text;
            bubbleDiv.appendChild(paragraph);
            
            messageDiv.appendChild(avatarDiv);
            messageDiv.appendChild(bubbleDiv);
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return messageDiv;
        }
    </script>
</body>
</html>
            """)
        
        # If index.html exists, try to read it
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
        
        # Get RTC configuration (with better error handling)
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
        
        logger.info("Successfully served main page")
        return HTMLResponse(content=html_content)
        
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

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway deployment"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "ai-psychologist-gemini",
            "version": "2.0.0",
            "fastrtc_available": FASTRTC_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/test")
async def test_gemini():
    """Test Gemini API connection"""
    try:
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

        # Simple test
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[{"parts": [{"text": "Say hello"}]}],
        )

        return {
            "status": "success",
            "response": response.text if hasattr(response, 'text') else str(response),
            "api_key_provided": bool(settings.gemini_api_key),
            "fastrtc_available": FASTRTC_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Gemini test failed: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "api_key_provided": bool(settings.gemini_api_key),
            "fastrtc_available": FASTRTC_AVAILABLE
        }

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

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return HTMLResponse(
        content="<h1>Page Not Found</h1><p>The requested resource was not found.</p>",
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return HTMLResponse(
        content="<h1>Internal Server Error</h1><p>Please try again later.</p>",
        status_code=500
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
