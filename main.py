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
from fastrtc import (
    AsyncStreamHandler,
    Stream,
    get_cloudflare_turn_credentials_async,
    wait_for_item,
)
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
                        # Remove problematic http_options for Live API
                    )
                logger.info("Gemini client initialized successfully")
            except Exception as client_error:
                logger.error(f"Failed to initialize Gemini client: {client_error}")
                return

            # Simplified LiveConnectConfig - remove problematic fields
            try:
                config = LiveConnectConfig(
                    response_modalities=["AUDIO"],  # Start with audio only
                    speech_config=SpeechConfig(
                        voice_config=VoiceConfig(
                            prebuilt_voice_config=PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                    # Try without system_instruction first to see if that's the issue
                    generation_config={
                        "temperature": 0.8,
                        "max_output_tokens": 256,  # Reduced for voice
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
                # If live session fails, fall back to text-only mode
                logger.info("Live session failed, handler will operate in degraded mode")

        except Exception as e:
            logger.error(f"Error in GeminiHandler start_up: {e}")
            logger.info("Handler startup failed, will operate in degraded mode")
            # Graceful degradation - continue without crashing
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
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=GeminiHandler(),
    rtc_configuration=get_cloudflare_turn_credentials_async,
    concurrency_limit=5,  # Reduced from 10
    time_limit=300,  # 5 minutes max per session
)

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
stream.mount(app)

@app.post("/input_hook")
async def set_input_hook(body: InputData):
    """Set input parameters for WebRTC stream with validation"""
    try:
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

        # FIX: Use correct method and parameters
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
        "files_in_directory": [f.name for f in current_dir.iterdir() if f.is_file()][:10]  # First 10 files
    }

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main HTML interface with better error handling"""
    try:
        # Check if index.html exists
        html_file = current_dir / "index.html"
        if not html_file.exists():
            logger.error(f"index.html not found at {html_file}")
            
            # Return a simple working interface
            return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Psychologist - Text Chat</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background: #1a1a2e;
            color: #eee;
        }
        #chat { 
            border: 1px solid #444; 
            height: 400px; 
            overflow-y: auto; 
            padding: 10px; 
            margin: 10px 0; 
            background: #16213e;
            border-radius: 8px;
        }
        #input { 
            width: 70%; 
            padding: 10px; 
            background: #0f1419;
            border: 1px solid #444;
            color: #eee;
            border-radius: 4px;
        }
        #send { 
            padding: 10px 20px; 
            background: #7c3aed;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #send:hover { background: #6d28d9; }
        .message { 
            margin: 10px 0; 
            padding: 10px; 
            border-radius: 8px; 
        }
        .user { 
            background: #1e40af; 
            text-align: right; 
            margin-left: 20%;
        }
        .ai { 
            background: #374151; 
            margin-right: 20%;
        }
        .typing { 
            background: #374151; 
            margin-right: 20%;
            font-style: italic;
            opacity: 0.7;
        }
        .typing::after {
            content: "...";
            animation: dots 1.5s steps(5, end) infinite;
        }
        .ai::after {
            content: none; /* Remove the dots when it becomes an AI message */
        }
        @keyframes dots {
            0%, 20% { color: rgba(0,0,0,0); text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
            40% { color: white; text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
            60% { text-shadow: .25em 0 0 white, .5em 0 0 rgba(0,0,0,0); }
            80%, 100% { text-shadow: .25em 0 0 white, .5em 0 0 white; }
        }
        h1 { color: #7c3aed; text-align: center; }
    </style>
</head>
<body>
    <h1>🧠 AI Psychologist - Text Chat</h1>
    <div id="chat"></div>
    <div>
        <input type="text" id="input" placeholder="Type your message here..." />
        <button id="send">Send</button>
    </div>
    
    <script>
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const send = document.getElementById('send');
        let typingIndicator = null;
        
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            // Check if it's a typing indicator
            if (event.data === 'typing') {
                // Create typing indicator only if it doesn't exist
                if (!typingIndicator) {
                    typingIndicator = document.createElement('div');
                    typingIndicator.className = 'message typing';
                    typingIndicator.textContent = 'AI is thinking';
                    chat.appendChild(typingIndicator);
                    chat.scrollTop = chat.scrollHeight;
                }
                return;
            }
            
            // Regular message - transform the existing typing indicator
            if (typingIndicator) {
                // Transform the existing typing bubble into AI response
                typingIndicator.className = 'message ai';
                typingIndicator.textContent = event.data;
                
                // Remove the CSS animation by clearing the ::after content
                typingIndicator.style.fontStyle = 'normal';
                typingIndicator.style.opacity = '1';
                
                typingIndicator = null; // Clear reference
            } else {
                // Fallback: create new message (shouldn't happen in normal flow)
                const message = document.createElement('div');
                message.className = 'message ai';
                message.textContent = event.data;
                chat.appendChild(message);
            }
            
            chat.scrollTop = chat.scrollHeight;
        };
        
        function sendMessage() {
            const text = input.value.trim();
            if (text) {
                const message = document.createElement('div');
                message.className = 'message user';
                message.textContent = text;
                chat.appendChild(message);
                
                ws.send(text);
                input.value = '';
                chat.scrollTop = chat.scrollHeight;
            }
        }
        
        send.addEventListener('click', sendMessage);
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
        
        // Connection status
        ws.onopen = function() {
            const message = document.createElement('div');
            message.className = 'message ai';
            message.textContent = 'Hello! I\\'m here to listen and support you. How are you feeling today?';
            chat.appendChild(message);
        };
        
        ws.onerror = function() {
            const message = document.createElement('div');
            message.className = 'message ai';
            message.textContent = 'Connection error. Please refresh the page.';
            chat.appendChild(message);
        };
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

                # Don't send typing indicator - just generate response directly
                
                # Generate response (non-streaming since streaming isn't supported)
                response = await client.aio.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[
                        {"parts": [{"text": f"System: {SYSTEM_PROMPT}\n\nUser: {user_msg}"}]}
                    ]
                )

                # Send the complete response at once
                if hasattr(response, 'text') and response.text:
                    # Clean up the response text - remove voice-specific artifacts
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
            pass  # Connection likely already closed

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway deployment"""
    try:
        # Basic health check - could be expanded to check Gemini API connectivity
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "ai-psychologist-gemini",
            "version": "2.0.0"
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
            "api_key_length": len(settings.gemini_api_key) if settings.gemini_api_key else 0
        }
    except Exception as e:
        logger.error(f"Gemini test failed: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "api_key_provided": bool(settings.gemini_api_key),
            "api_key_length": len(settings.gemini_api_key) if settings.gemini_api_key else 0,
            "troubleshooting": {
                "check_api_key": "Make sure GEMINI_API_KEY is set in your .env file",
                "verify_key": "Visit https://aistudio.google.com/apikey to get/verify your API key",
                "model_access": "Ensure you have access to gemini-2.0-flash-exp model"
            }
        }

@app.get("/test-streaming")
async def test_streaming():
    """Test streaming response from Gemini"""
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

        # Test the correct way to do streaming with the new SDK
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[{"parts": [{"text": "Say hello and introduce yourself as an AI psychologist"}]}],
                stream=True  # This is the correct way for the new SDK
            )
            
            result = ""
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    result += chunk.text
            
            return {
                "status": "streaming_success",
                "response": result,
                "method": "streaming"
            }
        except Exception as stream_error:
            # Fallback to non-streaming
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[{"parts": [{"text": "Say hello"}]}],
            )
            
            return {
                "status": "non_streaming_success", 
                "response": response.text if hasattr(response, 'text') else str(response),
                "method": "non-streaming",
                "stream_error": str(stream_error)
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}
async def test_gemini_live():
    """Test Gemini Live API specifically"""
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

        # Test Live API configuration
        config = LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(
                        voice_name="Aoede",
                    )
                )
            ),
            generation_config={
                "temperature": 0.8,
                "max_output_tokens": 128,
            }
        )

        # Try to establish a Live connection
        try:
            async with client.aio.live.connect(
                model="gemini-2.0-flash-exp", 
                config=config
            ) as session:
                return {
                    "status": "success",
                    "message": "Gemini Live API connection successful",
                    "voice": "Aoede"
                }
        except Exception as live_error:
            return {
                "status": "live_api_error",
                "error": str(live_error),
                "suggestion": "Live API might not be available for your API key. Try the regular /test endpoint first."
            }

    except Exception as e:
        logger.error(f"Gemini Live test failed: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "troubleshooting": {
                "check_access": "Gemini Live API might require special access",
                "try_regular": "Try /test endpoint to verify basic API access first",
                "contact_google": "Contact Google for Live API access if needed"
            }
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
        ]
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