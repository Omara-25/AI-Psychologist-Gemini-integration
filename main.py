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
    raise ValueError("GEMINI_API_KEY environment variable is required")

# System prompt for AI Psychologist with voice awareness
SYSTEM_PROMPT = """You are a compassionate AI psychologist having a natural voice conversation. Use:

**Core Principles:**
- **Active listening**: "I hear you're feeling..." and acknowledge emotions explicitly
- **CBT techniques**: "What evidence supports this thought?" and challenge negative patterns
- **Validation**: Acknowledge feelings before offering perspective
- **Brevity for voice**: Keep responses conversational and under 2-3 sentences for natural flow

**Voice-Specific Guidelines:**
- Speak naturally with pauses, like: "That sounds really difficult... [pause] ...how long have you been feeling this way?"
- Use inclusive language: "many people experience this" 
- Ask one follow-up question per response to maintain conversation flow
- Express empathy through tone: "I can hear the pain in your voice"

**Boundaries:**
- NEVER diagnose. Say: "A licensed therapist could provide a proper assessment"
- For crisis situations, respond: "This sounds like you need immediate support. Please contact a mental health helpline or emergency services right away"

**Therapeutic Techniques:**
- Cognitive restructuring: "What would you tell a friend in this situation?"
- Mindfulness: "Let's pause and notice what you're feeling right now"
- Behavioral insights: "What patterns do you notice in your mood?"

Keep your voice responses natural, supportive, and focused on one main therapeutic intervention per exchange."""

def encode_audio(data: np.ndarray) -> str:
    """Encode Audio data to send to Gemini"""
    return base64.b64encode(data.tobytes()).decode("UTF-8")

class GeminiHandler(AsyncStreamHandler):
    """Enhanced handler for Gemini API with therapeutic context"""

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
        
    def copy(self) -> "GeminiHandler":
        return GeminiHandler(
            expected_layout="mono",
            output_sample_rate=self.output_sample_rate,
        )

    async def start_up(self):
        try:
            if not self.phone_mode:
                await self.wait_for_args()
                voice_name = self.latest_args[1]  # Only get voice name, not API key
            else:
                voice_name = "Aoede"

            # Always use server-side API key
            if settings.use_vertex_ai and settings.google_cloud_project:
                self.client = genai.Client(
                    vertexai=True,
                    project=settings.google_cloud_project,
                    location='us-central1'
                )
            else:
                self.client = genai.Client(
                    api_key=settings.gemini_api_key,
                    http_options={"api_version": "v1alpha"},
                )

            # Configure for therapeutic conversation
            config = LiveConnectConfig(
                response_modalities=["AUDIO", "TEXT"],
                speech_config=SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
                system_instruction=SYSTEM_PROMPT,
            )
            
            logger.info(f"Starting Gemini Live session with voice: {voice_name}")
            
            async with self.client.aio.live.connect(
                model="gemini-2.0-flash-exp", 
                config=config
            ) as session:
                logger.info("Gemini Live session established")
                
                async for chunk in session.start_stream(
                    stream=self.stream(), 
                    mime_type="audio/pcm"
                ):
                    if chunk.data:
                        # Convert audio data to numpy array
                        array = np.frombuffer(chunk.data, dtype=np.int16)
                        self.output_queue.put_nowait((self.output_sample_rate, array))
                    
                    if chunk.text:
                        # Log therapeutic insights for monitoring
                        logger.info(f"Gemini text response: {chunk.text[:100]}...")

        except Exception as e:
            logger.error(f"Error in GeminiHandler start_up: {e}")
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
        """Receive audio from client and queue for Gemini"""
        try:
            _, array = frame
            array = array.squeeze()
            
            # Encode audio for Gemini
            audio_bytes = array.astype(np.int16).tobytes()
            self.input_queue.put_nowait(audio_bytes)
            
        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")

    async def emit(self) -> tuple[int, np.ndarray] | None:
        """Emit audio response from Gemini to client"""
        try:
            return await wait_for_item(self.output_queue)
        except Exception as e:
            logger.error(f"Error emitting audio: {e}")
            return None

    def shutdown(self) -> None:
        """Clean shutdown of the handler"""
        logger.info("Shutting down Gemini handler")
        self.quit.set()

# Configure FastRTC stream
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=GeminiHandler(),
    rtc_configuration=get_cloudflare_turn_credentials_async,
    concurrency_limit=10,
    time_limit=300,  # 5 minutes max per session
    additional_inputs=[
        {
            "type": "dropdown", 
            "label": "Voice",
            "choices": [
                "Puck",
                "Charon", 
                "Kore",
                "Fenrir",
                "Aoede",
            ],
            "value": "Aoede",
        },
    ],
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount FastRTC stream
stream.mount(app)

@app.post("/input_hook")
async def set_input_hook(body: InputData):
    """Set input parameters for WebRTC stream"""
    try:
        stream.set_input(body.webrtc_id, body.voice_name)
        logger.info(f"Input set for WebRTC ID: {body.webrtc_id}, Voice: {body.voice_name}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error setting input: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
                http_options={"api_version": "v1alpha"},
            )

        # Create chat session with therapeutic context
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
                {"role": "user", "parts": [{"text": request.prompt}]}
            ],
            generation_config={
                "temperature": request.temperature,
                "max_output_tokens": request.max_tokens,
            }
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

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main HTML interface"""
    try:
        # Get RTC configuration for WebRTC
        rtc_config = await get_cloudflare_turn_credentials_async()
        
        # Read HTML template
        html_content = (current_dir / "index.html").read_text()
        
        # Replace configuration placeholder
        html_content = html_content.replace(
            "__RTC_CONFIGURATION__", 
            json.dumps(rtc_config) if rtc_config else "null"
        )
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error serving main page: {e}")
        return HTMLResponse(
            content="<h1>Service Temporarily Unavailable</h1><p>Please try again later.</p>", 
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
                        http_options={"api_version": "v1alpha"},
                    )

                # Generate streaming response
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[
                        {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
                        {"role": "user", "parts": [{"text": user_msg}]}
                    ],
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 512,
                    },
                    stream=True
                )

                # Stream response token by token
                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        await websocket.send_text(chunk.text)
                        await asyncio.sleep(0.03)  # Simulate natural typing

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
