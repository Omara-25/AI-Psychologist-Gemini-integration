# AI Psychologist with Gemini Voice Integration

A revolutionary therapeutic AI assistant that combines Google Gemini's advanced real-time voice capabilities with evidence-based psychological frameworks. This application provides both text and voice-based therapy sessions with sub-250ms latency and natural conversation flow.

## 🌟 Key Features

### 🎤 **Real-Time Voice Therapy**
- **Gemini 2.0 Flash Multimodal Live API** integration
- **Sub-250ms latency** for natural conversation flow
- **5 therapeutic voices** optimized for mental health support
- **Real-time audio visualization** and voice activity detection
- **WebRTC streaming** with automatic quality adaptation

### 💬 **Intelligent Text Chat**
- **Streaming responses** with natural typing simulation
- **Speech-to-text input** for accessibility
- **Crisis detection** with immediate intervention protocols
- **Conversation memory** across text and voice modalities

### 🧠 **Therapeutic Framework**
- **Evidence-based CBT techniques** and active listening
- **Crisis intervention protocols** with safety measures
- **Empathetic response generation** tailored for mental health
- **Professional boundaries** with appropriate referral guidance

### 🔒 **Enterprise Security**
- **End-to-end encryption** for voice and text data
- **HIPAA-compliant** data handling practices
- **Secure API key management** with rotation support
- **Privacy-first design** with minimal data retention

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │    │   FastAPI Server │    │  Gemini 2.0     │
│                 │◄──►│                  │◄──►│  Live API       │
│ - Voice UI      │    │ - WebRTC Handler │    │                 │
│ - Text Chat     │    │ - Audio Pipeline │    │ - Voice AI      │
│ - Visualization │    │ - Crisis Detection│   │ - Text AI       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┼──────────────────────────────────┐
                                 │                                  │
                    ┌─────────────▼──────────┐    ┌─────────────────▼─┐
                    │    Railway Platform    │    │   WebRTC/TURN     │
                    │                        │    │   Infrastructure   │
                    │ - Auto Scaling         │    │                   │
                    │ - Load Balancing       │    │ - NAT Traversal   │
                    │ - Health Monitoring    │    │ - Media Relay     │
                    └────────────────────────┘    └───────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Gemini API Key** (for server deployment) - [Get one here](https://ai.google.dev/gemini-api/docs/api-key)
- **Modern web browser** with WebRTC support
- **Microphone access** for voice features

> **Note:** Users don't need to provide API keys - they're configured server-side for security.

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-psychologist-gemini.git
cd ai-psychologist-gemini

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
nano .env
```

**Required environment variables:**
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8000
ENVIRONMENT=development
```

### 3. Run Locally

```bash
# Development mode
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000` to access the application.

## 🔧 Configuration Guide

### Gemini API Setup

1. **Get API Key:**
   - Visit [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
   - Create a new API key
   - Copy the key to your `.env` file

2. **Optional - Vertex AI Setup:**
   ```env
   USE_VERTEX_AI=true
   GOOGLE_CLOUD_PROJECT=your-project-id
   ```

### Voice Configuration

Choose from 5 therapeutic voices:
- **Aoede** (Recommended): Melodic and soothing
- **Kore**: Gentle and empathetic  
- **Charon**: Calm and reassuring
- **Puck**: Warm and conversational
- **Fenrir**: Strong and supportive

### Crisis Detection

The system includes built-in crisis detection for keywords like:
- "suicide", "kill myself", "self-harm"
- "end it all", "hurt myself"
- "don't want to live", "better off dead"

Configure crisis response:
```env
ENABLE_CRISIS_DETECTION=true
CRISIS_WEBHOOK_URL=https://your-crisis-system.com/webhook
```

## 🚀 Railway Deployment

### Method 1: One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

### Method 2: Manual Deployment

1. **Create Railway Project:**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli

   # Login and create project
   railway login
   railway init
   ```

2. **Set Environment Variables:**
   ```bash
   railway variables set GEMINI_API_KEY=your_api_key_here
   railway variables set ENVIRONMENT=production
   railway variables set PORT=8000
   ```

3. **Deploy:**
   ```bash
   railway up
   ```

### Production Environment Variables

Set these in your Railway dashboard:

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Your Gemini API key | ✅ |
| `SECRET_KEY` | Application secret key | ✅ |
| `ENVIRONMENT` | Set to "production" | ✅ |
| `PORT` | Server port (default: 8000) | ✅ |
| `LOG_LEVEL` | Logging level (INFO/DEBUG) | ❌ |
| `MAX_CONNECTIONS_PER_IP` | Rate limiting | ❌ |
| `CORS_ORIGINS` | Allowed origins | ❌ |

## 📖 Usage Guide

### Voice Therapy Session

1. **Select Voice:** Choose a therapeutic voice (Aoede recommended)
2. **Start Session:** Click "Start Voice Session"
3. **Grant Permissions:** Allow microphone access
4. **Begin Conversation:** Speak naturally with the AI

**Voice Features:**
- Real-time conversation with natural interruptions
- Audio visualization showing voice activity
- Mute/unmute functionality
- Automatic session management

### Text Chat Mode

1. **Switch to Text:** Click the "Text Chat" tab
2. **Type or Speak:** Use keyboard or microphone button
3. **Send Message:** Press Enter or click send button
4. **View Response:** AI streams response in real-time

**Text Features:**
- Streaming responses for natural conversation flow
- Speech-to-text input support
- Auto-expanding text area
- Message history with avatars

## 🛡️ Security & Privacy

### Data Protection

- **End-to-End Encryption:** All voice data encrypted in transit
- **No Data Storage:** Voice conversations not stored permanently
- **API Key Security:** Keys encrypted and stored securely
- **Session Isolation:** Each conversation is independent

### HIPAA Compliance Features

- **Access Controls:** Configurable authentication
- **Audit Logging:** All interactions logged securely
- **Data Minimization:** Only necessary data processed
- **User Consent:** Clear privacy notifications

### Rate Limiting

```env
# Configure rate limits
MAX_CONNECTIONS_PER_IP=10
MAX_REQUESTS_PER_MINUTE=60
```

## 🔍 Monitoring & Analytics

### Health Checks

The application includes comprehensive health monitoring:

- **Endpoint:** `/health`
- **Gemini API connectivity** status
- **WebRTC infrastructure** status
- **Database connectivity** (if configured)

### Logging

Structured logging with configurable levels:

```env
LOG_LEVEL=INFO
LOG_FORMAT=json
```

Log categories:
- **Voice Sessions:** Connection and quality metrics
- **Text Conversations:** Message processing and response times
- **Crisis Detection:** Safety intervention triggers
- **API Usage:** Rate limiting and error tracking

## 🧪 Development & Testing

### Local Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with hot reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Code formatting
black main.py
flake8 main.py
```

### Testing Voice Features

1. **Microphone Test:** Verify browser microphone access
2. **Audio Quality:** Test with different network conditions
3. **Voice Selection:** Try different Gemini voices
4. **Crisis Detection:** Test safety keyword recognition

### Environment Variables for Testing

```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_DOCS=true
```

## 📚 API Documentation

When running locally with `ENABLE_DOCS=true`, visit:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application interface |
| `/health` | GET | Health check and status |
| `/chat` | POST | Text-based chat API |
| `/ws` | WebSocket | Real-time text chat |
| `/webrtc/offer` | POST | WebRTC connection setup |
| `/api/voices` | GET | Available voice options |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Submit a pull request

### Code Standards

- **Python:** Follow PEP 8 with Black formatting
- **JavaScript:** Use ES6+ features with consistent formatting
- **Documentation:** Update README and docstrings
- **Testing:** Maintain >90% test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Resources

### Getting Help

- **Issues:** [GitHub Issues](https://github.com/yourusername/ai-psychologist-gemini/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/ai-psychologist-gemini/discussions)
- **Documentation:** [Wiki](https://github.com/yourusername/ai-psychologist-gemini/wiki)

### Mental Health Resources

**🚨 Crisis Resources:**
- **National Suicide Prevention Lifeline:** 988 (US)
- **Crisis Text Line:** Text HOME to 741741 (US)
- **International Association for Suicide Prevention:** [iasp.info](https://iasp.info/resources/Crisis_Centres/)

**Professional Support:**
- **Psychology Today:** [psychologytoday.com](https://www.psychologytoday.com)
- **BetterHelp:** [betterhelp.com](https://www.betterhelp.com)
- **Talkspace:** [talkspace.com](https://www.talkspace.com)

### Technical Resources

- **Gemini API Documentation:** [ai.google.dev](https://ai.google.dev/gemini-api/docs)
- **FastRTC Documentation:** [fastrtc.org](https://fastrtc.org)
- **Railway Documentation:** [docs.railway.com](https://docs.railway.com)
- **WebRTC Resources:** [webrtc.org](https://webrtc.org)

## 📈 Roadmap

### Version 2.1 (Current)
- ✅ Gemini 2.0 Flash integration
- ✅ Real-time voice streaming
- ✅ Crisis detection system
- ✅ Railway deployment

### Version 2.2 (Planned)
- 🔄 Multi-language support
- 🔄 Session persistence
- 🔄 Advanced analytics
- 🔄 Mobile app companion

### Version 3.0 (Future)
- 📋 EHR integration
- 📋 Provider dashboard
- 📋 Group therapy sessions
- 📋 AI-assisted diagnosis support

---

## ⚠️ Important Disclaimer

**This AI assistant is not a substitute for professional mental health services.** If you're experiencing a mental health crisis or need immediate support, please contact:

- **Emergency Services:** 911 (US)
- **National Suicide Prevention Lifeline:** 988 (US)
- **Crisis Text Line:** Text HOME to 741741 (US)

This application is designed to provide supportive conversation and evidence-based therapeutic techniques, but it cannot replace the expertise and care of licensed mental health professionals.

---

**Made with ❤️ for mental health support**

*Last updated: January 2025*
