# AI Psychologist Gemini Integration - Deployment Plan

This document provides a comprehensive deployment plan for integrating Google Gemini's real-time voice capabilities with your existing AI psychologist application.

## 🎯 Integration Overview

### What Changed
- **Replaced Groq** with Google Gemini 2.0 Flash for superior voice capabilities
- **Added real-time voice streaming** using FastRTC and WebRTC
- **Enhanced therapeutic framework** with voice-aware conversation patterns
- **Improved crisis detection** across both text and voice modalities
- **Unified interface** supporting both text and voice interactions

### Key Benefits
- **Sub-250ms voice latency** for natural conversation flow
- **5 therapeutic voices** optimized for mental health support
- **Real-time audio visualization** with professional-grade quality
- **Seamless mode switching** between text and voice therapy
- **Enhanced security** with end-to-end voice encryption

## 📋 Pre-Deployment Checklist

### 1. Environment Requirements
- [ ] **Python 3.10+** installed
- [ ] **Railway CLI** installed (`npm install -g @railway/cli`)
- [ ] **Gemini API key** obtained from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)
- [ ] **Modern browser** with WebRTC support for testing
- [ ] **Microphone access** for voice feature testing

### 2. Railway Account Setup
- [ ] **Railway account** created at [railway.app](https://railway.app)
- [ ] **GitHub repository** connected to Railway
- [ ] **Domain configuration** (optional but recommended)
- [ ] **Environment variables** access configured

### 3. API Keys and Credentials
- [ ] **Gemini API key** with sufficient quota
- [ ] **Secret keys** generated for JWT and session management
- [ ] **Crisis webhook URL** configured (if using external crisis system)
- [ ] **Analytics keys** (optional)

## 🚀 Step-by-Step Deployment

### Phase 1: Local Setup and Testing

#### 1.1 Clone and Configure
```bash
# Clone your repository
git clone https://github.com/yourusername/ai-psychologist-gemini.git
cd ai-psychologist-gemini

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 1.2 Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env
```

**Minimum required configuration:**
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8000
ENVIRONMENT=development
SECRET_KEY=your_secret_key_here
```

#### 1.3 Local Testing
```bash
# Start development server
python main.py

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/api/voices
```

**Testing checklist:**
- [ ] Server starts without errors
- [ ] Health endpoint returns "healthy"
- [ ] Main page loads correctly
- [ ] Voice selection dropdown populates
- [ ] Text chat responds to messages
- [ ] Microphone permission prompt appears

### Phase 2: Railway Deployment

#### 2.1 Railway Project Setup
```bash
# Login to Railway
railway login

# Create new project
railway init

# Or connect existing project
railway link [project-id]
```

#### 2.2 Environment Variables Configuration

**Required Variables:**
```bash
railway variables set GEMINI_API_KEY="your_gemini_api_key_here"
railway variables set SECRET_KEY="$(openssl rand -base64 32)"
railway variables set ENVIRONMENT="production"
railway variables set PORT="8000"
```

**Optional but Recommended:**
```bash
railway variables set LOG_LEVEL="INFO"
railway variables set MAX_CONNECTIONS_PER_IP="10"
railway variables set ENABLE_CRISIS_DETECTION="true"
railway variables set CORS_ORIGINS="https://yourdomain.com"
```

#### 2.3 Deploy to Railway
```bash
# Deploy current branch
railway up

# Or deploy specific branch
railway up --branch main
```

**Deployment verification:**
- [ ] Deployment completes successfully
- [ ] Health check passes
- [ ] Application URL is accessible
- [ ] WebRTC connection establishes
- [ ] Voice sessions work correctly

### Phase 3: Domain and SSL Setup

#### 3.1 Custom Domain (Recommended)
```bash
# Add custom domain
railway domain add yourdomain.com

# Verify DNS configuration
dig yourdomain.com
```

#### 3.2 SSL Certificate
Railway automatically provides SSL certificates via Let's Encrypt. Verify:
- [ ] HTTPS redirects work
- [ ] SSL certificate is valid
- [ ] WebRTC works over HTTPS
- [ ] Mixed content warnings absent

### Phase 4: Production Optimization

#### 4.1 Performance Configuration
```env
# Production optimizations
WORKERS=1
MAX_WORKER_CONNECTIONS=1000
KEEPALIVE_TIMEOUT=5
AUDIO_SAMPLE_RATE=24000
```

#### 4.2 Monitoring Setup
```bash
# Enable structured logging
railway variables set LOG_FORMAT="json"

# Set up health check monitoring
railway variables set HEALTHCHECK_INTERVAL="30"
```

#### 4.3 Security Hardening
```env
# Security settings
DEBUG=false
ENABLE_DOCS=false
CORS_ORIGINS="https://yourdomain.com"
MAX_REQUESTS_PER_MINUTE=60
```

## 🔧 Configuration Deep Dive

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | - | ✅ |
| `PORT` | Server port | 8000 | ✅ |
| `ENVIRONMENT` | Deployment environment | development | ✅ |
| `SECRET_KEY` | Application secret | - | ✅ |
| `USE_VERTEX_AI` | Use Vertex AI instead of AI Studio | false | ❌ |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (for Vertex AI) | - | ❌ |
| `LOG_LEVEL` | Logging verbosity | INFO | ❌ |
| `MAX_CONNECTIONS_PER_IP` | Rate limiting | 10 | ❌ |
| `ENABLE_CRISIS_DETECTION` | Crisis keyword detection | true | ❌ |
| `CORS_ORIGINS` | Allowed CORS origins | * | ❌ |

### Voice Configuration Options

```python
# Available voices and their characteristics
VOICE_OPTIONS = {
    "Aoede": {
        "description": "Melodic and soothing - recommended for therapy",
        "tone": "calm, musical, reassuring",
        "best_for": "anxiety, depression, general therapy"
    },
    "Kore": {
        "description": "Gentle and empathetic",
        "tone": "soft, understanding, nurturing", 
        "best_for": "trauma, grief, emotional support"
    },
    "Charon": {
        "description": "Calm and reassuring",
        "tone": "steady, grounding, professional",
        "best_for": "crisis intervention, CBT sessions"
    },
    "Puck": {
        "description": "Warm and conversational",
        "tone": "friendly, approachable, optimistic",
        "best_for": "motivation, behavioral change"
    },
    "Fenrir": {
        "description": "Strong and supportive", 
        "tone": "confident, empowering, direct",
        "best_for": "addiction recovery, life coaching"
    }
}
```

## 🔍 Monitoring & Maintenance

### Health Monitoring
```bash
# Check application health
curl https://yourapp.railway.app/health

# Monitor logs
railway logs --follow

# Check resource usage
railway status
```

### Performance Metrics
Monitor these key metrics:
- **Response time:** < 2 seconds for text, < 250ms for voice
- **Error rate:** < 1% overall
- **WebRTC connection success:** > 95%
- **API quota usage:** Monitor Gemini API limits
- **Memory usage:** < 80% of allocated resources

### Log Analysis
```json
{
  "timestamp": "2025-01-04T10:30:00Z",
  "level": "INFO",
  "service": "ai-psychologist",
  "event": "voice_session_started",
  "session_id": "sess_123",
  "voice": "Aoede",
  "duration": 0,
  "quality_metrics": {
    "latency_ms": 180,
    "packet_loss": 0.01,
    "jitter_ms": 5
  }
}
```

## 🛠️ Troubleshooting Guide

### Common Issues and Solutions

#### 1. WebRTC Connection Failed
**Symptoms:** Voice sessions don't start, connection timeout
**Solutions:**
```bash
# Check TURN server configuration
railway variables get | grep TURN

# Verify firewall settings
curl -I https://yourapp.railway.app/webrtc/offer

# Test with different browsers
# Chrome DevTools > Console for WebRTC errors
```

#### 2. Gemini API Errors
**Symptoms:** Chat responses fail, quota exceeded errors
**Solutions:**
```bash
# Check API key validity
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
  https://generativelanguage.googleapis.com/v1/models

# Monitor quota usage
railway logs --filter "quota"

# Implement retry logic (already included)
```

#### 3. Audio Quality Issues
**Symptoms:** Choppy audio, high latency, connection drops
**Solutions:**
```env
# Optimize audio settings
AUDIO_SAMPLE_RATE=16000  # Lower for better performance
AUDIO_CHANNELS=1
ENABLE_NOISE_SUPPRESSION=true

# Adjust WebRTC settings
MAX_BITRATE=64000
PACKET_LOSS_TOLERANCE=0.05
```

#### 4. High Resource Usage
**Symptoms:** Slow responses, memory warnings
**Solutions:**
```bash
# Scale up resources
railway up --memory 2GB --cpu 2

# Optimize worker configuration
railway variables set WORKERS=1
railway variables set MAX_WORKER_CONNECTIONS=500
```

### Debug Mode
```env
# Enable debug logging
LOG_LEVEL=DEBUG
DEBUG=true
ENABLE_DOCS=true
```

## 📊 Scaling and Performance

### Horizontal Scaling
```toml
# railway.toml
[deploy.replicas]
min = 1
max = 5

[deploy.autoscaling]
enabled = true
target_cpu = 70
target_memory = 80
```

### Load Balancing
Railway automatically handles load balancing, but consider:
- **Session affinity** for WebRTC connections
- **Health check optimization** for faster failover
- **Geographic distribution** for global users

### Cost Optimization
```bash
# Monitor usage
railway metrics

# Optimize resource allocation
railway variables set WORKERS=1  # Single worker for WebRTC state
railway variables set KEEPALIVE_TIMEOUT=5  # Faster connection cleanup
```

## 🔐 Security Best Practices

### API Key Security
```bash
# Rotate API keys regularly
railway variables set GEMINI_API_KEY="new_key_here"

# Use Railway's secret management
railway variables set --secret GEMINI_API_KEY="your_key"
```

### Network Security
```env
# Restrict CORS origins
CORS_ORIGINS="https://yourdomain.com,https://www.yourdomain.com"

# Enable rate limiting
MAX_CONNECTIONS_PER_IP=10
MAX_REQUESTS_PER_MINUTE=60
```

### Data Protection
```env
# Enable encryption
ENCRYPT_SESSIONS=true
SESSION_TIMEOUT=3600

# Audit logging
ENABLE_AUDIT_LOGS=true
LOG_PERSONAL_DATA=false
```

## 📈 Post-Deployment Checklist

### Functional Testing
- [ ] **Voice sessions** start and end cleanly
- [ ] **Text chat** responds appropriately
- [ ] **Crisis detection** triggers correctly
- [ ] **Mode switching** works seamlessly
- [ ] **Audio visualization** displays properly
- [ ] **Mute/unmute** functions correctly
- [ ] **Browser compatibility** across major browsers

### Performance Testing
- [ ] **Response times** meet requirements (< 2s text, < 250ms voice)
- [ ] **Concurrent users** handled appropriately
- [ ] **Memory usage** remains stable
- [ ] **WebRTC quality** maintains high standards
- [ ] **Error rates** stay below 1%

### Security Testing
- [ ] **API keys** not exposed in client
- [ ] **HTTPS** enforced everywhere
- [ ] **CORS** properly configured
- [ ] **Rate limiting** effective
- [ ] **Input validation** prevents injection

### User Experience Testing
- [ ] **Onboarding** flow is clear
- [ ] **Voice quality** is therapeutic-grade
- [ ] **Interface** is intuitive
- [ ] **Error messages** are helpful
- [ ] **Crisis support** information is visible

## 🎉 Go-Live Process

### 1. Final Pre-Launch
```bash
# Final deployment
railway up --production

# Smoke test all features
curl https://yourapp.railway.app/health
# Test voice session manually
# Test text chat functionality
# Verify crisis detection
```

### 2. DNS and Domain
```bash
# Update DNS records
# A record: yourapp.railway.app
# CNAME: www -> yourapp.railway.app

# Verify propagation
dig +trace yourdomain.com
```

### 3. Monitoring Setup
```bash
# Set up monitoring alerts
railway notifications add --webhook https://yourmonitoring.com/webhook

# Configure log shipping (if needed)
railway addons create logdna
```

### 4. Documentation Update
- [ ] Update README with production URL
- [ ] Document environment variables
- [ ] Create user guide
- [ ] Set up support channels

## 📞 Support and Maintenance

### Regular Maintenance Tasks
- **Weekly:** Review logs and performance metrics
- **Monthly:** Update dependencies and security patches
- **Quarterly:** API key rotation and security audit
- **Annually:** Full security assessment

### Support Channels
- **Technical Issues:** GitHub Issues
- **Deployment Questions:** Railway Community
- **Gemini API:** Google AI Support
- **Mental Health Crisis:** Professional resources in README

### Backup and Recovery
```bash
# Database backups (if applicable)
railway db backup

# Configuration backup
railway variables list > env_backup.txt

# Code backup
git tag v2.1.0
git push origin v2.1.0
```

---

## 🎯 Success Metrics

### Technical KPIs
- **Uptime:** > 99.9%
- **Voice Latency:** < 250ms average
- **Text Response:** < 2s average
- **Error Rate:** < 0.5%
- **WebRTC Success:** > 98%

### User Experience KPIs
- **Session Duration:** Track therapeutic engagement
- **Mode Usage:** Text vs voice preference
- **Crisis Interventions:** Successful safety measures
- **User Retention:** Return usage patterns

### Business KPIs
- **Cost per Session:** Monitor API and infrastructure costs
- **Scaling Efficiency:** Resource utilization optimization
- **Support Volume:** Minimize through good UX

---

**🎉 Congratulations!** Your AI Psychologist with Gemini Voice Integration is now ready for production use. Remember to monitor performance closely in the first few weeks and be prepared to make adjustments based on real user feedback.

For ongoing support, refer to the main README.md and don't hesitate to reach out through the established support channels.

*Deployment Plan v2.1 - January 2025*
