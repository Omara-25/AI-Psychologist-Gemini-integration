# Contributing to AI Psychologist with Gemini Voice Integration

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🌟 Project Vision

This project aims to create a revolutionary therapeutic AI assistant that combines:
- **Advanced AI capabilities** from Google Gemini
- **Real-time voice interaction** for natural conversation
- **Evidence-based therapeutic techniques** from psychology
- **Ethical AI practices** with proper safeguards and privacy protection

## 🤝 How to Contribute

### Types of Contributions Welcome

- **🐛 Bug Reports** - Help us identify and fix issues
- **✨ Feature Requests** - Suggest new therapeutic features
- **📖 Documentation** - Improve guides and documentation
- **🧪 Testing** - Add tests or improve test coverage
- **🔒 Security** - Report security vulnerabilities responsibly
- **🎨 UI/UX** - Enhance user experience and accessibility
- **🌍 Accessibility** - Make the app more inclusive
- **🔧 Performance** - Optimize for better user experience

### What We're Looking For

**High Priority:**
- Crisis detection improvements
- Voice quality optimizations
- Mobile responsiveness
- Accessibility features (WCAG compliance)
- Multi-language support
- Integration with mental health APIs

**Medium Priority:**
- Advanced therapeutic frameworks (DBT, ACT, etc.)
- Session analytics and insights
- Provider dashboard features
- Integration with EHR systems

**Lower Priority:**
- Alternative AI model support
- Advanced visualization features
- Gamification elements

## 🚀 Getting Started

### 1. Development Setup

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/ai-psychologist-gemini.git
cd ai-psychologist-gemini

# Run the setup script
python setup.py

# Or manual setup:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Gemini API key
```

### 2. Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Add tests for new functionality
# Update documentation as needed

# Run tests
python run_tests.py
# or
pytest test_main.py -v

# Run the development server
python run_dev.py
# or
python main.py

# Test your changes at http://localhost:8000
```

### 3. Code Standards

#### Python Code
- **Follow PEP 8** style guidelines
- **Use Black** for code formatting: `black main.py test_main.py`
- **Use type hints** where appropriate
- **Write docstrings** for functions and classes
- **Maximum line length:** 88 characters (Black default)

```python
def process_therapeutic_response(
    user_input: str, 
    session_context: dict,
    temperature: float = 0.7
) -> str:
    """
    Process user input and generate therapeutic response.
    
    Args:
        user_input: User's message or question
        session_context: Current session state and history
        temperature: Response randomness (0.0-1.0)
        
    Returns:
        Generated therapeutic response
        
    Raises:
        ValueError: If temperature is out of range
    """
    # Implementation here
    pass
```

#### JavaScript/HTML/CSS
- **Use ES6+** modern JavaScript features
- **Consistent indentation** (2 spaces)
- **Semantic HTML** with proper accessibility attributes
- **Mobile-first** responsive design
- **Progressive enhancement** approach

#### Documentation
- **Clear, concise writing**
- **Include code examples** where helpful
- **Update README.md** for significant changes
- **Add inline comments** for complex logic

### 4. Testing Requirements

#### Unit Tests
- **Write tests** for new functionality
- **Maintain >90%** test coverage
- **Test edge cases** and error conditions
- **Mock external dependencies** (Gemini API, etc.)

```python
def test_crisis_detection():
    """Test that crisis keywords are properly detected"""
    crisis_input = "I want to hurt myself"
    result = detect_crisis_indicators(crisis_input)
    assert result.is_crisis is True
    assert "self-harm" in result.detected_keywords
```

#### Integration Tests
- **Test API endpoints** with various inputs
- **Test WebSocket connections** 
- **Test voice session lifecycle**
- **Test error handling paths**

#### Manual Testing Checklist
- [ ] Voice sessions connect successfully
- [ ] Audio quality is acceptable
- [ ] Text chat responds appropriately
- [ ] Crisis detection triggers correctly
- [ ] UI is responsive on mobile
- [ ] Accessibility features work
- [ ] Error messages are helpful

## 📋 Pull Request Process

### Before Submitting
1. **Ensure tests pass**: `pytest test_main.py -v`
2. **Check code formatting**: `black --check main.py`
3. **Update documentation** as needed
4. **Test manually** with the checklist above
5. **Verify no secrets** are committed

### PR Requirements
- **Clear title** describing the change
- **Detailed description** of what and why
- **Link to related issues** if applicable
- **Screenshots/videos** for UI changes
- **Breaking changes** clearly documented

### PR Template
```markdown
## What does this PR do?
Brief description of the changes

## Why is this change needed?
Context and motivation

## How has this been tested?
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] Tested on multiple browsers
- [ ] Tested voice functionality

## Screenshots (if applicable)
![Before and after screenshots]

## Checklist
- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] No sensitive data committed
```

## 🐛 Bug Reports

### Security Vulnerabilities
**Do NOT open public issues for security vulnerabilities.**

Instead:
1. Email: security@yourproject.com
2. Include detailed reproduction steps
3. Provide your contact information
4. We'll respond within 24 hours

### Regular Bug Reports
Use the GitHub issue template:

```markdown
## Bug Description
Clear description of the problem

## Steps to Reproduce
1. Go to...
2. Click on...
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 12.0]
- Browser: [e.g., Chrome 96]
- Device: [e.g., iPhone 13, Desktop]
- Version: [e.g., v2.1.0]

## Additional Context
Any other relevant information
```

## ✨ Feature Requests

### Guidelines for Good Feature Requests
- **Therapeutic value** - How does this help users?
- **Evidence-based** - Is this supported by psychology research?
- **Feasible** - Can this be implemented reasonably?
- **Accessible** - Does this maintain or improve accessibility?
- **Privacy-conscious** - Does this respect user privacy?

### Feature Request Template
```markdown
## Feature Summary
Brief description of the proposed feature

## Therapeutic Justification
How does this support mental health goals?

## User Story
As a [user type], I want [feature] so that [benefit]

## Acceptance Criteria
- [ ] Specific requirement 1
- [ ] Specific requirement 2
- [ ] Specific requirement 3

## Additional Context
Research, mockups, or other supporting information
```

## 🏗️ Architecture Guidelines

### Code Organization
```
├── main.py              # Main FastAPI application
├── models/              # Data models and schemas
├── handlers/            # WebRTC and voice handlers
├── services/            # Business logic services
├── utils/               # Helper functions
├── tests/               # Test files
├── static/              # Static assets
├── templates/           # HTML templates (if needed)
└── docs/                # Additional documentation
```

### Design Principles
1. **Privacy by Design** - Minimize data collection
2. **Fail-Safe Defaults** - Secure configurations by default
3. **Progressive Enhancement** - Works without JavaScript
4. **Accessibility First** - WCAG 2.1 AA compliance
5. **Therapeutic Ethics** - Evidence-based, professional boundaries

### Technology Choices
- **Backend**: FastAPI for modern Python API development
- **AI**: Google Gemini for advanced voice and text capabilities
- **WebRTC**: FastRTC library for real-time communication
- **Deployment**: Railway for simple, scalable hosting
- **Testing**: pytest for comprehensive test coverage

## 🔧 Development Tools

### Recommended Tools
- **IDE**: VS Code with Python and Prettier extensions
- **API Testing**: Postman or curl
- **WebRTC Testing**: Chrome DevTools WebRTC internals
- **Audio Testing**: Audacity for audio analysis

### Useful Commands
```bash
# Code formatting
black main.py test_main.py

# Linting
flake8 main.py

# Type checking
mypy main.py

# Security scanning
bandit -r .

# Run development server with hot reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests with coverage
pytest --cov=main test_main.py

# Build Docker image
docker build -t ai-psychologist .
```

## 🌍 Community Guidelines

### Code of Conduct
This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). 

**In summary:**
- **Be respectful** and inclusive
- **Focus on mental health** and user wellbeing
- **Provide constructive feedback**
- **Respect privacy** and confidentiality
- **Follow professional ethics** for mental health applications

### Communication Channels
- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and community chat
- **Discord** - Real-time community discussion (link in README)
- **Email** - security@yourproject.com for security issues

### Recognition
Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Annual contributor report**

## 📚 Learning Resources

### Mental Health & AI Ethics
- [APA Guidelines for AI in Psychology](https://www.apa.org/)
- [Responsible AI in Healthcare](https://www.who.int/publications/i/item/ethics-and-governance-of-artificial-intelligence-for-health)
- [Crisis Intervention Principles](https://www.suicidepreventionlifeline.org/)

### Technical Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [WebRTC Documentation](https://webrtc.org/getting-started/)
- [Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)

### Testing & Quality
- [pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)
- [Security Testing Guide](https://owasp.org/www-project-top-ten/)

## ❓ FAQ

### Q: How do I get a Gemini API key?
A: Visit [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key) to create a free API key.

### Q: Can I use a different AI model?
A: The architecture supports different models, but Gemini is optimized for voice. You could contribute support for other models.

### Q: How do I test voice features without a microphone?
A: Use browser developer tools to simulate microphone input, or contribute automated audio testing tools.

### Q: Is this HIPAA compliant?
A: The codebase includes privacy-by-design features, but full HIPAA compliance requires additional operational procedures. See the security documentation.

### Q: How can I contribute to crisis detection?
A: This is a sensitive area requiring mental health expertise. Please include references to psychological literature in your contributions.

### Q: Can I add features for specific mental health conditions?
A: Yes! Evidence-based features for anxiety, depression, PTSD, etc. are welcome. Please include research references.

---

## 🙏 Thank You

Your contributions help create better mental health support tools. Every improvement, no matter how small, can make a meaningful difference in someone's life.

**Remember**: This is a mental health application. Our code has the potential to impact vulnerable people. Let's build with compassion, responsibility, and excellence.

*Happy contributing!* 🧠💙

---

*For questions about contributing, feel free to open a GitHub Discussion or reach out to the maintainers.*
