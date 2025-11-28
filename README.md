# VoxAI - Intelligent Voice Interface

A real-time multilingual voice-powered conversational AI platform with RAG (Retrieval Augmented Generation) capabilities. Built with React and FastAPI, featuring browser-based speech recognition, AI-powered text-to-speech, and optimized for sub-2-second response times.

## 🚀 Features

### Core Capabilities
- **Real-time Voice Interaction**: Continuous voice chat with automatic speech detection
- **Multilingual Support**: English, Hindi, and Tamil language support
- **RAG-Powered Knowledge Base**: PDF document integration with semantic search
- **Intelligent Interruptions**: Users can interrupt AI responses mid-speech
- **Response Caching**: Optimized caching for faster repeated queries
- **Duplicate Detection**: Prevents processing duplicate queries
- **Multiple Voice Options**: 7+ voice options (Anushka, Vidya, Abhilash, Karun, Arya, Hitesh, Manisha)
- **Configurable Audio Quality**: 8kHz, 16kHz, 22kHz, 24kHz options

### Performance Optimizations
- **Sub-2-Second Response Time**: Optimized LLM generation targeting 2s response time
- **Parallel Processing**: Language detection and RAG retrieval run in parallel
- **Connection Pooling**: HTTP session pooling for reduced latency
- **Smart Context Limiting**: Optimized context size for faster processing
- **Efficient Token Management**: Greedy decoding with optimized parameters

## 🏗️ Architecture

### Frontend (React)
- **Framework**: React 18.2.0
- **Speech Recognition**: Browser Speech Recognition API (react-speech-recognition)
- **HTTP Client**: Axios
- **UI Components**: React Icons
- **Port**: 3000

### Backend (FastAPI)
- **Framework**: FastAPI 0.104.1
- **LLM**: IBM Watsonx (Llama 3.3 70B Instruct)
- **TTS**: Sarvam AI (Bulbul v2 model)
- **RAG**: ChromaDB with Sentence Transformers
- **PDF Processing**: PyPDF2
- **Port**: 8000

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- IBM Watsonx API credentials
- Sarvam AI API key (for TTS)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd voice_bot
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### 4. Environment Configuration

Create a `.env` file in the `backend` directory:

```env
# IBM Watsonx Configuration
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Sarvam AI Configuration (for TTS)
SARVAM_API_KEY=your_sarvam_api_key
```

### 5. Knowledge Base Setup

Place your PDF knowledge base file (`TallyKnowledge.pdf` or any PDF) in the `backend` directory. The system will automatically load it on startup.

## 🚀 Running the Application

### Start Backend Server

```bash
cd backend

# Activate virtual environment if not already active
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Run the server
python main.py
```

The backend will be available at `http://localhost:8000`

### Start Frontend Development Server

```bash
cd frontend

# Start React development server
npm start
```

The frontend will be available at `http://localhost:3000`

## 📖 Usage

### Voice Chat Modes

1. **Continuous Voice Chat**: Click "START CONTINUOUS VOICE CHAT" for hands-free conversation
2. **Manual Mode**: Use "SEND CURRENT SPEECH" to send transcribed text manually
3. **Stop Chat**: Click "STOP VOICE CHAT" to stop listening

### Voice Settings

- **Voice Selection**: Choose from available voices (Vidya, Anushka, Abhilash, Karun, etc.)
- **Audio Quality**: Select sample rate (8kHz for speed, 24kHz for quality)
- **Language**: Automatically detected from speech input

### Features

- **Interruptions**: Speak while AI is responding to interrupt and ask a new question
- **Conversation History**: View full conversation history in the chat panel
- **Clear Conversation**: Use "CLEAR CONVERSATION" to reset chat history

## 🔌 API Endpoints

### Chat Endpoint
```
POST /chat
Body: {
  "messages": [...],
  "user_input": "your question"
}
```

### Text-to-Speech Endpoint
```
POST /tts
Body: {
  "text": "text to convert",
  "target_language_code": "en-IN",
  "speaker": "vidya",
  "speech_sample_rate": 8000
}
```

### Health Check
```
GET /health
```

### Status
```
GET /status
```

### Clear Conversation
```
POST /clear-conversation
```

## 🎯 Performance Metrics

- **LLM Response Time**: Target < 2 seconds
- **Total Endpoint Time**: Target < 2.5 seconds (including TTS)
- **RAG Retrieval**: < 0.3 seconds
- **Language Detection**: Parallel with RAG retrieval

## 🛠️ Technology Stack

### Frontend
- React 18.2.0
- Axios 1.6.0
- react-speech-recognition 3.10.0
- react-icons 4.12.0

### Backend
- FastAPI 0.104.1
- Uvicorn 0.24.0
- ChromaDB 0.4.22
- Sentence Transformers 2.2.2
- PyPDF2 3.0.1
- LangChain Text Splitters 0.0.1
- Requests 2.31.0

### AI Services
- IBM Watsonx (Llama 3.3 70B Instruct)
- Sarvam AI (Bulbul v2 TTS)
- Browser Speech Recognition API

## 🔍 Key Optimizations

1. **Prompt Optimization**: Minimal system prompts, limited context size
2. **Token Management**: Optimized max_new_tokens (60), greedy decoding
3. **Caching**: Response and embedding caching for repeated queries
4. **Parallel Processing**: Language detection and RAG run simultaneously
5. **Connection Pooling**: HTTP session reuse for faster API calls
6. **Context Limiting**: Single chunk retrieval (100 chars) for speed
7. **Conversation History**: Limited to last 2 messages to reduce prompt size

## 🐛 Troubleshooting

### Backend Issues

**Port Already in Use**
```bash
# Change port in main.py or use:
uvicorn main:app --port 8001
```

**Watsonx Authentication Failed**
- Verify `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` in `.env`
- Check API key validity

**ChromaDB Errors**
- Delete `backend/chroma_db` folder and restart
- Ensure write permissions in backend directory

### Frontend Issues

**Speech Recognition Not Working**
- Ensure browser supports Speech Recognition API (Chrome, Edge recommended)
- Check microphone permissions
- Use HTTPS or localhost (required for Speech Recognition API)

**TTS Not Playing**
- Verify `SARVAM_API_KEY` is set correctly
- Check browser console for errors
- Ensure audio format is supported

### Performance Issues

**Slow Response Times**
- Check network connection
- Verify API keys are valid
- Reduce audio quality (use 8kHz)
- Check backend logs for bottlenecks

## 📝 Configuration Options

### Backend Configuration

Edit `backend/main.py` to customize:
- `max_new_tokens`: Maximum tokens generated (default: 60)
- `temperature`: LLM temperature (default: 0.1)
- `repetition_penalty`: Repetition penalty (default: 1.5)
- Context chunk size (default: 100 chars)
- Cache size (default: 100 entries)

### Frontend Configuration

Edit `frontend/src/App.js` to customize:
- Default voice selection
- Audio quality default
- Speech recognition timeout
- API base URL

## 🔐 Security Notes

- API keys are stored in `.env` file (never commit to git)
- CORS is configured for localhost only
- No authentication required for local development
- Add authentication middleware for production deployment

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Support

[Add support/contact information here]

## 🎉 Acknowledgments

- IBM Watsonx for LLM capabilities
- Sarvam AI for high-quality TTS
- ChromaDB for vector database
- Sentence Transformers for embeddings

---

**Built with ❤️ using React and FastAPI**

