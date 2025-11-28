from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
from dotenv import load_dotenv
from datetime import datetime
import base64
from io import BytesIO
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from collections import deque
import re

# Load environment variables
load_dotenv()

app = FastAPI(title="Voice Bot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
bot_name = "ava"
conversation_history = []
bearer_token = None

# STT/Response management variables
last_processed_query = None
ongoing_request_cancelled = False
recent_requests = deque(maxlen=10)  # Track recent requests for deduplication (last 10)
ongoing_request_task = None
request_lock = None  # Will be initialized on first use

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Caching for latency optimization
response_cache: Dict[str, str] = {}  # Cache for LLM responses
embedding_cache: Dict[str, List[float]] = {}  # Cache for query embeddings
MAX_CACHE_SIZE = 100  # Limit cache size to prevent memory issues

# HTTP Session with connection pooling for reduced latency - optimized for 2s target
session = requests.Session()
retry_strategy = Retry(
    total=1,  # Reduced retries for speed
    backoff_factor=0.05,  # Faster backoff
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=40)  # More connections for speed
session.mount("http://", adapter)
session.mount("https://", adapter)

# Sarvam AI Configuration
sarvam_api_key = os.getenv("SARVAM_API_KEY")
sarvam_stt_api_key = os.getenv("SARVAM_API_KEY_STT")

# RAG Components
chroma_client = None
embedding_model = None
text_splitter = None
pdf_collection = None

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: List[Message]
    user_input: str

class TTSRequest(BaseModel):
    text: str
    target_language_code: Optional[str] = "hi-IN"
    speaker: Optional[str] = None  # Will use the voice selected in frontend
    speech_sample_rate: Optional[int] = 22050  # Audio quality/sample rate

class STTRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    language_code: Optional[str] = "unknown"  # "unknown" for auto-detect

# Initialize RAG components on startup
@app.on_event("startup")
async def startup_event():
    global bearer_token, chroma_client, embedding_model, text_splitter, pdf_collection
    
    print("\n" + "="*80)
    print("🚀 STARTING VOICE BOT API")
    print("="*80)
    
    # Initialize authentication
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    
    if api_key and project_id:
        bearer_token = get_bearer_token(api_key)
        if bearer_token:
            print("✅ Watsonx authentication successful!")
        else:
            print("❌ Watsonx authentication failed!")
    else:
        print("❌ Missing WATSONX_API_KEY or WATSONX_PROJECT_ID in environment variables")
    
    # Initialize RAG components
    try:
        print("\n📚 Initializing RAG components...")
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Get or create collection
        pdf_collection = chroma_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Clear all documents on startup for fresh start
        try:
            pdf_collection.delete(where={})
            print("✅ ChromaDB cleared on startup!")
        except Exception as e:
            print(f"ℹ️ No existing documents to clear: {e}")
        
        print("✅ RAG components initialized successfully!")
        
        # Load TallyKnowledge.pdf automatically on startup
        await load_tally_knowledge()
        
    except Exception as e:
        print(f"❌ Error initializing RAG components: {e}")
    
    # Check Sarvam TTS configuration
    if sarvam_api_key:
        print("✅ Sarvam AI TTS configured (Multiple voices: Anushka, Vidya, Abhilash, Karun)")
    else:
        print("⚠️ SARVAM_API_KEY not found - TTS will not be available")
    
    # Sarvam STT is no longer used - browser STT is used instead
    # if sarvam_stt_api_key:
    #     print("✅ Sarvam AI STT configured (Saarika v2.5 model)")
    # else:
    #     print("⚠️ SARVAM_API_KEY_STT not found - STT will not be available")
    print("ℹ️  STT: Browser Speech Recognition (Sarvam STT not used)")
    
    print("\n" + "="*80)
    print("🎉 VOICE BOT API IS READY!")
    print("="*80)
    print("📍 Backend: http://localhost:8000")
    print("📍 Frontend: http://localhost:3000")
    print("🎤 STT: Browser Speech Recognition (Multilingual)")
    print("🔊 TTS: Sarvam AI (Multiple voices available)")
    print("🤖 LLM: IBM Watsonx (Llama 3.3 70B)")
    print("📖 RAG: TallyKnowledge.pdf (Auto-loaded)")
    print("="*80 + "\n")

async def load_tally_knowledge():
    """Load TallyKnowledge.pdf into RAG system on startup"""
    try:
        tally_pdf_path = os.path.join(os.path.dirname(__file__), "TallyKnowledge.pdf")
        
        if not os.path.exists(tally_pdf_path):
            print(f"⚠️ TallyKnowledge.pdf not found at {tally_pdf_path}")
            return
        
        print(f"\n📚 Loading TallyKnowledge.pdf into RAG system...")
        print(f"   Path: {tally_pdf_path}")
        
        # Read PDF file
        with open(tally_pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            total_pages = len(pdf_reader.pages)
            print(f"   Total pages: {total_pages}")
            
            # Extract text from all pages
            for i, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
                if (i + 1) % 50 == 0:  # Progress update every 50 pages
                    print(f"   Processed {i + 1}/{total_pages} pages...")
        
        print(f"   Extracted text length: {len(text)} characters")
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        print(f"   Generated {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"   Generating embeddings...")
        embeddings = embedding_model.encode(chunks)
        
        # Prepare documents for storage
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "filename": "TallyKnowledge.pdf",
                "chunk_index": i,
                "source": "pdf",
                "uploaded_at": datetime.now().isoformat()
            })
            ids.append(f"TallyKnowledge_{i}")
        
        # Add to collection
        print(f"   Storing in ChromaDB...")
        pdf_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        
        print(f"✅ TallyKnowledge.pdf loaded successfully!")
        print(f"   {len(chunks)} chunks indexed and ready for retrieval\n")
        
    except Exception as e:
        print(f"❌ Error loading TallyKnowledge.pdf: {e}")
        import traceback
        traceback.print_exc()

def get_bearer_token(api_key: str) -> Optional[str]:
    """Get bearer token for Watsonx API authentication"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"

    try:
        response = session.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            print(f"Failed to retrieve access token: {response.text}")
            return None
    except Exception as e:
        print(f"Error getting bearer token: {e}")
        return None

def clean_ai_response(response_text: str, user_input: str = None) -> str:
    """Clean the AI response by removing template tags and unwanted text"""
    if not response_text:
        return response_text
    
    # Remove common template tags
    unwanted_patterns = [
        "assistant<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "**",
        "assistant<|end_header_id|>\n\n",
        "assistant<|end_header_id|>\n",
    ]
    
    cleaned_response = response_text
    for pattern in unwanted_patterns:
        cleaned_response = cleaned_response.replace(pattern, "")
    
    # If user input is provided and appears at the start of response, remove it
    # This handles cases where the model repeats the user's question
    if user_input:
        user_input_clean = user_input.strip().lower()
        response_lower = cleaned_response.strip().lower()
        
        # Check if response starts with user input (case-insensitive)
        if response_lower.startswith(user_input_clean):
            # Remove the user input from the beginning
            cleaned_response = cleaned_response[len(user_input):].strip()
        else:
            # Try to find and remove user input if it appears at the start (with variations)
            # Look for user input words at the beginning
            user_words = user_input_clean.split()
            response_words = response_lower.split()
            
            # If first few words match user input, remove them
            if len(user_words) > 0 and len(response_words) >= len(user_words):
                match_count = 0
                for i in range(min(len(user_words), len(response_words))):
                    if user_words[i] in response_words[i] or response_words[i] in user_words[i]:
                        match_count += 1
                    else:
                        break
                
                # If most of the first words match, remove them
                if match_count >= min(3, len(user_words)) or (match_count > 0 and match_count == len(user_words)):
                    # Remove matching words from the beginning
                    words_to_remove = match_count
                    cleaned_response = ' '.join(cleaned_response.split()[words_to_remove:]).strip()
    
    # Remove leading/trailing whitespace and newlines
    cleaned_response = cleaned_response.strip()
    
    # Log the cleaned response for debugging
    print(f"📝 Cleaned AI Response Length: {len(cleaned_response)} characters")
    print(f"📝 Cleaned AI Response Preview: {cleaned_response[:200]}...")
    print(f"📝 Full Cleaned Response: {cleaned_response}")
    
    return cleaned_response

def is_complete_sentence(text: str) -> bool:
    """Check if the input is a complete sentence (not a fragment)"""
    if not text or not text.strip():
        return False
    
    # Count words
    words = text.strip().split()
    word_count = len(words)
    
    # Rule 2: Ignore if too short (less than 5-6 words)
    if word_count < 5:
        print(f"⚠️ Input too short ({word_count} words), ignoring fragment")
        return False
    
    # Check for sentence-ending punctuation
    text_stripped = text.strip()
    has_ending_punctuation = text_stripped.endswith(('.', '!', '?', '।', '؟'))
    
    # Check for question words (indicates complete question)
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'क्या', 'कब', 'कहाँ', 'कौन', 'क्यों', 'कैसे', 'என்ன', 'எப்போது', 'எங்கே', 'யார்', 'ஏன்', 'எப்படி']
    has_question_word = any(word.lower() in text.lower() for word in question_words)
    
    # Consider complete if:
    # 1. Has ending punctuation, OR
    # 2. Has question word and at least 5 words, OR
    # 3. Has at least 6 words (likely complete even without punctuation)
    is_complete = has_ending_punctuation or (has_question_word and word_count >= 5) or word_count >= 6
    
    if not is_complete:
        print(f"⚠️ Input appears incomplete ({word_count} words, no ending punctuation), ignoring fragment")
    
    return is_complete

def is_duplicate_query(text: str, time_window: float = 1.0) -> bool:
    """Check if this query is a duplicate of a recent query (within time_window seconds)"""
    import time
    current_time = time.time()
    
    # Rule 1: Check if identical to last processed query
    if last_processed_query and text.strip().lower() == last_processed_query.strip().lower():
        print(f"⚠️ Duplicate query detected (identical to last), ignoring")
        return True
    
    # Rule 4 & 6: Check for similar queries within 1 second
    for req_time, req_text in recent_requests:
        time_diff = current_time - req_time
        if time_diff <= time_window:
            # Check if texts are very similar (normalized comparison)
            text_normalized = re.sub(r'[^\w\s]', '', text.strip().lower())
            req_text_normalized = re.sub(r'[^\w\s]', '', req_text.strip().lower())
            
            # If texts are identical or very similar (90%+ match)
            if text_normalized == req_text_normalized:
                print(f"⚠️ Duplicate query detected (within {time_diff:.2f}s), ignoring")
                return True
            elif len(text_normalized) > 0 and len(req_text_normalized) > 0:
                # Check similarity (simple word overlap)
                text_words = set(text_normalized.split())
                req_words = set(req_text_normalized.split())
                if len(text_words) > 0 and len(req_words) > 0:
                    overlap = len(text_words & req_words) / max(len(text_words), len(req_words))
                    if overlap > 0.9:  # 90% word overlap
                        print(f"⚠️ Very similar query detected (within {time_diff:.2f}s, {overlap*100:.1f}% overlap), ignoring")
                        return True
    
    return False

def detect_language(text: str) -> str:
    """Detect the language of the input text"""
    # Check for Hindi (Devanagari script)
    if any('\u0900' <= char <= '\u097F' for char in text):
        return "Hindi"
    
    # Check for Tamil (Tamil script)
    if any('\u0B80' <= char <= '\u0BFF' for char in text):
        return "Tamil"
    
    # Default to English
    return "English"

def get_language_name(text: str) -> str:
    """Get the language name for the input text"""
    lang = detect_language(text)
    language_names = {
        "Hindi": "Hindi (हिंदी)",
        "Tamil": "Tamil (தமிழ்)",
        "English": "English"
    }
    return language_names.get(lang, "English")

def retrieve_relevant_context(query: str, top_k: int = 2) -> str:
    """Retrieve relevant context from TallyKnowledge based on query - ULTRA-OPTIMIZED WITH CACHING"""
    try:
        if not pdf_collection:
            return ""
        
        # Normalize query for cache key (lowercase, strip whitespace)
        query_normalized = query.lower().strip()
        cache_key = hashlib.md5(query_normalized.encode()).hexdigest()
        
        # Check embedding cache first
        if cache_key in embedding_cache:
            query_embedding = [embedding_cache[cache_key]]
            print(f"⚡ Using cached embedding for query")
        else:
            # Generate query embedding
            query_embedding = embedding_model.encode([query])
            # Cache the embedding
            if len(embedding_cache) < MAX_CACHE_SIZE:
                embedding_cache[cache_key] = query_embedding[0].tolist()
            else:
                # Clear old cache entries (simple FIFO)
                oldest_key = next(iter(embedding_cache))
                del embedding_cache[oldest_key]
                embedding_cache[cache_key] = query_embedding[0].tolist()
        
        # Search for similar documents - REDUCED to 1 chunk for maximum speed (2s target)
        results = pdf_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=1  # Only 1 chunk for maximum speed
        )
        
        if results['documents'] and results['documents'][0]:
            # Combine relevant chunks - limit total context size for 2s target
            relevant_chunks = results['documents'][0]
            # Limit chunk to 100 chars for fastest LLM processing (target: 2s)
            limited_chunks = [chunk[:100] for chunk in relevant_chunks]
            context = "\n".join(limited_chunks)  # Minimal separator for speed
            print(f"📖 Retrieved {len(relevant_chunks)} relevant chunk (optimized for 2s LLM target)")
            return context
        else:
            print(f"⚠️ No relevant context found for query")
            return ""
    except Exception as e:
        print(f"❌ Error retrieving context: {e}")
        return ""

async def get_watsonx_response_async(history: List[Message], user_input: str, cancellation_token: asyncio.Event = None) -> str:
    """Get response from Watsonx API with RAG context and multilingual support - ASYNC OPTIMIZED WITH CACHING
    
    Args:
        history: Conversation history
        user_input: User's input text
        cancellation_token: Event to signal cancellation (Rule 3: Handle interruptions)
    """
    import time
    global bearer_token, response_cache, ongoing_request_cancelled
    
    # Rule 3: Check if request was cancelled (interruption)
    if cancellation_token and cancellation_token.is_set():
        print("⚠️ Request cancelled due to interruption")
        return None
    
    if ongoing_request_cancelled:
        print("⚠️ Request cancelled due to interruption")
        ongoing_request_cancelled = False
        return None
    
    total_start = time.time()
    print(f"\n{'='*80}")
    print(f"⏱️  STARTING LLM REQUEST - Tracking Performance")
    print(f"{'='*80}")
    
    if not bearer_token:
        return "Error: Not authenticated with Watsonx API"
    
    # Check response cache first (normalize query for cache key)
    query_normalized = user_input.lower().strip()
    cache_key = hashlib.md5(query_normalized.encode()).hexdigest()
    
    if cache_key in response_cache:
        cached_response = response_cache[cache_key]
        print(f"⚡ CACHE HIT - Returning cached response (instant)")
        total_time = time.time() - total_start
        print(f"✅ TOTAL LLM PROCESSING TIME (CACHED): {total_time:.3f}s")
        return cached_response
    
    print(f"💾 CACHE MISS - Generating new response")
    
    # PARALLEL PROCESSING: Detect language and retrieve context simultaneously
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run language detection and RAG retrieval in parallel (optimized)
    parallel_start = time.time()
    detected_language_task = loop.run_in_executor(executor, detect_language, user_input)
    rag_context_task = loop.run_in_executor(executor, retrieve_relevant_context, user_input, 1)  # Only 1 chunk
    
    # Wait for both to complete
    detected_language, relevant_context = await asyncio.gather(detected_language_task, rag_context_task)
    parallel_time = time.time() - parallel_start
    
    print(f"⏱️  [1] Language Detection + RAG Retrieval: {parallel_time:.3f}s")
    print(f"🌐 Detected user language: {detected_language}")
    
    # Calculate remaining time budget for LLM (target: 2s total LLM time)
    remaining_budget = 2.0 - parallel_time
    if remaining_budget > 0:
        print(f"⏱️  Remaining time budget for LLM: {remaining_budget:.3f}s")
    else:
        print(f"⚠️  RAG took {parallel_time:.3f}s, may impact 2s LLM target")
    
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    # OPTIMIZED system message for LLM speed target (2s) - minimal prompt
    system_message = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "Tally expert. Same language. Answer in 1-2 sentences ONLY. Maximum 40 words. Be concise."
        "<|eot_id|>\n"
    )
    
    # Limit conversation history to last 2 messages only (reduce prompt size)
    limited_history = history[-2:] if len(history) > 2 else history
    
    # Construct the conversation history
    conversation = system_message + "".join(
        f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>\n" 
        for msg in limited_history
    )
    
    # Add context if available - minimal format
    context_prompt = ""
    if relevant_context:
        context_prompt = f"\nContext: {relevant_context}\n"
    
    conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n"

    # OPTIMIZED prompt - pre-compute language mapping for speed
    lang_map = {"Hindi": "Hindi", "Tamil": "Tamil", "English": "English"}
    lang = lang_map.get(detected_language, "English")
    
    # Build prompt efficiently (removed timing overhead) - minimal prompt
    if relevant_context:
        enhanced_prompt = f"{conversation}{context_prompt}Answer in {lang}. Maximum 40 words. Be brief and concise. Stop after 1-2 sentences."
    else:
        enhanced_prompt = f"{conversation}Answer in {lang}. Maximum 40 words. Be brief and concise. Stop after 1-2 sentences."
    
    print(f"📏 Prompt length: {len(enhanced_prompt)} characters")

    payload = {
        "input": enhanced_prompt,
        "parameters": {
            "decoding_method": "greedy",  # Greedy is fastest
            "max_new_tokens": 60,  # Reduced to limit verbose responses (40 words ≈ 50-60 tokens)
            "min_new_tokens": 5,  # Minimum for at least 1 sentence
            "stop_sequences": ["<|eot_id|>", "<|end_header_id|>", "\n\n", ".\n\n"],  # Stop on double newlines
            "repetition_penalty": 1.5,  # Higher penalty to stop generation faster and prevent rambling
            "temperature": 0.1  # Very low temp for fastest, most deterministic responses
        },
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "project_id": os.getenv("WATSONX_PROJECT_ID")
    }

    try:
        # Increased timeout for LLM generation - allow up to 60s for reliable responses
        # LLM generation can take longer, especially for complex queries
        # Use asyncio to make the request non-blocking
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        llm_start = time.time()
        print(f"⏱️  [3] Sending request to Watsonx LLM...")
        
        # Rule 3: Check for cancellation before making request
        if cancellation_token and cancellation_token.is_set():
            print("⚠️ Request cancelled before LLM call")
            return None
        if ongoing_request_cancelled:
            print("⚠️ Request cancelled before LLM call")
            ongoing_request_cancelled = False
            return None
        
        # Retry logic for LLM requests (max 1 retry for speed - 2s target)
        max_retries = 1
        retry_delay = 0.5
        response = None
        
        for attempt in range(max_retries + 1):
            try:
                # Make request with increased timeout (60 seconds)
                response = await loop.run_in_executor(
                    executor,
                    lambda: session.post(url, headers=headers, json=payload, timeout=60)
                )
                
                # If successful, break out of retry loop
                if response.status_code == 200:
                    break
                elif response.status_code in [429, 500, 502, 503, 504] and attempt < max_retries:
                    print(f"⚠️  Watsonx API returned {response.status_code}, retrying ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    # Non-retryable error or last attempt
                    break
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"⚠️  Watsonx API request timed out, retrying ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise  # Re-raise on last attempt
        
        # Safety check: ensure we have a valid response
        if response is None:
            raise requests.exceptions.RequestException("Watsonx API request failed after all retries")
        
        # Rule 3: Check for cancellation after request
        if cancellation_token and cancellation_token.is_set():
            print("⚠️ Request cancelled after LLM call")
            return None
        if ongoing_request_cancelled:
            print("⚠️ Request cancelled after LLM call")
            ongoing_request_cancelled = False
            return None
        
        llm_time = time.time() - llm_start
        print(f"⏱️  [3] Watsonx LLM Response: {llm_time:.3f}s")
        
        # Log if target is achieved (2s target)
        if llm_time <= 2.0:
            print(f"✅ LLM Generation Target ACHIEVED: {llm_time:.3f}s <= 2.0s")
        elif llm_time <= 2.5:
            print(f"⚡ LLM Generation Good: {llm_time:.3f}s (target: 2.0s)")
        else:
            print(f"⚠️  LLM Generation Target MISSED: {llm_time:.3f}s > 2.0s")
        
        if response.status_code == 200:
            response_data = response.json()
            if "results" in response_data and response_data["results"]:
                raw_response = response_data["results"][0]["generated_text"]
                print(f"📝 Raw AI Response Length: {len(raw_response)} characters")
                print(f"📝 Raw AI Response Preview: {raw_response[:200]}...")
                
                # Clean response and remove user input if it appears (removed timing overhead)
                cleaned = clean_ai_response(raw_response, user_input)
                
                total_time = time.time() - total_start
                print(f"\n{'='*80}")
                print(f"✅ TOTAL LLM PROCESSING TIME: {total_time:.3f}s")
                print(f"🎯 Target: 2.0s | Actual: {llm_time:.3f}s")
                if total_time <= 2.5:
                    print(f"✅ TOTAL PROCESSING TIME: {total_time:.3f}s <= 2.5s TARGET ✅")
                else:
                    print(f"⚠️  TOTAL PROCESSING TIME: {total_time:.3f}s > 2.5s TARGET")
                print(f"{'='*80}\n")
                
                # Cache the response
                if len(response_cache) < MAX_CACHE_SIZE:
                    response_cache[cache_key] = cleaned
                else:
                    # Clear old cache entries (simple FIFO)
                    oldest_key = next(iter(response_cache))
                    del response_cache[oldest_key]
                    response_cache[cache_key] = cleaned
                print(f"💾 Response cached for future use")
                
                return cleaned
            else:
                return "Error: 'generated_text' not found in the response."
        else:
            # Log the full error response for debugging
            error_detail = response.text
            print(f"❌ Watsonx API Error {response.status_code}: {error_detail}")
            return f"Error: Failed to fetch response from Watsonx.ai. Status code: {response.status_code}"
    except requests.exceptions.ConnectionError as e:
        error_msg = str(e)
        print(f"❌ Watsonx API connection failed: {error_msg}")
        print(f"❌ Full error details: {type(e).__name__}: {error_msg}")
        # Check if it's a DNS/network issue
        if "Name or service not known" in error_msg or "Failed to resolve" in error_msg:
            return "Error: Unable to resolve Watsonx API hostname. Please check your internet connection and DNS settings."
        elif "Connection refused" in error_msg:
            return "Error: Connection to Watsonx API was refused. The service may be down or blocked by a firewall."
        elif "certificate" in error_msg.lower() or "SSL" in error_msg:
            return "Error: SSL/TLS certificate error when connecting to Watsonx API. Please check your system's certificate settings."
        else:
            return f"Error: Unable to connect to Watsonx API. Connection error: {error_msg}"
    except requests.exceptions.Timeout:
        print("❌ Watsonx API request timed out after retries")
        return "Error: The AI service is taking too long to respond. Please try again or simplify your query."
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"❌ Watsonx API request failed: {error_type}: {error_msg}")
        print(f"❌ Full error details: {repr(e)}")
        import traceback
        traceback.print_exc()
        # Check if it's an authentication error
        if "401" in error_msg or "403" in error_msg or "Unauthorized" in error_msg or "Forbidden" in error_msg:
            return "Error: Watsonx API authentication failed. Please check your WATSONX_API_KEY and WATSONX_PROJECT_ID in the .env file."
        elif "404" in error_msg:
            return "Error: Watsonx API endpoint not found. Please verify your API configuration."
        else:
            return f"Error: Watsonx API request failed ({error_type}): {error_msg}"
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"❌ Exception in get_watsonx_response_async: {error_type}: {error_msg}")
        import traceback
        traceback.print_exc()
        return f"Error: Unexpected error ({error_type}): {error_msg}"

# API Routes
@app.get("/")
async def root():
    return {"message": "Voice Bot API is running!"}

@app.get("/status")
async def get_status():
    global bearer_token
    return {
        "authenticated": bearer_token is not None,
        "message_count": len(conversation_history),
        "bot_name": bot_name
    }

@app.post("/chat")
async def chat(request: ConversationRequest):
    import time
    global conversation_history, last_processed_query, ongoing_request_cancelled, recent_requests, ongoing_request_task, request_lock
    
    # Initialize lock on first use
    if request_lock is None:
        request_lock = asyncio.Lock()
    
    endpoint_start = time.time()
    user_input = request.user_input.strip()
    
    print(f"\n{'🎤'*40}")
    print(f"🎤 NEW CHAT REQUEST RECEIVED")
    print(f"📝 User Input: {user_input}")
    print(f"{'🎤'*40}\n")
    
    # Rule 1 & 4 & 6: Check for duplicate queries
    if is_duplicate_query(user_input, time_window=1.0):
        print("⚠️ Ignoring duplicate query")
        return {
            "success": False,
            "response": None,
            "conversation_history": conversation_history,
            "detected_language": "en-IN",
            "ignored": True,
            "reason": "duplicate_query"
        }
    
    # Rule 3: Handle interruptions - cancel ongoing request if exists
    async with request_lock:
        if ongoing_request_task and not ongoing_request_task.done():
            print("⚠️ Cancelling ongoing request due to new user input (interruption)")
            ongoing_request_cancelled = True
            # Cancel the task
            ongoing_request_task.cancel()
            # Wait a bit for cancellation to propagate
            try:
                await asyncio.wait_for(ongoing_request_task, timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Record this request
        current_time = time.time()
        recent_requests.append((current_time, user_input))
        
        # Create cancellation token for this request
        cancellation_token = asyncio.Event()
        
        # Add user message to history
        conversation_history.append(Message(role="user", content=user_input))
        
        # Rule 3: Start new request with cancellation support
        ongoing_request_cancelled = False
        ongoing_request_task = asyncio.create_task(
            get_watsonx_response_async(conversation_history[:-1], user_input, cancellation_token)
        )
    
    # Wait for response (with cancellation check)
    try:
        ai_response = await ongoing_request_task
    except asyncio.CancelledError:
        print("⚠️ Request was cancelled")
        ai_response = None
    except Exception as e:
        print(f"❌ Error in request: {e}")
        ai_response = None
    
    # Rule 3: Check if response was cancelled/interrupted
    if ai_response is None or ongoing_request_cancelled:
        print("⚠️ Response generation was interrupted/cancelled")
        # Remove the user message from history since we didn't respond
        if conversation_history and conversation_history[-1].role == "user":
            conversation_history.pop()
        return {
            "success": False,
            "response": None,
            "conversation_history": conversation_history,
            "detected_language": "en-IN",
            "ignored": True,
            "reason": "interrupted"
        }
    
    # Detect language from user input for TTS
    detected_language = detect_language(user_input)
    
    if ai_response and not ai_response.startswith("Error"):
        # Rule 5: Ensure response is clean and single (no duplicates)
        # Clean response is already handled in clean_ai_response function
        
        # Add AI response to history
        conversation_history.append(Message(role="assistant", content=ai_response))
        
        # Update last processed query (Rule 1)
        last_processed_query = user_input
        
        # Convert detected language to language code for TTS
        language_code_map = {
            "Hindi": "hi-IN",
            "Tamil": "ta-IN",
            "English": "en-IN"
        }
        detected_language_code = language_code_map.get(detected_language, "en-IN")
        
        endpoint_time = time.time() - endpoint_start
        print(f"\n{'✅'*40}")
        print(f"✅ CHAT ENDPOINT COMPLETED")
        print(f"⏱️  TOTAL ENDPOINT TIME: {endpoint_time:.3f}s")
        print(f"🎯 Target: 2.5s (LLM: 2.0s + TTS API: 0.5s)")
        if endpoint_time <= 2.5:
            print(f"✅ TARGET ACHIEVED: {endpoint_time:.3f}s <= 2.5s")
        else:
            print(f"⚠️  TARGET MISSED: {endpoint_time:.3f}s > 2.5s")
        print(f"📤 Response sent to frontend (TTS speaking time not included)")
        print(f"{'✅'*40}\n")
        
        return {
            "success": True,
            "response": ai_response,
            "conversation_history": conversation_history,
            "detected_language": detected_language_code
        }
    else:
        # Remove the user message from history since we didn't respond
        if conversation_history and conversation_history[-1].role == "user":
            conversation_history.pop()
        raise HTTPException(status_code=500, detail=f"AI Error: {ai_response}")

@app.get("/conversation")
async def get_conversation():
    return {"conversation_history": conversation_history}

@app.post("/clear-conversation")
async def clear_conversation():
    try:
        global conversation_history, response_cache, embedding_cache, last_processed_query, recent_requests, ongoing_request_cancelled, ongoing_request_task
        conversation_history = []
        response_cache.clear()  # Clear response cache
        embedding_cache.clear()  # Clear embedding cache
        last_processed_query = None  # Clear last processed query
        recent_requests.clear()  # Clear recent requests
        ongoing_request_cancelled = False  # Reset cancellation flag
        ongoing_request_task = None  # Clear ongoing task
        print("✅ Conversation history and caches cleared successfully")
        return {"success": True, "message": "Conversation cleared!"}
    except Exception as e:
        print(f"❌ Error clearing conversation: {e}")
        # Even if there's an error, try to clear it
        conversation_history = []
        response_cache.clear()
        embedding_cache.clear()
        last_processed_query = None
        recent_requests.clear()
        ongoing_request_cancelled = False
        ongoing_request_task = None
        return {"success": True, "message": "Conversation cleared (with warning)"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "watsonx_authenticated": bearer_token is not None,
        "sarvam_tts_available": sarvam_api_key is not None,
        "sarvam_stt_available": sarvam_stt_api_key is not None,
        "rag_enabled": pdf_collection is not None
    }

@app.post("/stt")
async def speech_to_text(request: STTRequest):
    """
    DEPRECATED: This endpoint is no longer used.
    The frontend now uses browser speech recognition for all STT operations.
    Sarvam AI is only used for TTS (text-to-speech).
    """
    raise HTTPException(status_code=410, detail="STT endpoint is deprecated. Use browser speech recognition instead.")

async def make_tts_request_with_retry(url: str, headers: dict, payload: dict, max_retries: int = 2) -> requests.Response:
    """Make a TTS request with retry logic and exponential backoff"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    retry_delay = 1.0
    response = None
    
    for attempt in range(max_retries + 1):
        try:
            # Make request with timeout (30 seconds)
            response = await loop.run_in_executor(
                executor,
                lambda: session.post(url, headers=headers, json=payload, timeout=30)
            )
            
            # If successful, return response
            if response.status_code == 200:
                return response
            elif response.status_code in [429, 500, 502, 503, 504] and attempt < max_retries:
                print(f"⚠️  TTS API returned {response.status_code}, retrying ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                # Non-retryable error or last attempt
                return response
                
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"⚠️  TTS API request timed out, retrying ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                raise  # Re-raise on last attempt
    
    # Return the last response (should not reach here in normal flow, but safety check)
    if response is None:
        raise requests.exceptions.RequestException("TTS request failed after all retries")
    return response

def split_text_into_chunks(text: str, max_chunk_size: int = 2000) -> List[str]:
    """Split text into chunks for TTS - ensures all text is processed"""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    # Split by sentences first to maintain natural flow
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed limit, save current chunk and start new one
        if current_chunk and len(current_chunk) + len(sentence) + 2 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
        else:
            if current_chunk:
                current_chunk += " " + sentence + "."
            else:
                current_chunk = sentence + "."
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If still too long, split by words
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split by words
            words = chunk.split()
            current_word_chunk = ""
            for word in words:
                if len(current_word_chunk) + len(word) + 1 > max_chunk_size:
                    if current_word_chunk:
                        final_chunks.append(current_word_chunk.strip())
                    current_word_chunk = word
                else:
                    if current_word_chunk:
                        current_word_chunk += " " + word
                    else:
                        current_word_chunk = word
            if current_word_chunk:
                final_chunks.append(current_word_chunk.strip())
    
    return final_chunks if final_chunks else [text]

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using Sarvam AI with selected voice - UNLIMITED TEXT LENGTH"""
    import time
    tts_start = time.time()
    
    print(f"\n{'🔊'*40}")
    print(f"🔊 TTS REQUEST RECEIVED")
    print(f"📝 Text length: {len(request.text)} characters")
    print(f"🌐 Language: {request.target_language_code}")
    print(f"🎤 Voice: {request.speaker}")
    print(f"🎵 Quality: {request.speech_sample_rate}Hz")
    print(f"{'🔊'*40}\n")
    
    if not sarvam_api_key:
        raise HTTPException(status_code=500, detail="SARVAM_API_KEY not configured")
    
    try:
        # Valid Sarvam voices
        valid_voices = ['anushka', 'vidya', 'manisha', 'abhilash', 'karun', 'arya', 'hitesh']
        
        # Use selected voice from frontend, default to 'vidya' if not provided
        speaker = request.speaker if request.speaker else 'vidya'
        
        # Validate speaker is one of the valid voices
        if speaker.lower() not in valid_voices:
            print(f"⚠️ Invalid speaker '{speaker}', defaulting to 'vidya'")
            speaker = 'vidya'
        
        # Convert language code to Sarvam-compatible format
        lang_code = request.target_language_code
        if lang_code == 'en-US' or lang_code == 'en':
            lang_code = 'en-IN'
        elif lang_code not in ['bn-IN', 'en-IN', 'gu-IN', 'hi-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'od-IN', 'pa-IN', 'ta-IN', 'te-IN']:
            lang_code = 'en-IN'
        
        # OPTIMIZED: Use 8kHz for SPEED (2-sec target) - lower quality but much faster
        audio_quality = request.speech_sample_rate if request.speech_sample_rate else 8000
        
        # Validate audio quality - common values are 8000, 16000, 22050, 24000
        valid_sample_rates = [8000, 16000, 22050, 24000]
        if audio_quality not in valid_sample_rates:
            print(f"⚠️ Invalid sample rate '{audio_quality}', defaulting to 8000 for speed")
            audio_quality = 8000
        
        api_call_start = time.time()
        print(f"⏱️  [1] Preparing TTS request to Sarvam AI...")
        print(f"   Language: {lang_code}")
        print(f"   Speaker: {speaker}")
        print(f"   Audio Quality: {audio_quality}Hz")
        print(f"   Text length: {len(request.text)} characters")
        print(f"   Word count: {len(request.text.split())} words")
        
        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": sarvam_api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": request.text,  # Send FULL text - NO LIMITS
            "target_language_code": lang_code,
            "speaker": speaker,
            "pitch": 0,
            "pace": 1.0,  # Normal speaking speed
            "loudness": 1.5,
            "speech_sample_rate": audio_quality,
            "enable_preprocessing": False,  # DISABLED for speed (no preprocessing delay)
            "model": "bulbul:v2"
        }
        
        # Increased timeout for TTS API - allow up to 30s for reliable responses
        # TTS generation can take longer, especially for longer texts
        print(f"⏱️  [2] Sending request to Sarvam AI...")
        api_start = time.time()
        
        # Make request with retry logic
        response = await make_tts_request_with_retry(url, headers, payload, max_retries=2)
        
        api_time = time.time() - api_start
        print(f"⏱️  [2] Sarvam AI Response: {api_time:.3f}s")
        
        # Log performance (target: < 5s for good UX, but allow up to 30s)
        if api_time <= 2.0:
            print(f"✅ TTS API Excellent: {api_time:.3f}s <= 2.0s")
        elif api_time <= 5.0:
            print(f"⚡ TTS API Good: {api_time:.3f}s <= 5.0s")
        elif api_time <= 10.0:
            print(f"⚠️  TTS API Slow: {api_time:.3f}s (acceptable but slow)")
        else:
            print(f"⚠️  TTS API Very Slow: {api_time:.3f}s (consider optimizing)")
        
        all_audio_bytes = []
        
        if response.status_code == 200:
            # Full text worked - use it
            result = response.json()
            if "audios" in result and len(result["audios"]) > 0:
                audio_base64 = result["audios"][0]
                audio_bytes = base64.b64decode(audio_base64)
                all_audio_bytes.append(audio_bytes)
                print(f"✅ FULL TEXT converted successfully: {len(audio_bytes)} bytes")
            else:
                print(f"⚠️ Full text response has no audio - trying chunking")
                # Fall back to chunking
                text_chunks = split_text_into_chunks(request.text, max_chunk_size=2000)
                print(f"   Split into {len(text_chunks)} chunks for processing")
                
                for i, chunk in enumerate(text_chunks):
                    print(f"   Processing chunk {i+1}/{len(text_chunks)}: {len(chunk)} characters")
                    
                    chunk_payload = {
                        "text": chunk,
                        "target_language_code": lang_code,
                        "speaker": speaker,
                        "pitch": 0,
                        "pace": 1.0,
                        "loudness": 1.5,
                        "speech_sample_rate": audio_quality,
                        "enable_preprocessing": True,
                        "model": "bulbul:v2"
                    }
                    
                    chunk_response = await make_tts_request_with_retry(url, headers, chunk_payload)
                    
                    if chunk_response.status_code == 200:
                        chunk_result = chunk_response.json()
                        if "audios" in chunk_result and len(chunk_result["audios"]) > 0:
                            audio_base64 = chunk_result["audios"][0]
                            audio_bytes = base64.b64decode(audio_base64)
                            all_audio_bytes.append(audio_bytes)
                            print(f"   ✅ Chunk {i+1} converted: {len(audio_bytes)} bytes")
        else:
            # API returned error - try chunking
            print(f"⚠️ Full text failed ({response.status_code}), trying chunking...")
            text_chunks = split_text_into_chunks(request.text, max_chunk_size=2000)
            print(f"   Split into {len(text_chunks)} chunks for processing")
            
            for i, chunk in enumerate(text_chunks):
                print(f"   Processing chunk {i+1}/{len(text_chunks)}: {len(chunk)} characters")
                
                chunk_payload = {
                    "text": chunk,
                    "target_language_code": lang_code,
                    "speaker": speaker,
                    "pitch": 0,
                    "pace": 1.0,
                    "loudness": 1.5,
                    "speech_sample_rate": audio_quality,
                    "enable_preprocessing": True,
                    "model": "bulbul:v2"
                }
                
                chunk_response = session.post(url, headers=headers, json=chunk_payload, timeout=5)
                
                if chunk_response.status_code == 200:
                    chunk_result = chunk_response.json()
                    if "audios" in chunk_result and len(chunk_result["audios"]) > 0:
                        audio_base64 = chunk_result["audios"][0]
                        audio_bytes = base64.b64decode(audio_base64)
                        all_audio_bytes.append(audio_bytes)
                        print(f"   ✅ Chunk {i+1} converted: {len(audio_bytes)} bytes")
                else:
                    print(f"   ❌ Chunk {i+1} failed: {chunk_response.status_code}")
                    # Continue with other chunks even if one fails
                    continue
        
        if all_audio_bytes:
            # Properly combine WAV audio chunks
            import wave
            import struct
            
            if len(all_audio_bytes) == 1:
                # Single chunk - return as is
                combined_audio = all_audio_bytes[0]
            else:
                # Multiple chunks - concatenate WAV files properly
                combined_audio_data = []
                sample_rate = None
                channels = None
                sample_width = None
                
                for audio_bytes in all_audio_bytes:
                    audio_io = BytesIO(audio_bytes)
                    with wave.open(audio_io, 'rb') as wav_file:
                        if sample_rate is None:
                            sample_rate = wav_file.getframerate()
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()
                        
                        # Read all frames from this chunk
                        frames = wav_file.readframes(wav_file.getnframes())
                        combined_audio_data.append(frames)
                
                # Create combined WAV file
                output = BytesIO()
                with wave.open(output, 'wb') as combined_wav:
                    combined_wav.setnchannels(channels)
                    combined_wav.setsampwidth(sample_width)
                    combined_wav.setframerate(sample_rate)
                    # Write all frames sequentially
                    for frames in combined_audio_data:
                        combined_wav.writeframes(frames)
                
                combined_audio = output.getvalue()
            
            tts_total_time = time.time() - tts_start
            
            print(f"\n{'='*80}")
            print(f"✅ TTS COMPLETED SUCCESSFULLY")
            print(f"   Total chunks processed: {len(all_audio_bytes)}")
            print(f"   Combined audio size: {len(combined_audio)} bytes")
            print(f"⏱️  TOTAL TTS TIME: {tts_total_time:.3f}s")
            print(f"{'='*80}\n")
            
            return StreamingResponse(
                BytesIO(combined_audio),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "inline; filename=speech.wav"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="No audio data generated from any chunks")
            
    except requests.exceptions.Timeout:
        print("❌ Sarvam AI TTS request timed out after retries")
        raise HTTPException(
            status_code=504, 
            detail="TTS request timed out. The service may be experiencing high load. Please try again."
        )
    except requests.exceptions.RequestException as e:
        print(f"❌ TTS API request failed: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"TTS service unavailable: {str(e)}. Please try again."
        )
    except Exception as e:
        print(f"❌ TTS conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import sys
    import platform
    # Disable reload on Windows to avoid multiprocessing/socket errors
    # On Windows, use: python main.py (no reload)
    # On Linux/Mac, you can use: uvicorn main:app --reload
    use_reload = "--reload" in sys.argv and platform.system() != "Windows"
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=use_reload
    )
