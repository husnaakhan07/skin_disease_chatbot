from groq import Groq
from flask import Flask, render_template, jsonify, request, url_for
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import os
import base64
import time
import random
import re

# Import your custom prompts
from src.prompt import system_prompt

app = Flask(__name__)
load_dotenv()

# --- 1. CONFIGURATION & DIRECTORIES ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Initialize the Groq SDK Client for Audio
groq_client = Groq(api_key=GROQ_API_KEY)

# Rate limiting for TTS
last_tts_time = 0
MIN_TTS_INTERVAL = 2  # Minimum seconds between TTS requests
MAX_TTS_CHARS = 800  # Conservative limit (1200 tokens ≈ 4800 chars, using 800 to be safe)
MAX_TTS_RETRIES = 3

for folder in ["static/audio", "static/uploads"]:
    os.makedirs(folder, exist_ok=True)

# Clean up old audio files
def cleanup_old_audio_files(max_files=30):
    """Keep only the most recent audio files"""
    try:
        audio_dir = "static/audio"
        files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
        files.sort(key=os.path.getctime, reverse=True)
        for file in files[max_files:]:
            try:
                os.remove(file)
                print(f"Cleaned up old audio file: {file}")
            except:
                pass
    except Exception as e:
        print(f"Cleanup error: {e}")

# --- 2. MODEL INITIALIZATION ---
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name="medical-chatbot", embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Chat Model (General RAG)
chatModel = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY, temperature=0.4)

# Vision Model (Image Analysis)
visionModel = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=GROQ_API_KEY)

# Prompt with History Placeholder
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Build the Chains
question_answer_chain = create_stuff_documents_chain(chatModel, rag_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# History Management
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# The Stateful Chain (This handles memory automatically)
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- 3. HELPER FUNCTIONS ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def clean_response_for_display(text):
    """Clean markdown tables for better display"""
    if not text:
        return text
    
    # Ensure tables are properly formatted
    lines = text.split('\n')
    cleaned_lines = []
    in_table = False
    
    for line in lines:
        # Fix table separator lines if needed
        if re.match(r'^\|\s*[-:]+\s*\|\s*[-:]+\s*\|', line):
            cleaned_lines.append(line)
            in_table = True
        # Ensure table rows have proper format
        elif '|' in line and line.strip().startswith('|'):
            cleaned_lines.append(line)
            in_table = True
        else:
            if in_table and line.strip() == '':
                cleaned_lines.append('')
            in_table = False
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def truncate_for_tts(text, max_chars=MAX_TTS_CHARS):
    """Intelligently truncate text for TTS"""
    if not text or len(text) <= max_chars:
        return text
    
    # Try to find a good breaking point
    truncated = text[:max_chars]
    
    # Look for sentence endings
    sentence_endings = ['. ', '? ', '! ', '.\n', '?\n', '!\n']
    best_break = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > max_chars * 0.5:  # Only use if it's past the halfway point
            best_break = max(best_break, pos + len(ending) - 1)
    
    if best_break > 0:
        return text[:best_break + 1] + " (Summary continues in text)"
    
    # If no good sentence break, try to break at a word
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        return text[:last_space] + "... (Full response in text)"
    
    # Last resort: just truncate
    return truncated + "... (Full response in text)"

def generate_speech_safe(text, max_retries=MAX_TTS_RETRIES):
    """Generate speech with comprehensive error handling"""
    global last_tts_time
    
    # Don't attempt TTS for empty or very short text
    if not text or len(text.strip()) < 10:
        return None
    
    # Truncate text to safe length
    tts_text = truncate_for_tts(text)
    
    # Rate limiting
    current_time = time.time()
    time_since_last = current_time - last_tts_time
    if time_since_last < MIN_TTS_INTERVAL:
        time.sleep(MIN_TTS_INTERVAL - time_since_last)
    
    for attempt in range(max_retries):
        try:
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            random_suffix = random.randint(1000, 9999)
            audio_filename = f"response_{timestamp}_{random_suffix}.wav"
            audio_path = os.path.join("static/audio", audio_filename)
            
            print(f"Generating TTS (attempt {attempt + 1}, length: {len(tts_text)} chars)")
            
            audio_response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="troy",
                input=tts_text,
                response_format="wav"
            )
            audio_response.write_to_file(audio_path)
            
            # Update last TTS time
            last_tts_time = time.time()
            
            # Verify file was created and has content
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                print(f"TTS successful: {audio_filename}")
                return audio_filename
            else:
                print(f"TTS file is empty or missing")
                
        except Exception as e:
            error_str = str(e)
            print(f"TTS attempt {attempt + 1} failed: {error_str[:100]}")
            
            if "rate_limit_exceeded" in error_str or "413" in error_str:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) + random.random()
                    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    
                    # Try with even shorter text on retry
                    if attempt == max_retries - 2:
                        tts_text = truncate_for_tts(text, max_chars=400)
                else:
                    print("Max retries reached for TTS")
            else:
                # Non-rate-limit error, don't retry
                break
    
    return None

# --- 4. ROUTES ---
@app.route("/")
def index():
    # Cleanup old files on page load
    cleanup_old_audio_files()
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    image_file = request.files.get("image")
    final_answer = ""
    audio_url = None

    try:
        # A. MULTIMODAL MODE
        if image_file and image_file.filename != '':
            # Save and process image
            file_path = os.path.join("static/uploads", f"img_{int(time.time())}_{image_file.filename}")
            image_file.save(file_path)
            base64_image = encode_image(file_path)
            
            # Get response from vision model
            response = visionModel.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": msg if msg else "Analyze this medical image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ])
            final_answer = response.content
            
            # Clean up uploaded image
            try:
                os.remove(file_path)
            except:
                pass
        
        # B. TEXT MODE
        else:
            if not msg:
                return jsonify({"answer": "Please provide a question."})
            
            # Get response from RAG chain
            response = conversational_rag_chain.invoke(
                {"input": msg},
                config={"configurable": {"session_id": "static_user"}}
            )
            final_answer = response["answer"]

        # Clean the response for display
        final_answer = clean_response_for_display(final_answer)
        
        # C. TEXT-TO-SPEECH - with safe error handling
        try:
            audio_filename = generate_speech_safe(final_answer)
            if audio_filename:
                audio_url = url_for('static', filename=f'audio/{audio_filename}')
                print(f"Audio URL generated: {audio_url}")
        except Exception as tts_error:
            print(f"TTS error (non-critical): {tts_error}")
            # Continue without audio - this is not fatal
        
        return jsonify({
            "answer": final_answer, 
            "audio_url": audio_url
        })

    except Exception as e:
        print(f"Error in /get route: {e}")
        return jsonify({
            "answer": "I encountered an error processing your request. Please try again.",
            "audio_url": None
        })

# Optional: Add a route to clear chat history
@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        session_id = request.json.get("session_id", "static_user")
        if session_id in store:
            del store[session_id]
        return jsonify({"success": True, "message": "History cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Optional: Add a route to check status
@app.route("/status", methods=["GET"])
def status():
    audio_files = [f for f in os.listdir("static/audio") if f.endswith('.wav')]
    return jsonify({
        "audio_files_count": len(audio_files),
        "last_tts_time": last_tts_time,
        "time_since_last_tts": time.time() - last_tts_time if last_tts_time > 0 else None,
        "tts_configured": True
    })

if __name__ == '__main__':
    # Initial cleanup
    cleanup_old_audio_files()
    print("Medical AI Assistant starting...")
    print(f"TTS Max Chars: {MAX_TTS_CHARS}")
    print(f"TTS Min Interval: {MIN_TTS_INTERVAL}s")
    app.run(host="0.0.0.0", port=8080, debug=True)