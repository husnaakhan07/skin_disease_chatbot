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
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

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
MIN_TTS_INTERVAL = 2
MAX_TTS_CHARS = 800
MAX_TTS_RETRIES = 3

for folder in ["static/audio", "static/uploads"]:
    os.makedirs(folder, exist_ok=True)

# Enable/Disable Text-to-Speech
tts_status = True

# Text-to-Speech Modifier (Should be in brackets [])
tts_modifier = "[quick]"

# --- 2. LOAD YOUR TRAINED SKIN DISEASE MODEL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skin_model = None
skin_class_names = []

# Session memory for last 3 prompts and detected diseases
conversation_history = {}  # Stores last 3 exchanges per session
last_detected_disease = {}  # Stores last detected disease per session

def get_conversation_history(session_id):
    """Get the last 3 conversation exchanges"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    return conversation_history[session_id]

def add_to_conversation_history(session_id, user_msg, bot_msg):
    """Add an exchange to history, keep only last 3"""
    history = get_conversation_history(session_id)
    history.append({"user": user_msg, "bot": bot_msg})
    if len(history) > 3:
        history.pop(0)

def get_session_disease(session_id="static_user"):
    """Get the last detected disease for a session"""
    return last_detected_disease.get(session_id)

def set_session_disease(session_id, disease):
    """Store the last detected disease for a session"""
    last_detected_disease[session_id] = disease

# Disease name mapping for better display
DISEASE_NAME_MAPPING = {
    'acne': 'Acne',
    'actinickeratosis': 'Actinic Keratosis',
    'baimpetigo': 'Impetigo',
    'basalcellcarcinoma': 'Basal Cell Carcinoma',
    'benigntumors': 'Benign Tumors',
    'bullous': 'Bullous Disease',
    'chickenpox': 'Chickenpox',
    'cowpox': 'Cowpox',
    'drugeruption': 'Drug Eruption',
    'eczema': 'Eczema',
    'healthy': 'Healthy Skin',
    'hfmd': 'Hand-Foot-Mouth Disease',
    'infestationsbites': 'Infestation/Bites',
    'lichen': 'Lichen Planus',
    'lupus': 'Lupus',
    'measles': 'Measles',
    'melanocyticnevi': 'Melanocytic Nevi',
    'moles': 'Moles',
    'monkeypox': 'Monkeypox',
    'psoriasis': 'Psoriasis',
    'rosacea': 'Rosacea',
    'seborrheakeratoses': 'Seborrheic Keratosis',
    'skincancer': 'Skin Cancer',
    'sunlightdamage': 'Sunlight Damage',
    'tinea': 'Tinea',
    'unknown': 'Unknown Condition',
    'vascularlesion': 'Vascular Lesion',
    'vasculartumors': 'Vascular Tumors',
    'vasculitis': 'Vasculitis',
    'vasculitis_': 'Vasculitis',
    'vitiligo': 'Vitiligo',
    'warts': 'Warts'
}

def get_pretty_disease_name(model_name):
    """Convert model class name to proper medical term"""
    return DISEASE_NAME_MAPPING.get(model_name, model_name.replace('_', ' ').title())

try:
    # Load your trained model
    checkpoint = torch.load('skin_disease_model.pth', map_location=DEVICE)
    skin_class_names = checkpoint['class_names']
    num_classes = len(skin_class_names)
    
    print(f"✅ Skin disease model loaded successfully")
    print(f"📊 Number of classes: {num_classes}")
    print(f"🏷️ Classes: {skin_class_names[:10]}...")
    
    # Use ResNet18
    skin_model = models.resnet18(weights=None)
    num_features = skin_model.fc.in_features
    skin_model.fc = nn.Linear(num_features, num_classes)
    skin_model.load_state_dict(checkpoint['model_state_dict'])
    skin_model = skin_model.to(DEVICE)
    skin_model.eval()
    
    print(f"✅ Model ready on {DEVICE}")
    
except Exception as e:
    print(f"⚠️ Could not load skin disease model: {e}")
    print("   Will use vision model as fallback")

# Image transform for skin model
skin_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. RAG SETUP (for medical book) ---
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(index_name="medical-chatbot", embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Chat Model (General RAG)
chatModel = ChatGroq(model="openai/gpt-oss-120b", groq_api_key=GROQ_API_KEY, temperature=0.4)

# Vision Model (Fallback for image analysis if your model fails)
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

# History Management for RAG
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# The Stateful Chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- 4. FUNCTION TO GET MEDICAL ADVICE (FIRST FROM RAG, THEN GENERAL) ---
def get_medical_advice(disease, question, session_id="static_user"):
    """Get medical advice - first try RAG from medical book, then fall back to general knowledge"""
    try:
        # Create a query that asks about the specific disease
        medical_query = f"Based on the medical book, provide information about {disease}. {question}"
        
        # Try RAG first
        response = conversational_rag_chain.invoke(
            {"input": medical_query},
            config={"configurable": {"session_id": session_id}}
        )
        
        answer = response["answer"]
        
        # Check if RAG returned a "not found" type response
        not_found_phrases = [
            "don't have information",
            "does not contain",
            "not in the provided",
            "cannot find",
            "no information about",
            "not mentioned"
        ]
        
        if any(phrase in answer.lower() for phrase in not_found_phrases):
            # Fall back to general knowledge
            general_query = f"Provide general medical information about {disease}. {question}"
            general_response = conversational_rag_chain.invoke(
                {"input": general_query},
                config={"configurable": {"session_id": session_id}}
            )
            return general_response["answer"]
        
        return answer
        
    except Exception as e:
        print(f"Error getting medical advice: {e}")
        # Final fallback
        return f"I can tell you that {get_pretty_disease_name(disease)} is a skin condition. For specific medical advice, please consult a healthcare professional."

# --- 5. HELPER FUNCTIONS ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def clean_response_for_display(text):
    """Clean markdown tables for better display"""
    if not text:
        return text
    
    lines = text.split('\n')
    cleaned_lines = []
    in_table = False
    
    for line in lines:
        if re.match(r'^\|\s*[-:]+\s*\|\s*[-:]+\s*\|', line):
            cleaned_lines.append(line)
            in_table = True
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
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    sentence_endings = ['. ', '? ', '! ', '.\n', '?\n', '!\n']
    best_break = -1
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > max_chars * 0.5:
            best_break = max(best_break, pos + len(ending) - 1)
    if best_break > 0:
        return text[:best_break + 1] + " (Summary continues in text)"
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.7:
        return text[:last_space] + "... (Full response in text)"
    return truncated + "... (Full response in text)"

def generate_speech_safe(text, max_retries=MAX_TTS_RETRIES):
    global last_tts_time
    if not text or len(text.strip()) < 10:
        return None
    tts_text = truncate_for_tts(text)
    current_time = time.time()
    time_since_last = current_time - last_tts_time
    if time_since_last < MIN_TTS_INTERVAL:
        time.sleep(MIN_TTS_INTERVAL - time_since_last)
    
    for attempt in range(max_retries):
        try:
            timestamp = int(time.time() * 1000)
            random_suffix = random.randint(1000, 9999)
            audio_filename = f"response_{timestamp}_{random_suffix}.wav"
            audio_path = os.path.join("static/audio", audio_filename)
            
            print(f"Generating TTS (attempt {attempt + 1}, length: {len(tts_text)} chars)")
            
            audio_response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="troy",
                input=tts_modifier + tts_text,
                response_format="wav"
            )
            audio_response.write_to_file(audio_path)
            
            last_tts_time = time.time()
            
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                print(f"TTS successful: {audio_filename}")
                return audio_filename
                
        except Exception as e:
            error_str = str(e)
            print(f"TTS attempt {attempt + 1} failed: {error_str[:100]}")
            if "rate_limit_exceeded" in error_str or "413" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.random()
                    time.sleep(wait_time)
                    if attempt == max_retries - 2:
                        tts_text = truncate_for_tts(text, max_chars=400)
            else:
                break
    return None

def cleanup_old_audio_files(max_files=30):
    try:
        audio_dir = "static/audio"
        files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
        files.sort(key=os.path.getctime, reverse=True)
        for file in files[max_files:]:
            try:
                os.remove(file)
            except:
                pass
    except Exception as e:
        print(f"Cleanup error: {e}")

# --- 6. ROUTES ---
@app.route("/")
def index():
    cleanup_old_audio_files()
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    image_file = request.files.get("image")
    final_answer = ""
    audio_url = None
    session_id = "static_user"

    try:
        # CASE 1: Image uploaded - Use your trained model
        if image_file and image_file.filename != '':
            # Save image
            file_path = os.path.join("static/uploads", f"img_{int(time.time())}_{image_file.filename}")
            image_file.save(file_path)
            
            try:
                if skin_model is not None:
                    # Load and preprocess image
                    image = Image.open(file_path).convert('RGB')
                    image_tensor = skin_transform(image).unsqueeze(0).to(DEVICE)
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = skin_model(image_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        
                        # Get top 3 predictions
                        top3_prob, top3_idx = torch.topk(probabilities, 3)
                        
                        # Get primary prediction
                        confidence, predicted = torch.max(probabilities, 0)
                        primary_disease_raw = skin_class_names[predicted.item()]
                        primary_disease = get_pretty_disease_name(primary_disease_raw)
                        primary_confidence = confidence.item() * 100
                        
                        # Store the detected disease for follow-up questions
                        set_session_disease(session_id, primary_disease_raw)
                        
                        # Format response
                        formatted_response = f"🔍 **Skin Analysis Result:**\n\n"
                        formatted_response += f"**Primary Detection:** {primary_disease}\n"
                        formatted_response += f"**Confidence:** {primary_confidence:.1f}%\n\n"
                        formatted_response += f"**Top 3 possibilities:**\n"
                        
                        for i, (prob, idx) in enumerate(zip(top3_prob, top3_idx), 1):
                            disease_raw = skin_class_names[idx.item()]
                            disease = get_pretty_disease_name(disease_raw)
                            conf = prob.item() * 100
                            formatted_response += f"{i}. {disease} ({conf:.1f}%)\n"
                        
                        final_answer = formatted_response
                        
                        # If user asked a question with the image, answer it
                        if msg:
                            advice = get_medical_advice(primary_disease_raw, msg, session_id)
                            final_answer += f"\n\n💬 **Regarding your question:**\n{advice}"
                else:
                    # Fallback to vision model
                    base64_image = encode_image(file_path)
                    response = visionModel.invoke([
                        SystemMessage(content="Analyze this medical image and identify the skin condition."),
                        HumanMessage(content=[
                            {"type": "text", "text": msg if msg else "What skin condition is this?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ])
                    ])
                    final_answer = response.content
            
            except Exception as e:
                print(f"Error analyzing image: {e}")
                final_answer = "I couldn't analyze the image. Please make sure it's a clear photo of a skin condition."
            
            finally:
                try:
                    os.remove(file_path)
                except:
                    pass
        
        # CASE 2: Text only - Check if it's a follow-up question
        else:
            if not msg:
                return jsonify({"answer": "Please provide a question or upload a skin image."})
            
            # Check if there's a recently detected disease
            previous_disease_raw = get_session_disease(session_id)
            
            # Check if this is a follow-up question (contains words like "it", "this", "the condition")
            follow_up_indicators = ['it', 'this', 'that', 'the condition', 'the above', 'for it']
            is_follow_up = any(indicator in msg.lower() for indicator in follow_up_indicators)
            
            if previous_disease_raw and is_follow_up:
                # This is a follow-up about the previously detected disease
                pretty_disease = get_pretty_disease_name(previous_disease_raw)
                
                # Get medical advice (first from RAG, then general)
                advice = get_medical_advice(previous_disease_raw, msg, session_id)
                
                # Format based on question type
                if 'symptom' in msg.lower():
                    final_answer = f"💬 **Symptoms of {pretty_disease}**\n\n{advice}"
                elif 'treatment' in msg.lower() or 'medication' in msg.lower() or 'cure' in msg.lower():
                    final_answer = f"💬 **Treatment for {pretty_disease}**\n\n{advice}"
                elif 'cause' in msg.lower() or 'why' in msg.lower():
                    final_answer = f"💬 **Causes of {pretty_disease}**\n\n{advice}"
                else:
                    final_answer = f"💬 **About {pretty_disease}**\n\n{advice}"
            else:
                # Regular text query
                response = conversational_rag_chain.invoke(
                    {"input": msg},
                    config={"configurable": {"session_id": session_id}}
                )
                final_answer = response["answer"]

        # Clean response and generate TTS
        final_answer = clean_response_for_display(final_answer)
        
        try:
            global tts_status
            if tts_status == False:
                audio_filename = generate_speech_safe(final_answer)
                if audio_filename:
                    audio_url = url_for('static', filename=f'audio/{audio_filename}')
        except Exception as tts_error:
            print(f"TTS error: {tts_error}")
        
        # Add to conversation history (last 3 prompts)
        add_to_conversation_history(session_id, msg, final_answer)
        
        return jsonify({
            "answer": final_answer, 
            "audio_url": audio_url
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "answer": "I encountered an error. Please try again.",
            "audio_url": None
        })

@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        session_id = request.json.get("session_id", "static_user")
        if session_id in store:
            del store[session_id]
        if session_id in conversation_history:
            del conversation_history[session_id]
        if session_id in last_detected_disease:
            del last_detected_disease[session_id]
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/status", methods=["GET"])
def status():
    audio_files = [f for f in os.listdir("static/audio") if f.endswith('.wav')]
    return jsonify({
        "audio_files_count": len(audio_files),
        "last_tts_time": last_tts_time,
        "skin_model_loaded": skin_model is not None,
        "skin_classes": len(skin_class_names) if skin_class_names else 0
    })

@app.route("/mute", methods=["POST"])
def mute():
    try:
        global tts_status
        tts_status = bool(request.json.get("isActive"))
        print(f"Received Mute Status: {tts_status}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    cleanup_old_audio_files()
    print("🚀 Medical AI Assistant starting...")
    print(f"🧠 Skin model loaded: {skin_model is not None}")
    if skin_class_names:
        print(f"🩺 Can detect: {len(skin_class_names)} conditions")
    app.run(host="0.0.0.0", port=8080, debug=True)