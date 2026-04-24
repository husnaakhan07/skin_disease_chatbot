from groq import Groq
from flask import Flask, render_template, jsonify, request, url_for, send_from_directory
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
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from src.prompt import system_prompt

app = Flask(__name__)
load_dotenv()

# --- ADMIN DATA STORAGE ---
prediction_history = []

#  CONFIGURATION & DIRECTORIES ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

last_tts_time = 0
MIN_TTS_INTERVAL = 0.3
MAX_TTS_CHARS = 800
MAX_TTS_RETRIES = 3

for folder in ["static/audio", "static/uploads"]:
    os.makedirs(folder, exist_ok=True)

tts_status = True
tts_modifier = "[quick]"

# --- 2. LOAD YOUR TRAINED SKIN DISEASE MODEL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
skin_model = None
skin_class_names = []

last_mentioned_disease = {}
detected_disease_history = {}

def get_last_mentioned_disease(session_id="static_user"):
    return last_mentioned_disease.get(session_id)

def set_last_mentioned_disease(session_id, disease):
    last_mentioned_disease[session_id] = disease

def get_detected_disease(session_id="static_user"):
    if session_id in detected_disease_history and detected_disease_history[session_id]:
        return detected_disease_history[session_id][-1]
    return None

def set_detected_disease(session_id, disease):
    if session_id not in detected_disease_history:
        detected_disease_history[session_id] = []
    detected_disease_history[session_id].append(disease)
    if len(detected_disease_history[session_id]) > 5:
        detected_disease_history[session_id].pop(0)

def clear_session_data(session_id):
    if session_id in detected_disease_history:
        detected_disease_history[session_id] = []
    if session_id in last_mentioned_disease:
        del last_mentioned_disease[session_id]

DISEASE_NAME_MAPPING = {
    "acne": "Acne",
    "actinickeratosis": "Actinic Keratosis",
    "arthritis": "Arthritis",
    "baimpetigo": "Impetigo",
    "basalcellcarcinoma": "Basal Cell Carcinoma",
    "benigntumors": "Benign Tumors",
    "bullous": "Bullous Disease",
    "chickenpox": "Chickenpox",
    "cowpox": "Cowpox",
    "drugeruption": "Drug Eruption",
    "eczema": "Eczema",
    "healthy": "Healthy Skin",
    "hfmd": "Hand-Foot-Mouth Disease",
    "infestationsbites": "Infestation/Bites",
    "lichen": "Lichen Planus",
    "lupus": "Lupus",
    "measles": "Measles",
    "melanocyticnevi": "Melanocytic Nevi",
    "moles": "Moles",
    "monkeypox": "Monkeypox",
    "psoriasis": "Psoriasis",
    "rosacea": "Rosacea",
    "seborrheakeratoses": "Seborrheic Keratosis",
    "skincancer": "Skin Cancer",
    "sunlightdamage": "Sunlight Damage",
    "tinea": "Tinea",
    "unknown": "Unknown Condition",
    "vascularlesion": "Vascular Lesion",
    "vasculartumors": "Vascular Tumors",
    "vasculitis": "Vasculitis",
    "vasculitis_": "Vasculitis",
    "vitiligo": "Vitiligo",
    "warts": "Warts",
}

def get_pretty_disease_name(model_name):
    return DISEASE_NAME_MAPPING.get(model_name, model_name.replace("_", " ").title())

DISEASE_KEYWORDS = {
    "acne": "acne",
    "pimple": "acne",
    "arthritis": "arthritis",
    "eczema": "eczema",
    "dermatitis": "eczema",
    "psoriasis": "psoriasis",
    "rosacea": "rosacea",
    "wart": "warts",
    "warts": "warts",
    "mole": "moles",
    "moles": "moles",
    "skincancer": "skincancer",
    "skin cancer": "skincancer",
    "melanoma": "skincancer",
    "basal cell": "basalcellcarcinoma",
    "seborrheic": "seborrheakeratoses",
    "keratosis": "seborrheakeratoses",
    "vitiligo": "vitiligo",
    "lupus": "lupus",
    "tinea": "tinea",
    "ringworm": "tinea",
    "chickenpox": "chickenpox",
    "chicken pox": "chickenpox",
    "monkeypox": "monkeypox",
    "monkey pox": "monkeypox",
    "herpes": "unknown",
    "impetigo": "baimpetigo",
}

FOLLOW_UP_INDICATORS = [
    "it",
    "this",
    "that",
    "the condition",
    "the disease",
    "its",
    "for it",
    "tell me more",
    "what about",
    "symptoms of it",
    "treatment for it",
    "causes of it",
]

try:
    checkpoint = torch.load("skin_disease_model.pth", map_location=DEVICE)
    skin_class_names = checkpoint["class_names"]
    num_classes = len(skin_class_names)

    print("✅ Skin disease model loaded successfully")
    print(f"📊 Number of classes: {num_classes}")
    print(f"📚 Classes: {skin_class_names[:10]}...")

    skin_model = models.resnet18(weights=None)
    num_features = skin_model.fc.in_features
    skin_model.fc = nn.Linear(num_features, num_classes)
    skin_model.load_state_dict(checkpoint["model_state_dict"])
    skin_model = skin_model.to(DEVICE)
    skin_model.eval()

    print(f"✅ Model ready on {DEVICE}")

except Exception as e:
    print(f"⚠️ Could not load skin disease model: {e}")
    print("   Vision model fallback will be used if needed.")

skin_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 3. RAG SETUP ---
embeddings = download_hugging_face_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.4,
)
fallbackModel = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.4,
)
visionModel = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=GROQ_API_KEY,
)

# General chat prompt with history
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Disease-specific prompt without history
medical_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, rag_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

medical_question_answer_chain = create_stuff_documents_chain(chatModel, medical_rag_prompt)
medical_rag_chain = create_retrieval_chain(retriever, medical_question_answer_chain)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- 4. FUNCTION TO GET MEDICAL ADVICE ---
def get_medical_advice(disease, question, session_id="static_user"):
    """Get disease-specific advice without contaminating from prior chat history."""
    try:
        pretty_disease = get_pretty_disease_name(disease)
        question_lower = question.lower().strip()

        phrases_to_remove = [
            "I’m sorry—I don’t have information about",
            "I'm sorry—I don't have information about",
            "I'm sorry, I don't have information about",
            "I’m sorry, I don’t have information about",
            "However, based on standard medical practice,",
            "in the material you provided.",
        ]

        if "symptom" in question_lower:
            medical_query = (
                f"Based on the medical book, give only the symptoms of {pretty_disease} ({disease}). "
                f"Answer clearly with bullet points. "
                f"Do not include apologies or mention missing material. {question}"
            )

        elif any(x in question_lower for x in ["medication", "medicine", "drug", "dosage", "dose"]):
            medical_query = (
                f"Based on the medical book, give only the medications used for {pretty_disease} ({disease}). "
                f"For each medication, include: medication name, purpose, dosage, frequency, duration, and important notes. "
                f"If exact dosage is not available in the book, say 'dosage not specified in provided material' instead of guessing. "
                f"Format the answer as a markdown table with columns: Medication, Purpose, Dosage, Frequency, Duration, Notes. "
                f"Do not include any apology or mention missing material. {question}"
            )

        elif any(x in question_lower for x in ["treatment", "therapy", "cure"]):
            medical_query = (
                f"Based on the medical book, give only the treatments for {pretty_disease} ({disease}). "
                f"Answer clearly with bullet points. "
                f"Do not include apologies or mention missing material. {question}"
            )

        elif any(x in question_lower for x in ["cause", "causes", "trigger", "triggers", "why"]):
            medical_query = (
                f"Based on the medical book, give only the causes or triggers of {pretty_disease} ({disease}). "
                f"Answer clearly with bullet points. "
                f"Do not include apologies or mention missing material. {question}"
            )

        else:
            medical_query = (
                f"Based on the medical book, provide information only about {pretty_disease} ({disease}). "
                f"Answer directly and clearly. "
                f"Do not include apologies or mention missing material. {question}"
            )

        print(f"[get_medical_advice] disease={disease}")
        print(f"[get_medical_advice] medical_query={medical_query}")

        response = medical_rag_chain.invoke({"input": medical_query})
        print(f"[get_medical_advice] raw RAG response={response}")

        answer = (response.get("answer") or "").strip()
        needs_medication_fallback = False

        if any(x in question_lower for x in ["medication", "medicine", "drug", "dosage", "dose"]):
            low_info_markers = [
                "dosage not specified in provided material",
                "frequency not specified in provided material",
                "duration not specified in provided material",
            ]

            marker_count = sum(answer.lower().count(marker) for marker in low_info_markers)

            # If RAG answer is mostly placeholder dosage info, force fallback
            if marker_count >= 2:
                needs_medication_fallback = True

        for phrase in phrases_to_remove:
            answer = answer.replace(phrase, "")

        answer = re.sub(r'^\s*(however[, ]*)?', '', answer, flags=re.IGNORECASE).strip()
        print(f"[get_medical_advice] cleaned answer={answer}")

        not_found_phrases = [
            "don't have information",
            "does not contain",
            "not in the provided",
            "cannot find",
            "no information about",
            "not mentioned",
            "no information",
        ]

        if answer and not any(phrase in answer.lower() for phrase in not_found_phrases) and not needs_medication_fallback:
            return answer

        if any(x in question_lower for x in ["medication", "medicine", "drug", "dosage", "dose"]):
            general_query = (
                f"Give the commonly used medications for {pretty_disease} ({disease}). "
                f"For each one, include medication name, purpose, dosage, frequency, duration, and important notes. "
                f"If a commonly used dosage is known, include it. "
                f"If a precise dosage is uncertain or varies by patient factors, say so clearly. "
                f"Format as a markdown table with columns: Medication, Purpose, Dosage, Frequency, Duration, Notes. "
                f"Do not include apologies or mention missing source material."
            )
        elif "symptom" in question_lower:
            general_query = (
                f"What are the symptoms of {pretty_disease} ({disease})? "
                f"Answer with clear bullet points. "
                f"Do not include apologies or mention missing source material."
            )
        elif any(x in question_lower for x in ["treatment", "therapy", "cure"]):
            general_query = (
                f"What are the treatments for {pretty_disease} ({disease})? "
                f"Answer with clear bullet points. "
                f"Do not include apologies or mention missing source material."
            )
        elif any(x in question_lower for x in ["cause", "causes", "trigger", "triggers", "why"]):
            general_query = (
                f"What are the causes or triggers of {pretty_disease} ({disease})? "
                f"Answer with clear bullet points. "
                f"Do not include apologies or mention missing source material."
            )
        else:
            general_query = (
                f"Provide a short explanation of {pretty_disease} ({disease}). "
                f"Answer directly and clearly. "
                f"Do not include apologies or mention missing source material."
            )

        print(f"[get_medical_advice] fallback query={general_query}")

        general_response = fallbackModel.invoke(general_query)
        print(f"[get_medical_advice] raw fallback response={general_response}")

        fallback_answer = ""
        if hasattr(general_response, "content"):
            fallback_answer = general_response.content or ""
        elif isinstance(general_response, dict):
            fallback_answer = general_response.get("content", "") or general_response.get("answer", "")
        else:
            fallback_answer = str(general_response)

        fallback_answer = fallback_answer.strip()

        for phrase in phrases_to_remove:
            fallback_answer = fallback_answer.replace(phrase, "")

        fallback_answer = re.sub(r'^\s*(however[, ]*)?', '', fallback_answer, flags=re.IGNORECASE).strip()
        print(f"[get_medical_advice] cleaned fallback_answer={fallback_answer}")

        if fallback_answer:
            return fallback_answer

        if "symptom" in question_lower:
            return (
                f"Common symptoms of {pretty_disease} may include:\n\n"
                f"• Visible skin changes or bumps\n"
                f"• Itching, irritation, or discomfort in some cases\n"
                f"• Changes in texture, size, or appearance depending on severity\n\n"
                f"For exact symptoms, please consult a healthcare professional."
            )

        if any(x in question_lower for x in ["medication", "medicine", "drug", "dosage", "dose"]):
            return (
                f"Common medications for {pretty_disease} depend on the exact diagnosis and severity.\n\n"
                f"Dosage not specified in provided material.\n\n"
                f"Please consult a healthcare professional for exact dosing."
            )

        if any(x in question_lower for x in ["treatment", "therapy", "cure"]):
            return (
                f"Treatment for {pretty_disease} depends on the exact diagnosis and severity.\n\n"
                f"Please consult a healthcare professional for the most appropriate treatment options."
            )

        if any(x in question_lower for x in ["cause", "causes", "trigger", "triggers", "why"]):
            return (
                f"{pretty_disease} can have different causes or triggers depending on the condition.\n\n"
                f"A healthcare professional can help determine the exact cause in your case."
            )

        return (
            f"{pretty_disease} is a skin condition. "
            f"Please ask about symptoms, causes, or treatments for more specific information."
        )

    except Exception as e:
        print(f"[get_medical_advice] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return (
            f"I could not generate a detailed answer for {get_pretty_disease_name(disease)} right now. "
            f"Please try asking again."
        )

# --- 5. HELPER FUNCTIONS ---
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def clean_response_for_display(text):
    if not text:
        return text

    lines = text.split("\n")
    cleaned_lines = []
    in_table = False

    for line in lines:
        if re.match(r"^\|\s*[-:]+\s*\|\s*[-:]+\s*\|", line):
            cleaned_lines.append(line)
            in_table = True
        elif "|" in line and line.strip().startswith("|"):
            cleaned_lines.append(line)
            in_table = True
        else:
            if in_table and line.strip() == "":
                cleaned_lines.append("")
            in_table = False
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def truncate_for_tts(text, max_chars=MAX_TTS_CHARS):
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    sentence_endings = [". ", "? ", "! ", ".\n", "?\n", "!\n"]
    best_break = -1
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > max_chars * 0.5:
            best_break = max(best_break, pos + len(ending) - 1)
    if best_break > 0:
        return text[:best_break + 1] + " (Summary continues in text)"
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        return text[:last_space] + "... (Full response in text)"
    return truncated + "... (Full response in text)"

def generate_speech_safe(text, max_retries=MAX_TTS_RETRIES):
    global last_tts_time
    if not text or len(text.strip()) < 10:
        return None
    if not tts_status:
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
                response_format="wav",
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
        files = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".wav")
        ]
        files.sort(key=os.path.getctime, reverse=True)
        for file in files[max_files:]:
            try:
                os.remove(file)
            except Exception:
                pass
    except Exception as e:
        print(f"Cleanup error: {e}")

# --- 6. ROUTES ---
@app.route("/")
def user_dashboard():
    cleanup_old_audio_files()
    return render_template("user.html")

@app.route("/admin")
def admin_dashboard():
    return render_template("admin.html")

@app.route("/static/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory("static/audio", filename)

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    image_file = request.files.get("image")
    actual_class = request.form.get("actual_class", "").strip()
    final_answer = ""
    audio_url = None
    session_id = "static_user"

    try:
        # CASE 1: Image uploaded
        if image_file and image_file.filename != "":
            file_path = os.path.join("static/uploads", f"img_{int(time.time())}_{image_file.filename}")
            image_file.save(file_path)

            try:
                if skin_model is not None:
                    image = Image.open(file_path).convert("RGB")
                    image_tensor = skin_transform(image).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        outputs = skin_model(image_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

                        top3_prob, top3_idx = torch.topk(probabilities, 3)
                        confidence, predicted = torch.max(probabilities, 0)

                        primary_disease_raw = skin_class_names[predicted.item()]
                        primary_disease = get_pretty_disease_name(primary_disease_raw)
                        primary_confidence = confidence.item() * 100

                        set_detected_disease(session_id, primary_disease_raw)
                        set_last_mentioned_disease(session_id, primary_disease_raw)

                        formatted_response = "🔍 **Skin Analysis Result:**\n\n"
                        formatted_response += f"**Primary Detection:** {primary_disease}\n"
                        formatted_response += f"**Confidence:** {primary_confidence:.1f}%\n\n"
                        formatted_response += "**Top 3 possibilities:**\n"

                        for i, (prob, idx) in enumerate(zip(top3_prob, top3_idx), 1):
                            disease_raw = skin_class_names[idx.item()]
                            disease = get_pretty_disease_name(disease_raw)
                            conf = prob.item() * 100
                            formatted_response += f"{i}. {disease} ({conf:.1f}%)\n"

                        formatted_response += "\n---\n"
                        # formatted_response += "💡 **Ask me about:**\n"
                        # formatted_response += "• 'What are the symptoms?'\n"
                        # formatted_response += "• 'What medications treat this?'\n"
                        # formatted_response += "• 'What causes this?'"

                        final_answer = formatted_response

                        prediction_history.append({
                            "Actual": actual_class if actual_class else "unknown",
                            "Predicted": primary_disease_raw,
                            "Confidence": f"{primary_confidence:.1f}%",
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })

                        if msg:
                            advice = get_medical_advice(primary_disease_raw, msg, session_id)
                            final_answer += f"\n\n{advice}"

                else:
                    base64_image = encode_image(file_path)
                    response = visionModel.invoke([
                        SystemMessage(content="Analyze this medical image and identify the skin condition."),
                        HumanMessage(content=[
                            {"type": "text", "text": msg if msg else "What skin condition is this?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ]),
                    ])
                    final_answer = response.content

            except Exception as e:
                print(f"Error analyzing image: {e}")
                import traceback
                traceback.print_exc()
                final_answer = "I couldn't analyze the image. Please make sure it's a clear photo of a skin condition."

            finally:
                try:
                    os.remove(file_path)
                except Exception:
                    pass

        # CASE 2: Text only
        else:
            if not msg:
                return jsonify({"answer": "Please provide a question or upload a skin image.", "audio_url": None})

            print(f"📝 User asked: {msg}")
            msg_lower = msg.lower().strip()

            greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
            is_greeting = any(re.search(rf"\b{re.escape(g)}\b", msg_lower) for g in greetings)

            if is_greeting and len(msg_lower.split()) <= 3:
                final_answer = "Hello! 👋 How can I help you today?"
            else:
                mentioned_disease_raw = None
                for keyword, disease_raw in DISEASE_KEYWORDS.items():
                    if keyword.lower() in msg_lower:
                        mentioned_disease_raw = disease_raw
                        break

                is_follow_up = any(indicator in msg_lower for indicator in FOLLOW_UP_INDICATORS)

                if mentioned_disease_raw:
                    pretty_disease = get_pretty_disease_name(mentioned_disease_raw)
                    print(f"🎯 User mentioned disease: {pretty_disease}")
                    set_last_mentioned_disease(session_id, mentioned_disease_raw)
                    advice = get_medical_advice(mentioned_disease_raw, msg, session_id)
                    final_answer = advice

                elif is_follow_up:
                    last_disease = get_last_mentioned_disease(session_id) or get_detected_disease(session_id)

                    if last_disease:
                        pretty_disease = get_pretty_disease_name(last_disease)
                        print(f"🔄 Follow-up about: {pretty_disease}")
                        advice = get_medical_advice(last_disease, msg, session_id)
                        final_answer = advice
                    else:
                        print("📚 No context, using RAG")
                        response = conversational_rag_chain.invoke(
                            {"input": msg},
                            config={"configurable": {"session_id": session_id}},
                        )
                        final_answer = response["answer"]

                else:
                    print("📚 General query, using RAG")
                    response = conversational_rag_chain.invoke(
                        {"input": msg},
                        config={"configurable": {"session_id": session_id}},
                    )
                    final_answer = response["answer"]

            print(f"✅ Answer generated (length: {len(final_answer)})")

        final_answer = clean_response_for_display(final_answer)

        try:
            if tts_status:
                audio_filename = generate_speech_safe(final_answer)
                if audio_filename:
                    audio_url = url_for("serve_audio", filename=audio_filename)
        except Exception as tts_error:
            print(f"TTS error: {tts_error}")

        return jsonify({
            "answer": final_answer,
            "audio_url": audio_url,
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "answer": "I encountered an error. Please try again.",
            "audio_url": None,
        })

@app.route("/admin/data", methods=["GET"])
def admin_data():
    global prediction_history

    total_samples = len(prediction_history)

    if total_samples == 0:
        return jsonify({
            "total_samples": 0,
            "correct_predictions": 0,
            "global_accuracy": 0,
            "audio_count": len([f for f in os.listdir("static/audio") if f.endswith(".wav")]),
            "per_class_stats": [],
            "history": [],
        })

    df = pd.DataFrame(prediction_history)
    known_df = df[df["Actual"] != "unknown"]
    known_samples = len(known_df)

    if known_samples > 0:
        correct_preds = len(known_df[known_df["Actual"] == known_df["Predicted"]])
        global_accuracy = (correct_preds / known_samples) * 100
    else:
        correct_preds = 0
        global_accuracy = 0

    per_class_stats = []
    tested_classes = sorted(known_df["Actual"].unique()) if known_samples > 0 else []

    for class_name in tested_classes:
        class_df = known_df[known_df["Actual"] == class_name]
        total = len(class_df)
        tp = len(class_df[class_df["Predicted"] == class_name])
        fn = total - tp
        fp = len(known_df[(known_df["Actual"] != class_name) & (known_df["Predicted"] == class_name)])
        tn = len(known_df[(known_df["Actual"] != class_name) & (known_df["Predicted"] != class_name)])

        accuracy = (tp / total) * 100 if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class_stats.append({
            "class": get_pretty_disease_name(class_name),
            "samples": total,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "accuracy": f"{accuracy:.1f}%",
            "precision": f"{precision:.2f}",
            "recall": f"{recall:.2f}",
            "f1_score": f"{f1:.2f}",
        })

    return jsonify({
        "total_samples": known_samples,
        "correct_predictions": correct_preds,
        "global_accuracy": round(global_accuracy, 1),
        "audio_count": len([f for f in os.listdir("static/audio") if f.endswith(".wav")]),
        "per_class_stats": per_class_stats,
        "history": prediction_history[-50:],
    })

@app.route("/admin/clear", methods=["POST"])
def admin_clear():
    global prediction_history
    prediction_history = []
    return jsonify({"success": True})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        session_id = request.json.get("session_id", "static_user")
        if session_id in store:
            del store[session_id]
        clear_session_data(session_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/status", methods=["GET"])
def status():
    audio_files = [f for f in os.listdir("static/audio") if f.endswith(".wav")]
    return jsonify({
        "audio_files_count": len(audio_files),
        "last_tts_time": last_tts_time,
        "skin_model_loaded": skin_model is not None,
        "skin_classes": len(skin_class_names) if skin_class_names else 0,
        "tts_enabled": tts_status,
    })

@app.route("/mute", methods=["POST"])
def mute():
    try:
        global tts_status
        tts_status = not request.json.get("isActive", False)
        print(f"TTS Status: {tts_status}")
        return jsonify({"success": True, "tts_enabled": tts_status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    cleanup_old_audio_files()
    print("=" * 60)
    print("🚀 MEDICAL AI ASSISTANT - TEST VERSION")
    print("=" * 60)
    print(f"👤 User Dashboard: http://localhost:8080/")
    print(f"📊 Admin Dashboard: http://localhost:8080/admin")
    print(f"🧠 Skin model loaded: {skin_model is not None}")
    if skin_class_names:
        print(f"🩺 Can detect: {len(skin_class_names)} conditions")
    print(f"🔊 TTS Enabled: {tts_status}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8080, debug=True)