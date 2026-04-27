from groq import Groq
from flask import Flask, render_template, jsonify, request, url_for, send_from_directory, abort
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
from datetime import datetime
from torchvision import transforms, models
from PIL import Image
from werkzeug.utils import secure_filename
from src.prompt import system_prompt

import os
import base64
import time
import random
import re
import json
import shutil
import pandas as pd
import torch
import torch.nn as nn


app = Flask(__name__)
load_dotenv()


# CONFIGURATION

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Live user uploads are NOT used for real accuracy.
# Keep this False so unknown live uploads remain Actual="unknown".
USE_PREDICTION_AS_ACTUAL_WHEN_MISSING = False

UPLOAD_DIR = "static/uploads"
AUDIO_DIR = "static/audio"
ADMIN_UPLOAD_DIR = "static/admin_uploads"
DATA_DIR = "data"
TEST_DATA_DIR = os.path.join(DATA_DIR, "test_data")

LIVE_LOG_FILE = os.path.join(DATA_DIR, "admin_logs.json")
EVAL_LOG_FILE = os.path.join(DATA_DIR, "evaluation_logs.json")

for folder in [UPLOAD_DIR, AUDIO_DIR, ADMIN_UPLOAD_DIR, DATA_DIR, TEST_DATA_DIR]:
    os.makedirs(folder, exist_ok=True)


# GROQ / TTS SETUP

groq_client = Groq(api_key=GROQ_API_KEY)

last_tts_time = 0
MIN_TTS_INTERVAL = 0.3
MAX_TTS_CHARS = 800
MAX_TTS_RETRIES = 3

tts_status = True
tts_modifier = "[quick]"


# JSON STORAGE HELPERS

def load_json_list(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return []


def save_json_list(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving {path}: {e}")


prediction_history = load_json_list(LIVE_LOG_FILE)
evaluation_history = load_json_list(EVAL_LOG_FILE)



# SKIN DISEASE MODEL SETUP

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
    if not model_name:
        return "Unknown"
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
    "it", "this", "that", "the condition", "the disease", "its", "for it",
    "tell me more", "what about", "symptoms of it", "treatment for it", "causes of it",
]

TOP_RESULT_KEYWORDS = [
    "acne", "actinickeratosis", "basalcellcarcinoma", "benigntumors", "bullous", "chickenpox", "cowpox",
    "eczema", "healthy", "hfmd", "infestationsbites", "lichen", "lupus", "measles", "melanocyticnevi", "moles",
    "monkeypox", "psoriasis", "rosacea", "seborrhkeratoses", "skincancer", "sunsunlightdamage", "tinea", "unknown",
    "vascularlesion", "vasculartumors", "vasculitis", "vitiligo", "warts"
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


# RAG SETUP

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

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

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


# MEDICAL ADVICE

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
                f"Answer clearly with bullet points. Do not include apologies or mention missing material. {question}"
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
                f"Answer clearly with bullet points. Do not include apologies or mention missing material. {question}"
            )
        elif any(x in question_lower for x in ["cause", "causes", "trigger", "triggers", "why"]):
            medical_query = (
                f"Based on the medical book, give only the causes or triggers of {pretty_disease} ({disease}). "
                f"Answer clearly with bullet points. Do not include apologies or mention missing material. {question}"
            )
        else:
            medical_query = (
                f"Based on the medical book, provide information only about {pretty_disease} ({disease}). "
                f"Answer directly and clearly. Do not include apologies or mention missing material. {question}"
            )

        response = medical_rag_chain.invoke({"input": medical_query})
        answer = (response.get("answer") or "").strip()

        needs_medication_fallback = False
        if any(x in question_lower for x in ["medication", "medicine", "drug", "dosage", "dose"]):
            low_info_markers = [
                "dosage not specified in provided material",
                "frequency not specified in provided material",
                "duration not specified in provided material",
            ]
            marker_count = sum(answer.lower().count(marker) for marker in low_info_markers)
            if marker_count >= 2:
                needs_medication_fallback = True

        for phrase in phrases_to_remove:
            answer = answer.replace(phrase, "")
        answer = re.sub(r"^\s*(however[, ]*)?", "", answer, flags=re.IGNORECASE).strip()

        not_found_phrases = [
            "don't have information", "does not contain", "not in the provided",
            "cannot find", "no information about", "not mentioned", "no information",
        ]

        if answer and not any(phrase in answer.lower() for phrase in not_found_phrases) and not needs_medication_fallback:
            return answer

        if any(x in question_lower for x in ["medication", "medicine", "drug", "dosage", "dose"]):
            general_query = (
                f"Give the commonly used medications for {pretty_disease} ({disease}). "
                f"For each one, include medication name, purpose, dosage, frequency, duration, and important notes. "
                f"If a precise dosage is uncertain or varies by patient factors, say so clearly. "
                f"Format as a markdown table with columns: Medication, Purpose, Dosage, Frequency, Duration, Notes."
            )
        elif "symptom" in question_lower:
            general_query = f"What are the symptoms of {pretty_disease} ({disease})? Answer with clear bullet points."
        elif any(x in question_lower for x in ["treatment", "therapy", "cure"]):
            general_query = f"What are the treatments for {pretty_disease} ({disease})? Answer with clear bullet points."
        elif any(x in question_lower for x in ["cause", "causes", "trigger", "triggers", "why"]):
            general_query = f"What are the causes or triggers of {pretty_disease} ({disease})? Answer with clear bullet points."
        else:
            general_query = f"Provide a short explanation of {pretty_disease} ({disease}). Answer directly and clearly."

        general_response = fallbackModel.invoke(general_query)
        if hasattr(general_response, "content"):
            fallback_answer = general_response.content or ""
        elif isinstance(general_response, dict):
            fallback_answer = general_response.get("content", "") or general_response.get("answer", "")
        else:
            fallback_answer = str(general_response)

        fallback_answer = fallback_answer.strip()
        for phrase in phrases_to_remove:
            fallback_answer = fallback_answer.replace(phrase, "")
        fallback_answer = re.sub(r"^\s*(however[, ]*)?", "", fallback_answer, flags=re.IGNORECASE).strip()

        if fallback_answer:
            return fallback_answer

        return (
            f"{pretty_disease} is a skin condition. "
            f"Please ask about symptoms, causes, medications, or treatments for more specific information."
        )

    except Exception as e:
        print(f"[get_medical_advice] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"I could not generate a detailed answer for {get_pretty_disease_name(disease)} right now. Please try asking again."


# HELPER FUNCTIONS

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
    # Remove images encapsulated by <div> tag from text
    pattern = r"<div[^>]*>([\s\S]*?)<\/div>"
    text = re.sub(pattern, "", text)
    
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
            audio_path = os.path.join(AUDIO_DIR, audio_filename)

            audio_response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice="troy",
                input=tts_modifier + tts_text,
                response_format="wav",
            )
            audio_response.write_to_file(audio_path)

            last_tts_time = time.time()

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
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
        files = [
            os.path.join(AUDIO_DIR, f)
            for f in os.listdir(AUDIO_DIR)
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


def save_admin_case_image(temp_file_path, original_filename):
    """Copy uploaded image into static/admin_uploads and return browser-visible URL."""
    safe_name = secure_filename(original_filename or "uploaded_image.jpg")
    timestamp = int(time.time() * 1000)
    admin_image_filename = f"patient_{timestamp}_{safe_name}"
    admin_image_path = os.path.join(ADMIN_UPLOAD_DIR, admin_image_filename)
    shutil.copy2(temp_file_path, admin_image_path)
    return url_for("static", filename=f"admin_uploads/{admin_image_filename}")


def add_prediction_log(actual_class, predicted_class, confidence, report, image_path=None, source="User Chatbot"):
    """Append one live prediction result for admin dashboard and persist it."""
    if not actual_class:
        actual_class = "unknown"

    row = {
        "Actual": actual_class,
        "Predicted": predicted_class,
        "PrettyPredicted": get_pretty_disease_name(predicted_class),
        "Confidence": f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else str(confidence),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Report": report or "",
        "ImagePath": image_path or "",
        "Source": source,
    }
    prediction_history.append(row)
    save_json_list(LIVE_LOG_FILE, prediction_history)
    print("✅ Live prediction log saved:", row)


def predict_skin_image(image_path):
    """Run the trained skin model on one image and return prediction, pretty label, confidence, top 3."""
    if skin_model is None:
        return None, None, None, []

    image = Image.open(image_path).convert("RGB")
    image_tensor = skin_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = skin_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        confidence, predicted = torch.max(probabilities, 0)

    predicted_class = skin_class_names[predicted.item()]
    pretty_prediction = get_pretty_disease_name(predicted_class)
    confidence_percent = confidence.item() * 100

    top3 = []
    for prob, idx in zip(top3_prob, top3_idx):
        raw = skin_class_names[idx.item()]
        top3.append({
            "raw_class": raw,
            "class": get_pretty_disease_name(raw),
            "confidence": f"{prob.item() * 100:.1f}%",
        })

    return predicted_class, pretty_prediction, confidence_percent, top3


def safe_eval_relative_path(image_path):
    """Make sure a requested evaluation image path stays inside TEST_DATA_DIR."""
    abs_base = os.path.abspath(TEST_DATA_DIR)
    abs_path = os.path.abspath(os.path.join(TEST_DATA_DIR, image_path))
    if not abs_path.startswith(abs_base):
        return None
    return abs_path


def build_evaluation_summary():
    total_samples = len(evaluation_history)

    if total_samples == 0:
        return {
            "total_samples": 0,
            "correct_predictions": 0,
            "global_accuracy": 0,
            "per_class_stats": [],
            "history": [],
        }

    df = pd.DataFrame(evaluation_history)

    for col in ["Actual", "Predicted", "Confidence", "Correct"]:
        if col not in df.columns:
            df[col] = ""

    correct_predictions = int(len(df[df["Actual"] == df["Predicted"]]))
    global_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

    per_class_stats = []

    for class_name in sorted(df["Actual"].unique()):
        total = int(len(df[df["Actual"] == class_name]))
        tp = int(len(df[(df["Actual"] == class_name) & (df["Predicted"] == class_name)]))
        fn = int(len(df[(df["Actual"] == class_name) & (df["Predicted"] != class_name)]))
        fp = int(len(df[(df["Actual"] != class_name) & (df["Predicted"] == class_name)]))
        tn = int(len(df[(df["Actual"] != class_name) & (df["Predicted"] != class_name)]))

        accuracy = (tp / total) * 100 if total > 0 else 0

        per_class_stats.append({
            "class": get_pretty_disease_name(class_name),
            "raw_class": class_name,
            "samples": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": f"{accuracy:.1f}%",
        })

    return {
        "total_samples": int(total_samples),
        "correct_predictions": int(correct_predictions),
        "global_accuracy": round(global_accuracy, 1),
        "per_class_stats": per_class_stats,
        "history": evaluation_history[-50:],
    }



# ROUTES

@app.route("/")
def user_dashboard():
    cleanup_old_audio_files()
    return render_template("user.html")


@app.route("/admin")
def admin_dashboard():
    return render_template("admindashboard.html")


@app.route("/static/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    image_file = request.files.get("image")
    final_answer = ""
    audio_url = None
    session_id = request.form.get("session_id", "static_user").strip() or "static_user"
    source = request.form.get("source", "User Chatbot").strip() or "User Chatbot"

    try:
        # CASE 1: Image uploaded
        if image_file and image_file.filename != "":
            safe_name = secure_filename(image_file.filename)
            temp_filename = f"img_{int(time.time() * 1000)}_{safe_name}"
            file_path = os.path.join(UPLOAD_DIR, temp_filename)
            image_file.save(file_path)

            try:
                admin_image_url = save_admin_case_image(file_path, image_file.filename)

                if skin_model is not None:
                    primary_disease_raw, primary_disease, primary_confidence, top3 = predict_skin_image(file_path)

                    set_detected_disease(session_id, primary_disease_raw)
                    set_last_mentioned_disease(session_id, primary_disease_raw)

                    formatted_response = "🔍 **Skin Analysis Result:**\n\n"
                    formatted_response += f"**Primary Detection:** {primary_disease}\n"
                    formatted_response += f"**Confidence:** {primary_confidence:.1f}%\n\n"
                    formatted_response += "**Top 3 possibilities:**\n"

                    top_results_path = "./static/top_results/"

                    for i, item in enumerate(top3, 1):
                        formatted_response += f"{i}. {item['class']} ({item['confidence']})\n"
                        # Attach example image of disease
                        disease_raw = item['raw_class']
                        if disease_raw in TOP_RESULT_KEYWORDS:
                            print("DISEASE FOUND: " + str(item['raw_class']))
                            top_image_filename = "".join(os.listdir(top_results_path + str(disease_raw).lower()))
                            top_image_path = top_results_path + str(disease_raw).lower() + "/" + top_image_filename
                            formatted_response += f"<div><img src='{top_image_path}' class=\"top-result\"></div>"
                        else:
                            print("DISEASE NOT FOUND: " + str(item['raw_class']))

                    formatted_response += "\n---\n"
                    final_answer = formatted_response

                    if msg:
                        advice = get_medical_advice(primary_disease_raw, msg, session_id)
                        final_answer += f"\n\n{advice}"

                    add_prediction_log(
                        actual_class="unknown",
                        predicted_class=primary_disease_raw,
                        confidence=primary_confidence,
                        report=final_answer,
                        image_path=admin_image_url,
                        source=source,
                    )

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

                    add_prediction_log(
                        actual_class="unknown",
                        predicted_class="vision_model_result",
                        confidence="N/A",
                        report=final_answer,
                        image_path=admin_image_url,
                        source=source,
                    )

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
                    set_last_mentioned_disease(session_id, mentioned_disease_raw)
                    final_answer = get_medical_advice(mentioned_disease_raw, msg, session_id)

                elif is_follow_up:
                    last_disease = get_last_mentioned_disease(session_id) or get_detected_disease(session_id)

                    if last_disease:
                        final_answer = get_medical_advice(last_disease, msg, session_id)
                    else:
                        response = conversational_rag_chain.invoke(
                            {"input": msg},
                            config={"configurable": {"session_id": session_id}},
                        )
                        final_answer = response["answer"]

                else:
                    response = conversational_rag_chain.invoke(
                        {"input": msg},
                        config={"configurable": {"session_id": session_id}},
                    )
                    final_answer = response["answer"]

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


# LIVE CASES
@app.route("/admin/data", methods=["GET"])
def admin_data():
    audio_count = len([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])
    return jsonify({
        "total_samples": len(prediction_history),
        "audio_count": audio_count,
        "history": prediction_history[-50:],
    })


@app.route("/admin/clear", methods=["POST"])
def admin_clear():
    global prediction_history
    prediction_history = []
    save_json_list(LIVE_LOG_FILE, prediction_history)

    if os.path.exists(ADMIN_UPLOAD_DIR):
        for filename in os.listdir(ADMIN_UPLOAD_DIR):
            file_path = os.path.join(ADMIN_UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    return jsonify({"success": True})


#MODEL EVALUATION 
@app.route("/admin/evaluation/classes", methods=["GET"])
def evaluation_classes():
    classes = []

    if os.path.exists(TEST_DATA_DIR):
        for cls in sorted(os.listdir(TEST_DATA_DIR)):
            class_path = os.path.join(TEST_DATA_DIR, cls)
            if os.path.isdir(class_path):
                classes.append(cls)

    return jsonify({"classes": classes})


@app.route("/admin/evaluation/sample", methods=["GET"])
def evaluation_sample():
    selected_class = request.args.get("class", "").strip()
    if not selected_class:
        return jsonify({"error": "Missing class"}), 400

    class_path = os.path.abspath(os.path.join(TEST_DATA_DIR, selected_class))
    base_path = os.path.abspath(TEST_DATA_DIR)

    if not class_path.startswith(base_path) or not os.path.isdir(class_path):
        return jsonify({"error": "Invalid class"}), 400

    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    images = [f for f in os.listdir(class_path) if f.lower().endswith(valid_exts)]

    if not images:
        return jsonify({"error": "No images found for this class"}), 404

    chosen = random.choice(images)
    relative_path = os.path.join(selected_class, chosen).replace("\\", "/")

    return jsonify({
        "actual_class": selected_class,
        "image_path": relative_path,
        "image_url": url_for("evaluation_image", image_path=relative_path),
    })


@app.route("/admin/evaluation/image/<path:image_path>", methods=["GET"])
def evaluation_image(image_path):
    abs_path = safe_eval_relative_path(image_path)
    if abs_path is None or not os.path.exists(abs_path):
        abort(404)

    directory = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)
    return send_from_directory(directory, filename)


@app.route("/admin/evaluation/run-one", methods=["POST"])
def evaluation_run_one():
    global evaluation_history

    data = request.get_json(silent=True) or {}
    image_path = data.get("image_path", "").strip()
    actual_class = data.get("actual_class", "").strip()

    if not image_path or not actual_class:
        return jsonify({"error": "Missing image_path or actual_class"}), 400

    abs_path = safe_eval_relative_path(image_path)
    if abs_path is None or not os.path.exists(abs_path):
        return jsonify({"error": "Invalid image path"}), 400

    if skin_model is None:
        return jsonify({"error": "Skin model is not loaded"}), 500

    predicted_class, pretty_prediction, confidence_percent, top3 = predict_skin_image(abs_path)
    correct = actual_class == predicted_class

    row = {
        "Actual": actual_class,
        "PrettyActual": get_pretty_disease_name(actual_class),
        "Predicted": predicted_class,
        "PrettyPredicted": pretty_prediction,
        "Confidence": f"{confidence_percent:.1f}%",
        "Correct": bool(correct),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ImagePath": url_for("evaluation_image", image_path=image_path),
    }

    evaluation_history.append(row)
    save_json_list(EVAL_LOG_FILE, evaluation_history)

    return jsonify({
        "actual_class": actual_class,
        "actual_pretty": get_pretty_disease_name(actual_class),
        "predicted_class": pretty_prediction,
        "raw_predicted_class": predicted_class,
        "confidence": f"{confidence_percent:.1f}%",
        "correct": bool(correct),
        "top3": top3,
    })


@app.route("/admin/evaluation/summary", methods=["GET"])
def evaluation_summary():
    return jsonify(build_evaluation_summary())


@app.route("/admin/evaluation/clear", methods=["POST"])
def evaluation_clear():
    global evaluation_history
    evaluation_history = []
    save_json_list(EVAL_LOG_FILE, evaluation_history)
    return jsonify({"success": True})


#  MISC
@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id", "static_user")
        if session_id in store:
            del store[session_id]
        clear_session_data(session_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/status", methods=["GET"])
def status():
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    return jsonify({
        "audio_files_count": len(audio_files),
        "last_tts_time": last_tts_time,
        "skin_model_loaded": skin_model is not None,
        "skin_classes": len(skin_class_names) if skin_class_names else 0,
        "tts_enabled": tts_status,
        "live_history_count": len(prediction_history),
        "evaluation_history_count": len(evaluation_history),
    })


@app.route("/mute", methods=["POST"])
def mute():
    try:
        global tts_status
        data = request.get_json(silent=True) or {}
        tts_status = not data.get("isActive", False)
        return jsonify({"success": True, "tts_enabled": tts_status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    cleanup_old_audio_files()
    print("=" * 60)
    print("MEDICAL AI ASSISTANT")
    print("=" * 60)
    print("👤 User Dashboard: http://localhost:8080/")
    print("📊 Admin Dashboard: http://localhost:8080/admin")
    print(f"Skin model loaded: {skin_model is not None}")
    if skin_class_names:
        print(f"🩺 Can detect: {len(skin_class_names)} conditions")
    print(f"TTS Enabled: {tts_status}")
    print(f"Live case rows loaded: {len(prediction_history)}")
    print(f"Evaluation rows loaded: {len(evaluation_history)}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8080, debug=True)
