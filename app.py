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
import shutil

# Import your custom prompts
from src.prompt import system_prompt

app = Flask(__name__)
load_dotenv()

# --- 1. CONFIGURATION & DIRECTORIES ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Initialize the Groq SDK Client for Audio
groq_client = Groq(api_key=GROQ_API_KEY)

for folder in ["static/audio", "static/uploads"]:
    os.makedirs(folder, exist_ok=True)

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

# --- 4. ROUTES ---
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    image_file = request.files.get("image")
    final_answer = ""

    try:
        # A. MULTIMODAL MODE
        if image_file and image_file.filename != '':
            file_path = os.path.join("static/uploads", image_file.filename)
            image_file.save(file_path)
            base64_image = encode_image(file_path)
            
            response = visionModel.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": msg if msg else "Analyze this medical image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ])
            final_answer = response.content
        
        # B. TEXT MODE (Now correctly using Conversational Chain)
        else:
            if not msg:
                return jsonify({"answer": "Please provide a question."})
            
            # Using session_id "static_user" for now to keep memory active
            response = conversational_rag_chain.invoke(
                {"input": msg},
                config={"configurable": {"session_id": "static_user"}}
            )
            final_answer = response["answer"]

        # C. TEXT-TO-SPEECH
        audio_filename = f"response_{int(time.time())}.wav"
        audio_path = os.path.join("static/audio", audio_filename)
        
        audio_response = groq_client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="troy",
            input=final_answer,
            response_format="wav"
        )
        audio_response.write_to_file(audio_path)
        
        return jsonify({
            "answer": final_answer, 
            "audio_url": url_for('static', filename=f'audio/{audio_filename}')
        })

    except Exception as e:
        print(f"Error in /get route: {e}")
        return jsonify({"answer": "I hit a snag. Check your console for details."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)