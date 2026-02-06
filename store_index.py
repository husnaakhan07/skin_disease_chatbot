import os
import time
from dotenv import load_dotenv
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

# --- 1. CONFIGURATION ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY 

# --- 2. DATA PROCESSING ---
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# --- 3. EMBEDDINGS ---
embeddings = download_hugging_face_embeddings()

# --- 4. PINECONE INITIALIZATION ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    print(f"Creating index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for index to be initialized (2026 best practice)
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# --- 5. VECTOR STORE UPLOAD ---
print("Uploading vectors to Pinecone... this may take a moment.")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings 
)

print("Ingestion complete. Your medical data is now vectorized in Pinecone.")