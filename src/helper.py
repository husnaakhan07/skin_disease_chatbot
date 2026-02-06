from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated Path
from langchain_huggingface import HuggingFaceEmbeddings            # Modern Standard
from langchain_core.documents import Document                       # Critical Fix
from typing import List

# 1. Extract Data From the PDF File
def load_pdf_file(data):
    # 'glob="**/*.pdf"' allows searching in subfolders as well
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# 2. Filter to minimal docs (Keeps the vector store lean)
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# 3. Split the Data into Text Chunks
def text_split(extracted_data):
    # Adjusted chunk_overlap to 50 for better context retention
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# 4. Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings