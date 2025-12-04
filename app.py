import os
import zipfile
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gdown

st.set_page_config(page_title="📚 AI NCERT Tutor", layout="wide")
st.title("📚 AI NCERT Tutor")

# ---------------- Download ZIP from Google Drive ----------------
FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"  # Replace with your Drive file ID
ZIP_PATH = "/tmp/ncrt.zip"
EXTRACT_FOLDER = "/tmp/ncert_extracted"

if not os.path.exists(ZIP_PATH):
    st.info("Downloading NCERT ZIP from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)
    st.success("Download completed!")

# ---------------- Extract ZIP ----------------
if not os.path.exists(EXTRACT_FOLDER):
    os.makedirs(EXTRACT_FOLDER, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)

    # Extract nested ZIPs
    for root, dirs, files in os.walk(EXTRACT_FOLDER):
        for file in files:
            if file.endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                nested_extract_path = os.path.join(root, file.replace(".zip", ""))
                os.makedirs(nested_extract_path, exist_ok=True)
                with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(nested_extract_path)

st.success("NCERT ZIP extracted successfully!")

# ---------------- Load documents ----------------
def load_documents(folder):
    texts = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            if file.lower().endswith(".pdf"):
                try:
                    doc = fitz.open(path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    texts.append(text)
                except Exception as e:
                    st.warning(f"Failed to read PDF {path}: {e}")
            elif file.lower().endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                except Exception as e:
                    st.warning(f"Failed to read TXT {path}: {e}")
    return texts

texts = load_documents(EXTRACT_FOLDER)

if len(texts) == 0:
    st.warning("No PDF/TXT files found!")
    st.stop()

# ---------------- Split documents ----------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(" ".join(texts))

# ---------------- Embeddings & FAISS ----------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(chunks)

embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)
embedding_matrix = np.array(embeddings).astype("float32")
index.add(embedding_matrix)

# ---------------- User query ----------------
query = st.text_input("Ask a question about the content:")

if query:
    # Retrieve top 5 relevant chunks
    model = SentenceTransformer("all-MiniLM-L
