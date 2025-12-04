import os
import requests
import zipfile
import streamlit as st
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

# -----------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------
st.set_page_config(page_title="📘 AI NCERT Tutor", layout="wide")
st.title("📘 AI NCERT Tutor (Google Drive → RAG)")

# Google Drive File ID
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

ZIP_PATH = "/tmp/ncrt.zip"
EXTRACT_DIR = "/tmp/ncert_extracted"

# -----------------------------------------------------------
# OpenAI Setup
# -----------------------------------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------------------------------------
# Download ZIP from Google Drive (no gdown)
# -----------------------------------------------------------
@st.cache_resource
def download_zip():
    r = requests.get(DOWNLOAD_URL, stream=True)
    if r.status_code != 200:
        raise Exception("Failed to download ZIP from Google Drive")

    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)

    return ZIP_PATH


with st.spinner("Downloading NCERT ZIP from Google Drive..."):
    download_zip()

# -----------------------------------------------------------
# Extract ZIP
# -----------------------------------------------------------
@st.cache_resource
def extract_zip():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR


with st.spinner("Extracting files..."):
    extract_zip()

# -----------------------------------------------------------
# Load PDF documents
# -----------------------------------------------------------
@st.cache_resource
def load_pdfs(folder):
    docs = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                try:
                    reader = PdfReader(path)
                    text = "\n".join([p.extract_text() or "" for p in reader.pages])
                    docs.append(text)
                except:
                    pass
    return docs


with st.spinner("Reading PDF files..."):
    documents = load_pdfs(EXTRACT_DIR)

if len(documents) == 0:
    st.error("No PDFs found in the ZIP file!")
    st.stop()

st.success(f"Loaded {len(documents)} PDF documents.")

# -----------------------------------------------------------
# Split into text chunks
# -----------------------------------------------------------
@st.cache_resource
def split_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = []
    for d in docs:
        chunks.extend(splitter.split_text(d))
    return chunks


chunks = split_into_chunks(documents)
st.success(f"Created {len(chunks)} text chunks.")

# -----------------------------------------------------------
# Build FAISS Index
# -----------------------------------------------------------
@st.cache_resource
def create_faiss(chunks):
    embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectors = embedder.encode(chunks, convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, embedder


with st.spinner("Building FAISS index..."):
    index, embed_model = create_faiss(chunks)

st.success("FAISS index ready.")

# -----------------------------------------------------------
# Retrieval function
# -----------------------------------------------------------
def retrieve_context(query):
    q = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q, 5)
    return "\n\n".join([chunks[i] for i in I[0]])

# -----------------------------------------------------------
# Answer using OpenAI (small, fast model)
# -----------------------------------------------------------
def generate_answer(question):
    context = retrieve_context(question)

    prompt = f"""
You are an NCERT expert. Answer strictly using the context below.

Context:
{context}

Question:
{question}

Answer clearly and correctly:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )

    return response.choices[0].message["content"]

# -----------------------------------------------------------
# User Input
# -----------------------------------------------------------
query = st.text_input("Ask your NCERT question:")

if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)

    st.subheader("Answer:")
    st.write(answer)
