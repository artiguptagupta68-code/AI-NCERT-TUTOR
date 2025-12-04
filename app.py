import os
import zipfile
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import gdown

# ---------------- CONFIG ----------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "/tmp/ncert.zip"
EXTRACT_DIR = "/tmp/ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4

st.set_page_config(page_title="📚 AI NCERT Tutor", layout="wide")
st.title("📚 AI NCERT Tutor")

# ---------------- Download ZIP from Google Drive ----------------
@st.cache_resource
def download_zip(file_id, zip_path):
    if not os.path.exists(zip_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)
    return zip_path

zip_path = download_zip(FILE_ID, ZIP_PATH)

# ---------------- Extract ZIP ----------------
@st.cache_resource
def extract_zip(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Extract nested ZIPs
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.endswith(".zip"):
                    nested_zip_path = os.path.join(root, file)
                    nested_extract_path = os.path.join(root, file.replace(".zip", ""))
                    os.makedirs(nested_extract_path, exist_ok=True)
                    with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                        zip_ref.extractall(nested_extract_path)
    return extract_dir

extract_dir = extract_zip(zip_path, EXTRACT_DIR)

# ---------------- Load PDFs ----------------
@st.cache_resource
def load_documents(folder):
    documents = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                try:
                    reader = PdfReader(path)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():
                        documents.append({"file": file, "text": text})
                except Exception as e:
                    st.warning(f"Failed to read PDF: {file}, {e}")
    return documents

documents = load_documents(extract_dir)

if len(documents) == 0:
    st.error("No PDF files found in ZIP!")
    st.stop()

st.success(f"Loaded {len(documents)} PDF documents.")

# ---------------- Chunk text ----------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

@st.cache_resource
def create_chunks(documents):
    all_chunks = []
    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append(chunk)
    return all_chunks

all_chunks = create_chunks(documents)
st.success(f"Total chunks: {len(all_chunks)}")

# ---------------- Embeddings & FAISS ----------------
@st.cache_resource
def create_faiss_index(chunks):
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embed_model

index, embed_model = create_faiss_index(all_chunks)
st.success("FAISS index built.")

# ---------------- Load LLM ----------------
@st.cache_resource
def load_generator():
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    if device == 0:
        model = model.to("cuda")
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return generator

generator = load_generator()
st.success("Generator model loaded.")

# ---------------- RAG Functions ----------------
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    return [all_chunks[i] for i in I[0]]

def build_prompt(retrieved_chunks, question):
    context = "\n---\n".join([c.strip() for c in retrieved_chunks])
    prompt = (
        "You are an AI tutor specialized in NCERT content. Use the provided context to answer the question accurately.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and clearly:"
    )
    return prompt

def generate_answer(query):
    retrieved = retrieve(query)
    if not retrieved:
        return "No relevant documents found."
    prompt = build_prompt(retrieved, query)
    out = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return out.strip()

# ---------------- Streamlit UI ----------------
query = st.text_input("Ask a question about NCERT content:")
if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)
        st.subheader("Answer:")
        st.write(answer)
