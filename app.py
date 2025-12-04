import os
import zipfile
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
import numpy as np
import faiss
import gdown
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------------- CONFIG ----------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "/tmp/ncrt.zip"
EXTRACT_DIR = "/tmp/ncert_extracted"
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"  # SMALL MODEL
TOP_K = 4

st.set_page_config(page_title="📘 AI NCERT Tutor", layout="wide")
st.title("📘 AI NCERT Tutor (RAG)")


# ---------------- OpenAI Client ----------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------- Download ZIP ----------------
@st.cache_resource
def download_zip():
    if not os.path.exists(ZIP_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_PATH, quiet=False)
    return ZIP_PATH


download_zip()


# ---------------- Extract ZIP ----------------
@st.cache_resource
def extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR


extract_zip()


# ---------------- Read PDF Text ----------------
@st.cache_resource
def load_pdfs(folder):
    docs = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".pdf"):
                try:
                    reader = PdfReader(os.path.join(root, f))
                    text = "\n".join([p.extract_text() or "" for p in reader.pages])
                    docs.append(text)
                except:
                    pass
    return docs


pdf_docs = load_pdfs(EXTRACT_DIR)


# ---------------- Split into Chunks ----------------
def chunk(text, size=500):
    return [text[i:i + size] for i in range(0, len(text), size)]


all_chunks = []
for d in pdf_docs:
    all_chunks.extend(chunk(d))

st.success(f"Loaded {len(all_chunks)} chunks.")


# ---------------- Embeddings + FAISS ----------------
@st.cache_resource
def build_faiss(chunks):
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(chunks, convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, model


index, embed_model = build_faiss(all_chunks)


# ---------------- RAG Answer ----------------
def retrieve(query):
    q = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q, TOP_K)
    return [all_chunks[i] for i in I[0]]


def generate_answer(question):
    context = "\n\n".join(retrieve(question))

    prompt = f"""
You are an NCERT expert tutor.
Use the context below to answer the student's question accurately.

Context:
{context}

Question: {question}

Answer with correct NCERT facts:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )

    return response.choices[0].message["content"]


# ---------------- UI ----------------
question = st.text_input("Ask any NCERT question:")

if question:
    with st.spinner("Thinking..."):
        answer = generate_answer(question)
    st.subheader("Answer:")
    st.write(answer)
