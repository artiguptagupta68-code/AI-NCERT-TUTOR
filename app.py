import os
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# ----------------------------
# CONFIG
# ----------------------------
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4

st.title("📘 NCERT AI Tutor (RAG)")

# ----------------------------
# STEP 1: Read PDFs
# ----------------------------
documents = []
for root, dirs, files in os.walk(EXTRACT_DIR):
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

st.text(f"Loaded {len(documents)} PDF documents.")

# ----------------------------
# STEP 2: Chunk text
# ----------------------------
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

all_chunks = []
for doc in documents:
    for chunk in chunk_text(doc["text"]):
        all_chunks.append(chunk)

st.text(f"Total chunks: {len(all_chunks)}")

# ----------------------------
# STEP 3: Create embeddings and FAISS index
# ----------------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = embed_model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True).astype("float32")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
st.text("FAISS index built.")

# ----------------------------
# STEP 4: Load generator
# ----------------------------
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
if device == 0:
    gen_model = gen_model.to("cuda")
generator = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer, device=device)

# ----------------------------
# STEP 5: RAG functions
# ----------------------------
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    return [all_chunks[i] for i in I[0]]

def build_prompt(retrieved_chunks, question, max_context_chars=3000):
    ctx_parts = []
    total = 0
    for r in retrieved_chunks:
        t = r.strip()
        if not t:
            continue
        remaining = max_context_chars - total
        if remaining <= 0:
            break
        if len(t) > remaining:
            t = t[:remaining]
        ctx_parts.append(t)
        total += len(t)
    context = "\n---\n".join(ctx_parts)
    prompt = (
        "You are an AI tutor specialized in NCERT content. Use the provided context excerpts to answer the question accurately.\n\n"
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

# ----------------------------
# STEP 6: Streamlit UI
# ----------------------------
query = st.text_input("Ask a question about NCERT content:")
if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)
        st.write("**Answer:**", answer)
