# app.py — Minimal NCERT AI Tutor (Option A1)
import os
import zipfile
from pathlib import Path
import time

import streamlit as st
from pypdf import PdfReader
import gdown
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ----------------------------
# CONFIG (tweak if needed)
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"  # replace if you want auto-download
DEFAULT_ZIP = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"  # small is CPU friendlier
TOP_K = 4
MAX_CONTEXT_CHARS = 1500
MIN_ZIP_BYTES = 1024

st.set_page_config(page_title="NCERT AI Tutor - Minimal", page_icon="🤖", layout="centered")
st.title("🤖 NCERT AI Tutor — Minimal")
st.write("Upload a ZIP of NCERT PDFs or download from Google Drive (optional). Then ask a question.")

# ----------------------------
# Upload / Download ZIP
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload ncert ZIP (or any ZIP containing PDFs)", type=["zip"])
with col2:
    if st.button("Download from Google Drive"):
        try:
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, DEFAULT_ZIP, quiet=False, fuzzy=True)
            st.success("Downloaded ZIP (saved as ncrt.zip).")
        except Exception as e:
            st.error(f"Download failed: {e}")

zip_path = Path(DEFAULT_ZIP)
if uploaded:
    zip_path = Path(EXTRACT_DIR) / "uploaded_ncert.zip"
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with open(zip_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved uploaded ZIP to {zip_path}")

if not zip_path.exists() or zip_path.stat().st_size < MIN_ZIP_BYTES or not zipfile.is_zipfile(zip_path):
    st.warning("No valid ZIP found. Upload a ZIP with PDFs or download from Drive and try again.")
    st.stop()

# ----------------------------
# Extract ZIP
# ----------------------------
extract_dir = Path(EXTRACT_DIR)
extract_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_dir)
st.success(f"Extracted ZIP to {extract_dir}")

# Extract nested zips (simple)
for root, _, files in os.walk(extract_dir):
    for f in files:
        if f.lower().endswith(".zip"):
            nested = Path(root) / f
            nested_dir = Path(root) / Path(f).stem
            nested_dir.mkdir(exist_ok=True)
            try:
                with zipfile.ZipFile(nested, "r") as nz:
                    nz.extractall(nested_dir)
            except Exception:
                pass

# ----------------------------
# Read PDFs
# ----------------------------
st.info("Loading PDFs...")
documents = []
for root, _, files in os.walk(extract_dir):
    for fname in files:
        if fname.lower().endswith(".pdf"):
            path = Path(root) / fname
            try:
                reader = PdfReader(str(path))
                pages = []
                for p in reader.pages:
                    txt = p.extract_text()
                    if txt:
                        pages.append(txt)
                full = "\n".join(pages).strip()
                if full:
                    documents.append({"file": fname, "text": full})
            except Exception as e:
                st.warning(f"Failed to read {fname}: {e}")

st.success(f"Loaded {len(documents)} PDF documents.")
if len(documents) == 0:
    st.error("No readable PDFs found. Ensure ZIP contains PDF files.")
    st.stop()

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == L:
            break
        start = end - overlap
    return chunks

all_chunks = []
for doc in documents:
    doc_chunks = chunk_text(doc["text"])
    for i, c in enumerate(doc_chunks):
        all_chunks.append({"doc_id": doc["file"], "chunk_id": f"{Path(doc['file']).stem}_chunk_{i}", "text": c})

st.info(f"Total chunks: {len(all_chunks)}")

if len(all_chunks) == 0:
    st.error("No chunks created from PDFs.")
    st.stop()

# ----------------------------
# Build embeddings + FAISS
# ----------------------------
st.info("Loading embedding model and creating FAISS index (this may take some time)...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
texts = [c["text"] for c in all_chunks]
embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in all_chunks]
st.success("FAISS index built.")

# ----------------------------
# Load generator (model + pipeline)
# ----------------------------
device = 0 if torch.cuda.is_available() else -1
st.info(f"Loading generator model ({GEN_MODEL_NAME}) on {'GPU' if device==0 else 'CPU'}...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
if device == 0:
    gen_model = gen_model.to("cuda")
generator = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer, device=device)
st.success("Generator ready.")

# ----------------------------
# RAG functions
# ----------------------------
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    retrieved = []
    for idx in I[0]:
        if 0 <= idx < len(metadata):
            retrieved.append(metadata[idx])
    return retrieved

def build_prompt(retrieved_chunks, question, max_context_chars=MAX_CONTEXT_CHARS):
    ctx_parts = []
    total = 0
    for r in retrieved_chunks:
        t = (r.get("text") or "").strip()
        if not t:
            continue
        remaining = max_context_chars - total
        if remaining <= 0:
            break
        if len(t) > remaining:
            t = t[:remaining]
        ctx_parts.append(f"Source ({r['doc_id']} / {r['chunk_id']}):\n{t}\n")
        total += len(t)
    if not ctx_parts:
        return question
    context = "\n---\n".join(ctx_parts)
    prompt = (
        "You are an AI tutor specialized in NCERT content. Use the provided context to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"
    )
    return prompt

def generate_answer(query):
    retrieved = retrieve(query)
    if not retrieved:
        return {"answer": "No relevant documents found.", "sources": []}
    prompt = build_prompt(retrieved, query)
    try:
        out = generator(prompt, max_length=256, do_sample=False)[0]
        text = out.get("generated_text") or out.get("text") or str(out)
    except Exception as e:
        return {"answer": f"Generation error: {e}", "sources": []}
    sources = [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"]} for r in retrieved]
    return {"answer": text.strip(), "sources": sources}

# ----------------------------
# Streamlit Query UI
# ----------------------------
st.markdown("---")
st.subheader("Ask a question from NCERT content")
user_q = st.text_input("Enter your question here", placeholder="E.g., What is photosynthesis?")

if user_q:
    t0 = time.time()
    with st.spinner("Retrieving and generating answer..."):
        res = generate_answer(user_q)
    t1 = time.time()
    elapsed = t1 - t0

    st.markdown("**Answer:**")
    st.write(res["answer"])

    st.markdown("**Sources:**")
    for src in res.get("sources", []):
        st.write(f"- {src['doc_id']} / {src['chunk_id']}")

    st.caption(f"Response time: {elapsed:.2f}s")

# ----------------------------
# Small debug info
# ----------------------------
with st.expander("Debug info"):
    st.write(f"Documents: {len(documents)}")
    st.write(f"Chunks: {len(all_chunks)}")
    st.write(f"Embedding dim: {d}")
    st.write(f"Top-K used: {TOP_K}")
