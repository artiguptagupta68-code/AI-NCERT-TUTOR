# app.py — Optimized NCERT AI Tutor (Option B)
import os
from pathlib import Path
import zipfile
import io
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
# CONFIG
# ----------------------------
# Replace with your file ID if you want auto-download from Google Drive
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
DEFAULT_ZIP_NAME = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4
MAX_CONTEXT_CHARS = 1800  # keep prompts small to avoid OOM
MIN_ZIP_BYTES = 1024

st.set_page_config(page_title="NCERT AI Tutor", page_icon="🤖", layout="wide")

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div style="text-align:center">
      <h1 style="color:#2B3467">🤖 NCERT AI Tutor</h1>
      <p style="color:#4B5563;margin-top:-10px">Ask questions from your NCERT textbooks — instant, sourced answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 8, TOP_K)
chunk_size = st.sidebar.number_input("Chunk size (chars)", 200, 2000, CHUNK_SIZE, step=50)
chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", 0, 500, CHUNK_OVERLAP, step=10)
st.sidebar.write("Model:", EMBEDDING_MODEL_NAME, " | Generator:", GEN_MODEL_NAME)

# ----------------------------
# Helpers & Caching
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_generator_model(name=GEN_MODEL_NAME):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    if device == 0:
        model = model.to("cuda")
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return gen, tokenizer

@st.cache_resource(show_spinner=False)
def create_faiss_index(embeddings_np):
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    return index

def safe_extract_zip(zip_bytes_io, extract_to):
    # helper to extract from bytes buffer
    with zipfile.ZipFile(zip_bytes_io) as zf:
        zf.extractall(extract_to)

# ----------------------------
# Data loading: Download or upload
# ----------------------------
st.markdown("### Load NCERT ZIP")
col1, col2 = st.columns([2, 3])
with col1:
    st.write("Auto-download from Google Drive (optional)")
    if st.button("Download from Google Drive"):
        try:
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            st.info("Starting download...")
            gdown.download(url, DEFAULT_ZIP_NAME, quiet=False, fuzzy=True)
            st.success("Download finished.")
        except Exception as e:
            st.error(f"Download failed: {e}")
with col2:
    uploaded_file = st.file_uploader("Or upload `ncrt.zip` / your ZIP file here", type=["zip"])

zip_path = Path(DEFAULT_ZIP_NAME)
# If user uploaded a zip, save it
if uploaded_file is not None:
    zip_path = Path(EXTRACT_DIR) / "uploaded_ncert.zip"
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded and saved to {zip_path}")

# If the downloaded zip exists and is valid, proceed; else ask to upload
proceed = False
if zip_path.exists() and zip_path.stat().st_size >= MIN_ZIP_BYTES and zipfile.is_zipfile(zip_path):
    proceed = True
else:
    st.warning("No valid ZIP found locally. Use the uploader above or press Download from Google Drive then upload if needed.")

if not proceed:
    st.stop()

# ----------------------------
# Extract ZIP (with caching using last-modified timestamp)
# ----------------------------
extract_dir = Path(EXTRACT_DIR)
# Clear or create extraction dir
if extract_dir.exists():
    # keep existing extraction if present
    pass
else:
    extract_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(extract_dir)
st.success(f"ZIP extracted to: {extract_dir}")

# Handle nested zips
def extract_nested(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".zip"):
                nested = Path(root) / f
                nested_dir = Path(root) / nested.stem
                nested_dir.mkdir(exist_ok=True)
                try:
                    with zipfile.ZipFile(nested, 'r') as nz:
                        nz.extractall(nested_dir)
                except Exception:
                    # ignore bad nested zips
                    continue

extract_nested(extract_dir)

# ----------------------------
# Read PDFs into documents list (caching)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_documents_from_folder(folder):
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                fpath = Path(root) / fname
                try:
                    reader = PdfReader(str(fpath))
                    text_parts = []
                    for p in reader.pages:
                        txt = p.extract_text()
                        if txt:
                            text_parts.append(txt)
                    full_text = "\n".join(text_parts).strip()
                    if full_text:
                        docs.append({"file": fname, "text": full_text})
                except Exception as e:
                    # skip problematic pdfs
                    st.warning(f"Failed to load {fname}: {e}")
    return docs

with st.spinner("Loading PDF documents..."):
    documents = load_documents_from_folder(extract_dir)
st.success(f"Loaded {len(documents)} PDF documents.")

if len(documents) == 0:
    st.error("No PDF documents found in the extracted ZIP. Please upload a ZIP with PDFs.")
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
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = end - overlap
    return chunks

all_chunks = []
for doc in documents:
    chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=chunk_overlap)
    for i, c in enumerate(chunks):
        all_chunks.append({
            "doc_id": doc["file"],
            "chunk_id": f"{Path(doc['file']).stem}_chunk_{i}",
            "text": c
        })

st.info(f"Total chunks: {len(all_chunks)}")

# ----------------------------
# Build embeddings and FAISS index (cached)
# ----------------------------
embed_model = load_embedding_model()

@st.cache_resource(show_spinner=False)
def build_index_and_metadata(chunks):
    texts = [c["text"] for c in chunks]
    if len(texts) == 0:
        return None, None, None
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    index = create_faiss_index(np.array(embeddings))
    metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
    return index, embeddings, metadata

with st.spinner("Creating embeddings and FAISS index (cached)..."):
    index, embeddings_matrix, metadata = build_index_and_metadata(all_chunks)

if index is None:
    st.error("Failed to build FAISS index.")
    st.stop()

st.success("FAISS index built.")

# ----------------------------
# Load generator
# ----------------------------
with st.spinner("Loading generator model..."):
    generator, gen_tokenizer = load_generator_model()

# ----------------------------
# RAG functions
# ----------------------------
def retrieve(query, top_k=top_k):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    retrieved = []
    for idx in I[0]:
        if idx < len(metadata):
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
    if len(ctx_parts) == 0:
        return question  # fallback to only question
    context = "\n---\n".join(ctx_parts)
    prompt = (
        "You are an AI tutor specialized in NCERT textbooks. Use ONLY the provided context excerpts to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and clearly. If answer not found in context, say 'Information not available in NCERT excerpts provided.'"
    )
    return prompt

def generate_answer(query):
    retrieved = retrieve(query, top_k=top_k)
    if not retrieved:
        return {"answer": "No relevant documents found in NCERT content.", "sources": []}
    prompt = build_prompt(retrieved, query)
    try:
        out = generator(prompt, max_length=256, do_sample=False)[0]
        # Some pipelines return "generated_text" or "text"
        out_text = out.get("generated_text") if isinstance(out, dict) else str(out)
        if not out_text:
            out_text = out.get("text", "")
    except Exception as e:
        return {"answer": f"Generation error: {e}", "sources": []}
    sources = [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"]} for r in retrieved]
    return {"answer": out_text.strip(), "sources": sources}

# ----------------------------
# UI: Query and Results
# ----------------------------
st.markdown("---")
st.subheader("Ask a question from NCERT content")
user_q = st.text_input("Enter your question here", placeholder="E.g., What is osmosis?")

if user_q:
    t0 = time.time()
    with st.spinner("Retrieving and generating answer..."):
        res = generate_answer(user_q)
    t1 = time.time()
    elapsed = t1 - t0
    if res.get("answer"):
        st.markdown("**Answer:**")
        st.write(res["answer"])
    else:
        st.info("No answer generated.")

    st.markdown("**Sources used:**")
    for s in res.get("sources", []):
        st.write(f"- {s['doc_id']} / {s['chunk_id']}")

    st.caption(f"Response time: {elapsed:.2f}s (retrieval + generation)")

# ----------------------------
# Small debug / utilities
# ----------------------------
with st.expander("Debug / Info"):
    st.write(f"Documents loaded: {len(documents)}")
    st.write(f"Chunks: {len(all_chunks)}")
    st.write(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    st.write(f"Generator: {GEN_MODEL_NAME}")
    st.write(f"Top-K: {top_k}")
    if st.button("Run sample query"):
        sample = "What is the structure of a plant cell?"
        r = generate_answer(sample)
        st.write("Sample question:", sample)
        st.write("Sample answer:", r.get("answer"))
