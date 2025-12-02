import os
import zipfile
import shutil
import glob
import json
import faiss
import numpy as np
import streamlit as st

# Text and PDF handling
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (fitz) not installed. Please pip install PyMuPDF") from e

# Embeddings
from sentence_transformers import SentenceTransformer

# Transformers for Falcon
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------
# Config
# -------------------------
ZIP_PATH = "ncrt subject.zip"        # <-- place this file alongside app.py
EXTRACT_FOLDER = "ncrt_extracted"
NESTED_EXTRACTED_FLAG = "nested_extracted.flag"
FAISS_INDEX_PATH = "ncrt_faiss.index"
CHUNKS_PATH = "ncrt_chunks.npy"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "tiiuae/falcon-7b-instruct"  # chosen model A
CHUNK_CHARS = 1000     # ~characters per chunk
CHUNK_OVERLAP = 200    # overlap chars
TOP_K = 3

# -------------------------
# Utilities: safe filesystem
# -------------------------
def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# -------------------------
# Unzip top-level and nested zips
# -------------------------
def extract_top_and_nested(top_zip=ZIP_PATH, dest=EXTRACT_FOLDER):
    if not os.path.exists(top_zip):
        raise FileNotFoundError(f"ZIP file not found at path: {top_zip}")
    # remove old extraction (optional) only if we want fresh
    if os.path.exists(dest):
        # don't remove if we have indexed already
        pass
    ensure_folder(dest)
    with zipfile.ZipFile(top_zip, "r") as z:
        z.extractall(dest)
    # Extract nested zips recursively
    for root, _, files in os.walk(dest):
        for fname in files:
            if fname.lower().endswith(".zip"):
                zip_path = os.path.join(root, fname)
                out_dir = os.path.join(root, fname[:-4])
                ensure_folder(out_dir)
                try:
                    with zipfile.ZipFile(zip_path, "r") as nz:
                        nz.extractall(out_dir)
                except zipfile.BadZipFile:
                    st.warning(f"Nested ZIP corrupted or unreadable: {zip_path}")

# -------------------------
# Load docs (PDF/TXT) recursively
# -------------------------
class SimpleDoc:
    def __init__(self, text, source):
        self.page_content = text
        self.metadata = {"source": source}

def load_documents_from_folder(folder=EXTRACT_FOLDER):
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            path = os.path.join(root, fname)
            lower = fname.lower()
            try:
                if lower.endswith(".pdf"):
                    txt = ""
                    pdf = fitz.open(path)
                    for page in pdf:
                        txt += page.get_text()
                    if txt.strip():
                        docs.append(SimpleDoc(txt, os.path.relpath(path, folder)))
                elif lower.endswith(".txt"):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                        if txt.strip():
                            docs.append(SimpleDoc(txt, os.path.relpath(path, folder)))
                # you can add .docx handling if needed
            except Exception as e:
                st.warning(f"Failed to read file {path}: {e}")
    return docs

# -------------------------
# Chunker (character-based)
# -------------------------
def chunk_text(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks

def docs_to_chunks(docs):
    chunks = []
    for d in docs:
        pieces = chunk_text(d.page_content)
        for i, p in enumerate(pieces):
            chunks.append({"text": p, "source": d.metadata.get("source"), "chunk_id": i})
    return chunks

# -------------------------
# Embeddings and FAISS
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(name=EMBED_MODEL_NAME):
    return SentenceTransformer(name)

def build_faiss_index(chunk_texts, embed_model, faiss_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH):
    # compute embeddings in batches to avoid OOM
    batch_size = 128
    embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        emb = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    faiss.write_index(idx, faiss_path)
    np.save(chunks_path, np.array(chunk_texts, dtype=object))
    return idx

def load_faiss_if_exists(faiss_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH):
    if os.path.exists(faiss_path) and os.path.exists(chunks_path):
        idx = faiss.read_index(faiss_path)
        chunks = list(np.load(chunks_path, allow_pickle=True))
        return idx, chunks
    return None, None

# -------------------------
# LLM load
# -------------------------
@st.cache_resource(show_spinner=False)
def load_llm(model_name=LLM_MODEL_NAME):
    # tokenizer + model load with device_map auto; use dtype instead of torch_dtype
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # try loading with half precision to save memory
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        # fallback to cpu or non-fp16
        st.warning(f"Auto device_map load failed ({e}). Falling back to cpu load (slower).")
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": "cpu"})
    return tokenizer, model

def generate_with_model(prompt, tokenizer, model, max_new_tokens=200, temperature=0.7):
    # Tokenize with truncation and move to model device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# -------------------------
# Retrieval
# -------------------------
def retrieve_top_k(query, embed_model, index, chunks, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue
        results.append(chunks[idx])
    return results

# -------------------------
# MAIN: prepare backend (extract & index) — executed once
# -------------------------
st.set_page_config(page_title="NCERT AI Tutor (Offline RAG)", layout="centered")
st.title("📚 NCERT AI Tutor (Backend-processed, Offline RAG)")

# Ensure extract folder exists and contains nested content
if not os.path.exists(EXTRACT_FOLDER) or not os.path.exists(NESTED_EXTRACTED_FLAG):
    st.info("Extracting top-level ZIP and nested ZIP files (backend)...")
    try:
        extract_top_and_nested(ZIP_PATH, EXTRACT_FOLDER)
        # mark flag file
        with open(NESTED_EXTRACTED_FLAG, "w") as f:
            f.write("done")
    except Exception as e:
        st.error(f"Failed to extract ZIP: {e}")
        st.stop()

# Load existing FAISS or build
index, chunks_texts = load_faiss_if_exists()
embed_model = load_embedding_model()

if index is None or chunks_texts is None:
    st.info("Creating document index (this may take a while)...")
    docs = load_documents_from_folder(EXTRACT_FOLDER)
    if len(docs) == 0:
        st.error("No readable documents found in the extracted folder. Check ZIP contents (PDF/TXT).")
        st.stop()

    # chunk
    chunk_objs = docs_to_chunks(docs)
    chunk_texts = [c["text"] for c in chunk_objs]
    # build faiss
    index = build_faiss_index(chunk_texts, embed_model)
    chunks_texts = chunk_texts
    st.success(f"Index built with {index.ntotal} vectors.")
else:
    st.success(f"Loaded existing index with {index.ntotal} vectors.")

# Load LLM (this may be slow)
with st.spinner("Loading LLM (may take a few minutes)..."):
    try:
        tokenizer, model = load_llm()
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        st.stop()

# -------------------------
# UI: Q&A only
# -------------------------
st.markdown("Ask questions from the NCERT content (backend processed).")
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            top_chunks = retrieve_top_k(query, embed_model, index, chunks_texts, top_k=TOP_K)
            context = "\n\n".join(top_chunks)
            prompt = f"You're an NCERT tutor. Use ONLY the context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
            try:
                answer = generate_with_model(prompt, tokenizer, model, max_new_tokens=200, temperature=0.7)
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("---")
                st.markdown("### Retrieved context (for transparency)")
                for i, c in enumerate(top_chunks, start=1):
                    st.markdown(f"**Context {i}:** {c[:800]}{'...' if len(c)>800 else ''}")
            except Exception as e:
                st.error(f"Generation failed: {e}")
