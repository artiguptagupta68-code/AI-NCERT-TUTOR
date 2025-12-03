# app.py — Ultra-Clean Professional NCERT AI Tutor (Option D)
import os
import zipfile
import time
from pathlib import Path
from typing import List, Dict, Any

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
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"  # replace if needed
DEFAULT_ZIP_NAME = "ncrt.zip"
EXTRACT_DIR = Path("ncert_extracted")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Default to small model for CPU-friendly operation; users can change in sidebar
GEN_MODEL_NAME = "google/flan-t5-small"
TOP_K_DEFAULT = 4
MAX_CONTEXT_CHARS = 1800  # keep prompts within reasonable length
MIN_ZIP_BYTES = 1024

# ----------------------------
# Page config + header
# ----------------------------
st.set_page_config(page_title="NCERT AI Tutor — Pro", page_icon="📚🤖", layout="wide")

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:16px">
      <div style="font-size:42px">📚🤖</div>
      <div>
        <h1 style="margin:0;color:#0B3D91">NCERT AI Tutor — Professional</h1>
        <div style="color:#555">Ultra-clean RAG tutor, Teacher tools, PDF viewer, and Chat mode</div>
      </div>
    </div>
    <hr />
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar (settings + uploads)
# ----------------------------
st.sidebar.header("NCERT AI Tutor — Settings")

top_k = st.sidebar.slider("Top-K retrieved chunks", 1, 8, TOP_K_DEFAULT)
chunk_size = st.sidebar.number_input("Chunk size (chars)", 200, 2000, CHUNK_SIZE, step=50)
chunk_overlap = st.sidebar.number_input("Chunk overlap (chars)", 0, 500, CHUNK_OVERLAP, step=10)
gen_model_choice = st.sidebar.selectbox(
    "Generator model (change if you have resources)",
    options=["google/flan-t5-small", "google/flan-t5-base"],
    index=0
)
st.sidebar.write("Embedding model:", EMBEDDING_MODEL_NAME)
st.sidebar.write("Generator:", gen_model_choice)

# Upload or download
st.sidebar.markdown("---")
st.sidebar.write("Load NCERT ZIP (contains PDFs)")
download_btn = st.sidebar.button("Download from Google Drive (optional)")
uploaded_zip = st.sidebar.file_uploader("Or upload a ZIP file", type=["zip"])

# ----------------------------
# Utility functions & caching
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(name: str = EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)

@st.cache_resource(show_spinner=False)
def load_generator_pipeline(name: str):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    if device == 0:
        model = model.to("cuda")
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    return gen

@st.cache_resource(show_spinner=False)
def create_faiss_index_from_embeddings(embeddings_np: np.ndarray):
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    return index

def download_from_drive(file_id: str, out_path: Path) -> bool:
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(out_path), quiet=False, fuzzy=True)
        return True
    except Exception as e:
        st.sidebar.error(f"Download failed: {e}")
        return False

def safe_extract_zip(zip_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

# ----------------------------
# Prepare ZIP and extraction
# ----------------------------
zip_path = Path(DEFAULT_ZIP_NAME)
if download_btn:
    st.sidebar.info("Attempting download...")
    successful = download_from_drive(FILE_ID, zip_path)
    if successful:
        st.sidebar.success(f"Downloaded {zip_path}")

if uploaded_zip is not None:
    # save uploaded
    zip_path = EXTRACT_DIR / "uploaded_ncert.zip"
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    st.sidebar.success("Uploaded ZIP saved.")

# Validate zip presence
if not zip_path.exists() or zip_path.stat().st_size < MIN_ZIP_BYTES or not zipfile.is_zipfile(zip_path):
    st.warning("No valid ZIP file available. Please upload a ZIP with NCERT PDFs or download from Drive.")
    st.stop()

# Extract
with st.spinner("Extracting ZIP..."):
    safe_extract_zip(zip_path, EXTRACT_DIR)
st.success(f"ZIP extracted to: {EXTRACT_DIR}")

# Extract nested zips
def extract_nested_zips(base_dir: Path):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".zip"):
                nested = Path(root) / f
                nd = Path(root) / nested.stem
                nd.mkdir(exist_ok=True)
                try:
                    with zipfile.ZipFile(nested, 'r') as nz:
                        nz.extractall(nd)
                except Exception:
                    continue

extract_nested_zips(EXTRACT_DIR)

# ----------------------------
# Load PDFs robustly (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_pdfs_from_folder(folder: Path) -> List[Dict[str, str]]:
    docs = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                fpath = Path(root) / fname
                try:
                    reader = PdfReader(str(fpath))
                    pages = []
                    for p in reader.pages:
                        txt = p.extract_text()
                        if txt:
                            pages.append(txt)
                    full_text = "\n".join(pages).strip()
                    if full_text:
                        docs.append({"file": fname, "path": str(fpath), "text": full_text})
                except Exception as e:
                    # log and continue
                    st.warning(f"Failed reading {fname}: {e}")
    return docs

with st.spinner("Loading PDF documents..."):
    documents = load_pdfs_from_folder(EXTRACT_DIR)

if len(documents) == 0:
    st.error("No PDFs found in ZIP. Upload a ZIP containing NCERT PDFs.")
    st.stop()
st.success(f"Loaded {len(documents)} PDF documents.")

# ----------------------------
# Chunking (simple sliding window)
# ----------------------------
def chunk_text(text: str, chunk_size: int, overlap: int):
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
    parts = chunk_text(doc["text"], chunk_size=int(chunk_size), overlap=int(chunk_overlap))
    for i, p in enumerate(parts):
        all_chunks.append({
            "doc_id": doc["file"],
            "chunk_id": f"{Path(doc['file']).stem}_chunk_{i}",
            "text": p,
            "path": doc["path"]
        })

st.info(f"Total chunks created: {len(all_chunks)}")

# ----------------------------
# Build embeddings & FAISS (cached)
# ----------------------------
embed_model = load_embedding_model(EMBEDDING_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def build_embeddings_and_index(chunks):
    texts = [c["text"] for c in chunks]
    if len(texts) == 0:
        return None, None, None
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    index = create_faiss_index_from_embeddings(np.array(embeddings))
    return index, embeddings, texts

with st.spinner("Creating embeddings and FAISS index... (cached if available)"):
    index, embeddings_matrix, texts_list = build_embeddings_and_index(all_chunks)

if index is None:
    st.error("Failed to create FAISS index.")
    st.stop()
st.success("FAISS index built.")

# ----------------------------
# Load generator pipeline (cached)
# ----------------------------
with st.spinner("Loading generator..."):
    generator = load_generator_pipeline(gen_model_choice)
st.success("Generator loaded.")

# ----------------------------
# Retrieval & prompt utilities
# ----------------------------
def retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(all_chunks):
            results.append(all_chunks[idx])
    return results

def build_prompt(retrieved_chunks: List[Dict[str, Any]], question: str, max_context_chars=MAX_CONTEXT_CHARS):
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
        # fallback (no context)
        prompt = f"Question: {question}\nAnswer concisely. If not present, say 'Information not available in provided NCERT excerpts.'"
        return prompt
    context = "\n---\n".join(ctx_parts)
    prompt = (
        "You are an expert NCERT tutor. Use ONLY the context below to answer the question accurately and cite sources.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer concisely and clearly. If answer not contained, say 'Information not available in provided NCERT excerpts.'"
    )
    return prompt

def generate_answer(query: str, top_k: int = None):
    k = top_k or int(top_k_setting)
    retrieved = retrieve(query, k)
    if not retrieved:
        return {"answer": "No relevant NCERT content found.", "sources": []}
    prompt = build_prompt(retrieved, query, max_context_chars=MAX_CONTEXT_CHARS)
    try:
        out = generator(prompt, max_length=256, do_sample=False)[0]
        if isinstance(out, dict):
            text = out.get("generated_text") or out.get("text") or ""
        else:
            text = str(out)
    except Exception as e:
        text = f"Generation error: {e}"
    sources = [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "path": r.get("path")} for r in retrieved]
    return {"answer": text.strip(), "sources": sources}

# store top_k setting for closure
top_k_setting = top_k

# ----------------------------
# App Tabs: Tutor | Chat | Teacher | Admin
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Tutor", "Chat", "Teacher", "Admin"])

# ----------------------------
# Tutor Tab (single-question RAG)
# ----------------------------
with tab1:
    st.header("NCERT Tutor — Ask a question")
    q = st.text_input("Ask a question from the NCERT content:", key="tutor_q")
    col1, col2 = st.columns([3, 1])
    if col2.button("Ask", key="tutor_ask"):
        if not q:
            st.warning("Enter a question.")
        else:
            with st.spinner("Retrieving relevant passages and generating answer..."):
                start = time.time()
                res = generate_answer(q, top_k=top_k)
                elapsed = time.time() - start
            st.markdown("### Answer")
            st.write(res["answer"])
            st.markdown("### Sources")
            for s in res["sources"]:
                st.write(f"- {s['doc_id']} / {s['chunk_id']} — `{s.get('path')}`")
            st.caption(f"Response time: {elapsed:.2f}s")

    st.markdown("---")
    st.subheader("Preview a PDF")
    doc_names = [d["file"] for d in documents]
    sel = st.selectbox("Choose a document to preview", doc_names)
    if sel:
        sel_doc = next((d for d in documents if d["file"] == sel), None)
        if sel_doc:
            # Show path and display first N characters + option to download
            st.markdown(f"**{sel_doc['file']}**")
            st.write(sel_doc["text"][:3000] + ("..." if len(sel_doc["text"]) > 3000 else ""))
            st.download_button("Download PDF", data=open(sel_doc["path"], "rb"), file_name=sel_doc["file"])

# ----------------------------
# Chat Tab (conversational RAG with short memory)
# ----------------------------
with tab2:
    st.header("Chat with NCERT Tutor (session memory)")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    chat_col1, chat_col2 = st.columns([3, 1])
    user_msg = chat_col1.text_input("Your message:", key="chat_input")
    if chat_col2.button("Send", key="chat_send"):
        if not user_msg:
            st.warning("Type a message first.")
        else:
            with st.spinner("Generating reply..."):
                # Combine recent chat context (last 3 Qs) as a short context, plus retrieval on latest user_msg
                recent = " ".join([m["user"] for m in st.session_state.chat_history[-3:]])
                query_text = (recent + " " + user_msg).strip()
                res = generate_answer(query_text, top_k=top_k)
                bot_reply = res["answer"]
                st.session_state.chat_history.append({"user": user_msg, "bot": bot_reply})
    # Display chat history
    for entry in reversed(st.session_state.chat_history[-10:]):
        st.markdown(f"**You:** {entry['user']}")
        st.markdown(f"**Tutor:** {entry['bot']}")
        st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

# ----------------------------
# Teacher Tab (summarize, explain, MCQs)
# ----------------------------
with tab3:
    st.header("Teacher Mode — Summarize / Explain / Generate MCQs")
    st.write("Use retrieved context to create teaching materials.")

    # Select a document or use an arbitrary text
    teacher_doc = st.selectbox("Choose source document", ["<Use selected chunk>"] + [d["file"] for d in documents])
    teacher_query = st.text_area("Topic / instructive prompt (e.g., 'Explain osmosis in simple words')", height=80)
    teacher_action = st.selectbox("Action", ["Summarize", "Explain", "Generate MCQs (5)"])
    teacher_topk = st.number_input("Top-K (context chunks)", 1, 8, 4)

    if st.button("Create", key="teacher_create"):
        if not teacher_query:
            st.warning("Enter a prompt.")
        else:
            # Retrieve context from teacher_query
            res = generate_answer(teacher_query, top_k=teacher_topk)
            ctx = res.get("answer", "")
            # For MCQs, we will prompt generator to produce 5 MCQs based on context
            if teacher_action == "Summarize":
                prompt = f"Summarize the following NCERT context concisely:\n\n{ctx}\n\nSummary:"
            elif teacher_action == "Explain":
                prompt = f"Explain the following content step-by-step for a class 10 student:\n\n{ctx}\n\nExplanation:"
            else:  # MCQs
                prompt = (
                    f"Create 5 multiple-choice questions (question + 4 options + correct option indicated) "
                    f"based on the following NCERT context. Keep questions clear and concise.\n\nContext:\n{ctx}\n\nMCQs:"
                )
            try:
                out = generator(prompt, max_length=512, do_sample=False)[0]
                out_text = out.get("generated_text") or out.get("text") or str(out)
            except Exception as e:
                out_text = f"Generation error: {e}"
            st.markdown("### Result")
            st.write(out_text)

# ----------------------------
# Admin Tab (debug + utilities)
# ----------------------------
with tab4:
    st.header("Admin / Debug")
    st.write("Diagnostics and utilities for maintainers.")
    st.write(f"Documents loaded: {len(documents)}")
    st.write(f"Chunks: {len(all_chunks)}")
    st.write(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    st.write(f"Generator: {gen_model_choice}")
    st.write(f"Top-K default: {top_k}")
    st.write(f"Chunk size: {chunk_size}")
    st.write(f"Chunk overlap: {chunk_overlap}")

    if st.button("Run sample query"):
        sample_q = "What is photosynthesis?"
        with st.spinner("Running sample..."):
            r = generate_answer(sample_q)
        st.write("Sample question:", sample_q)
        st.write("Sample answer:", r.get("answer"))

    st.markdown("---")
    if st.button("Rebuild embeddings & index (force)"):
        # Clear caches by re-calling building function (Streamlit cache_resource requires new params)
        with st.spinner("Rebuilding embeddings..."):
            # we rebuild by calling the cached function again with same input; to force eviction, user must restart
            _, _, _ = build_embeddings_and_index(all_chunks)
        st.success("Rebuild requested (restart the app to fully clear caches if needed).")

    st.markdown("### Raw chunk preview (first 5)")
    for i, c in enumerate(all_chunks[:5]):
        st.write(f"{i}: {c['doc_id']} / {c['chunk_id']}")
        st.write(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))

st.markdown("<hr><div style='text-align:center;color:#777'>NCERT AI Tutor — Professional • Built for teachers and students</div>", unsafe_allow_html=True)
