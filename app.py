
import streamlit as st
import zipfile
import os
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import shutil
import gdown
from pathlib import Path 
from pypdf import PdfReader


FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4
# ----------------------------
# STEP 0: Streamlit UI for info
# ----------------------------
st.title("NCERT AI Tutor")
st.text("Downloading and extracting NCERT ZIP from Google Drive...")

# ----------------------------
# STEP 1: Download ZIP
# ----------------------------
if not os.path.exists(ZIP_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)
st.text("Download completed.")

# ----------------------------
# STEP 2: Validate ZIP
# ----------------------------
if not zipfile.is_zipfile(ZIP_PATH):
    st.error(f"{ZIP_PATH} is not a valid ZIP file. Check Google Drive link or permissions.")
    st.stop()
else:
    st.text("ZIP file is valid!")

# ----------------------------
# STEP 3: Extract ZIP
# ----------------------------
os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
st.text(f"ZIP extracted to: {EXTRACT_DIR}")

# Handle nested ZIPs (like class 11/12 PDFs inside)
for root, dirs, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.lower().endswith(".zip"):
            nested_zip_path = os.path.join(root, file)
            nested_extract_dir = os.path.join(root, Path(file).stem)
            os.makedirs(nested_extract_dir, exist_ok=True)
            with zipfile.ZipFile(nested_zip_path, 'r') as nz:
                nz.extractall(nested_extract_dir)

st.text("All nested ZIPs extracted.")

# ----------------------------
# STEP 4: Read PDFs
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

extract_folder = "/mount/src/ai-ncert-tutor/data/ncert_extracted"


#STEP 5: Chunk text
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
    doc_id = doc["file"]
    for i, chunk in enumerate(chunk_text(doc["text"])):
        all_chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{Path(doc_id).stem}_chunk_{i}",
            "text": chunk
        })
all_chunks = split_documents(docs)
st.text(f"Total chunks: {len(all_chunks)}")

#embedding 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 5  # number of chunks to retrieve

if 'all_chunks' not in st.session_state:
    st.warning("No chunks loaded! Please load NCERT content first.")
else:
    all_chunks = st.session_state['all_chunks']

    # Create embeddings only once
    @st.cache_resource(show_spinner=True)
    def build_index(chunks):
        st.text("Creating embeddings. This may take a few minutes...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        texts = [c["text"] for c in chunks]
        embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in chunks]
        st.text(f"FAISS index built with {len(chunks)} chunks.")
        return embed_model, index, metadata

    embed_model, index, metadata = build_index(all_chunks)

    # ---------------------------- STEP 7: Load generator ----------------------------
    @st.cache_resource(show_spinner=True)
    def load_generator():
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
        gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
        if device == 0:
            gen_model = gen_model.to("cuda")
        generator = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer, device=device)
        return generator

    generator = load_generator()

    # ---------------------------- STEP 8: RAG FUNCTIONS ----------------------------
    def retrieve(query, top_k=TOP_K):
        q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = index.search(q_emb, top_k)
        return [metadata[idx] for idx in I[0]]

    def build_prompt(retrieved_chunks, question, max_context_chars=3000):
        ctx_parts = []
        total = 0
        for r in retrieved_chunks:
            t = r["text"].strip()
            if not t:
                continue
            remaining = max_context_chars - total
            if remaining <= 0:
                break
            if len(t) > remaining:
                t = t[:remaining]
            ctx_parts.append(f"Source ({r['doc_id']} / {r['chunk_id']}):\n{t}\n")
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
            return {"answer": "No relevant documents found.", "sources": []}
        prompt = build_prompt(retrieved, query)
        out = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
        sources = [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"]} for r in retrieved]
        return {"answer": out.strip(), "sources": sources}

    # ---------------------------- STEP 9: STREAMLIT UI ----------------------------
    query = st.text_input("Ask a question about NCERT content:")
    if query:
        with st.spinner("Generating answer..."):
            result = generate_answer(query)
            st.write("**Answer:**", result["answer"])
            st.write("**Sources:**")
            for src in result["sources"]:
                st.write(f"{src['doc_id']} / {src['chunk_id']}")
