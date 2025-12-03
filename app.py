import os
import zipfile
import gdown
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import streamlit as st

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncrt subject.zip"
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

# ----------------------------
# STEP 5: Chunk text
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

st.text(f"Total chunks: {len(all_chunks)}")

# ----------------------------
# STEP 6: Create embeddings and FAISS index
# ----------------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
texts = [c["text"] for c in all_chunks]
embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in all_chunks]
st.text("FAISS index built.")

# ----------------------------
# STEP 7: Load generator model
# ----------------------------
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

if device == 0:
    gen_model = gen_model.to("cuda")

generator = pipeline(
    "text2text-generation",
    model=gen_model,
    tokenizer=tokenizer,
    device=device
)

# ----------------------------
# STEP 8: RAG functions
# ----------------------------
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)

    retrieved = []
    for idx in I[0]:
        retrieved.append({
            "doc_id": metadata[idx]["doc_id"],
            "chunk_id": metadata[idx]["chunk_id"],
            "text": metadata[idx]["text"]
        })
    return retrieved


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

        ctx_parts.append(
            f"Source ({r['doc_id']} / {r['chunk_id']}):\n{t}\n"
        )
        total += len(t)

    context = "\n---\n".join(ctx_parts)

    prompt = (
        "You are an AI tutor specialized in NCERT content. "
        "Use the provided context excerpts to answer the question accurately.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER concisely:"
    )
    return prompt


def generate_answer(query):
    retrieved = retrieve(query)

    if not retrieved:
        return {
            "answer": "No relevant NCERT content found.",
            "sources": []
        }

    prompt = build_prompt(retrieved, query)

    output = generator(
        prompt,
        max_length=256,
        do_sample=False
    )[0]["generated_text"]

    sources = []
    for r in retrieved:
        sources.append({
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"]
        })

    return {
        "answer": output.strip(),
        "sources": sources
    }


# ----------------------------
# STEP 9: Streamlit UI
# ----------------------------
st.subheader("Ask NCERT AI Tutor")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        result = generate_answer(query)

        st.write("### **Answer:**")
        st.write(result["answer"])

        st.write("### **Sources used:**")
        for s in result["sources"]:
            st.write(f"- Document {s['doc_id']} | Chunk {s['chunk_id']}")
