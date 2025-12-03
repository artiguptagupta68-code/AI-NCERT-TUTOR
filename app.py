import os
import zipfile
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from fastapi import FastAPI
from pydantic import BaseModel

# ----------------------------
# CONFIG
# ----------------------------
ZIP_DRIVE_PATH = "/content/drive/MyDrive/ncrt_subject.zip"  # update to your Drive path
EXTRACT_DIR = "/content/ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4

app = FastAPI()

import os
import zipfile
import streamlit as st
from io import BytesIO
import zipfile
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path

st.title("NCERT AI Tutor")

# ----------------------------
# Step 1: Upload ZIP
# ----------------------------
uploaded_file = st.file_uploader("Upload NCERT ZIP", type="zip")

if uploaded_file is not None:
    EXTRACT_DIR = "ncert_extracted"
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Extract ZIP
    with zipfile.ZipFile(BytesIO(uploaded_file.read())) as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    st.success("ZIP extracted successfully!")

    # ----------------------------
    # Step 2: Read PDFs
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
                    st.warning(f"Failed to read PDF: {path} ({e})")

    st.write(f"Loaded {len(documents)} PDF documents.")

    # ----------------------------
    # Step 3: Chunk text, embeddings, FAISS
    # ----------------------------
    # (Add your RAG pipeline here: chunking, embeddings, generator)


# ----------------------------
# STEP 3: Chunk text
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

print("Total chunks:", len(all_chunks))

# ----------------------------
# STEP 4: Create embeddings and FAISS index
# ----------------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
texts = [c["text"] for c in all_chunks]
embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
metadata = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"]} for c in all_chunks]
print("FAISS index built.")

# ----------------------------
# STEP 5: Load generator
# ----------------------------
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
if device == 0:
    gen_model = gen_model.to("cuda")
generator = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer, device=device)

# ----------------------------
# STEP 6: RAG functions
# ----------------------------
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

# ----------------------------
# STEP 7: FastAPI endpoint
# ----------------------------
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    return generate_answer(query.question)

@app.get("/")
def root():
    return {"message": "NCERT AI Tutor Backend Running"}
