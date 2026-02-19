import streamlit as st
import os
import zipfile
import numpy as np
import faiss
import torch
import gdown
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -----------------------------
# CONFIG
# -----------------------------
FILE_ID = "1toFD-1u6BSpdDU-cop12nne2ysPgPHM0"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4
BATCH_SIZE = 64

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="NCERT AI Tutor", layout="wide")
st.title("📘 NCERT AI Tutor")
st.caption("Ask questions from NCERT books using AI")

# -----------------------------
# DOWNLOAD ZIP (only once)
# -----------------------------
@st.cache_resource
def download_and_extract():
    if not os.path.exists(ZIP_PATH):
        with st.spinner("Downloading NCERT data..."):
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                ZIP_PATH,
                quiet=False
            )

    if not zipfile.is_zipfile(ZIP_PATH):
        st.error("Invalid ZIP file.")
        st.stop()

    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

    return EXTRACT_DIR

data_folder = download_and_extract()

# -----------------------------
# LOAD PDF TEXT
# -----------------------------
@st.cache_resource
def load_documents(folder):
    docs = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                path = os.path.join(root, file)
                try:
                    reader = PdfReader(path)
                    text = ""

                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text += t + "\n"

                    if text.strip():
                        docs.append({
                            "doc_id": file,
                            "text": text
                        })
                except:
                    continue

    return docs

documents = load_documents(data_folder)
st.success(f"Loaded {len(documents)} PDF files")

# -----------------------------
# SPLIT INTO CHUNKS
# -----------------------------
@st.cache_resource
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for doc in docs:
        split_texts = splitter.split_text(doc["text"])

        for i, chunk in enumerate(split_texts):
            chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "text": chunk
            })

    return chunks

all_chunks = split_documents(documents)
st.success(f"Created {len(all_chunks)} chunks")

# -----------------------------
# BUILD FAISS INDEX (BATCHED)
# -----------------------------
@st.cache_resource(show_spinner=True)
def build_index(chunks):

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]

    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        emb = embed_model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    metadata = [
        {
            "doc_id": c["doc_id"],
            "chunk_id": c["chunk_id"],
            "text": c["text"]
        }
        for c in chunks
    ]

    return embed_model, index, metadata


embed_model, index, metadata = build_index(all_chunks)
st.success("Vector index ready")

# -----------------------------
# LOAD GENERATOR
# -----------------------------
@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    model.eval()
    return tokenizer, model

tokenizer, model = load_generator()



# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query]).astype("float32")
    D, I = index.search(q_emb, top_k)
    return [metadata[i] for i in I[0]]

# -----------------------------
# PROMPT BUILDER
# -----------------------------
def build_prompt(context_chunks, question):

    context_text = "\n\n".join(
        [f"{c['text']}" for c in context_chunks]
    )

    prompt = f"""
You are an AI tutor specialized in NCERT textbooks.

Use the context below to answer clearly and concisely.

Context:
{context_text}

Question:
{question}

Answer:
"""

    return prompt

# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(query):

    retrieved = retrieve(query)

    if not retrieved:
        return "No relevant information found.", []

    prompt = build_prompt(retrieved, query)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    sources = [
        f"{r['doc_id']} ({r['chunk_id']})"
        for r in retrieved
    ]

    return answer.strip(), sources

# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("Ask your question from NCERT:")

if query:
    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(query)

    st.markdown("### 📖 Answer")
    st.write(answer)

    st.markdown("### 📚 Sources")
    for s in sources:
        st.write("-", s)
