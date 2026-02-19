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

# ==========================================================
# CONFIGURATION
# ==========================================================
FILE_ID = "1toFD-1u6BSpdDU-cop12nne2ysPgPHM0"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"
TOP_K = 4
BATCH_SIZE = 64

# ==========================================================
# PAGE SETUP
# ==========================================================
st.set_page_config(page_title="NCERT AI Tutor", layout="wide")
st.title("📘 NCERT AI Tutor")
st.caption("Ask questions from NCERT textbooks")

# ==========================================================
# DOWNLOAD + EXTRACT (cached)
# ==========================================================
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
        st.error("ZIP file is invalid.")
        st.stop()

    if not os.path.exists(EXTRACT_DIR):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

    return EXTRACT_DIR

data_path = download_and_extract()

# ==========================================================
# LOAD PDF TEXT
# ==========================================================
@st.cache_resource
def load_documents(folder):
    documents = []

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
                        documents.append({
                            "doc_id": file,
                            "text": text
                        })
                except:
                    continue

    return documents

documents = load_documents(data_path)
st.success(f"Loaded {len(documents)} PDF files")

# ==========================================================
# CHUNK DOCUMENTS
# ==========================================================
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
st.success(f"Created {len(all_chunks)} text chunks")

# ==========================================================
# BUILD VECTOR INDEX (BATCHED + PROGRESS BAR)
# ==========================================================
@st.cache_resource(show_spinner=False)
def build_faiss_index(chunks):

    embed_model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    all_embeddings = []

    progress_bar = st.progress(0)
    total = len(texts)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        emb = embed_model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        all_embeddings.append(emb)
        progress_bar.progress(min((i+BATCH_SIZE)/total, 1.0))

    embeddings = np.vstack(all_embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    metadata = chunks

    progress_bar.empty()

    return embed_model, index, metadata


st.info("Building vector index (first run may take a few minutes)...")
embed_model, index, metadata = build_faiss_index(all_chunks)
st.success("Vector index ready")

# ==========================================================
# LOAD GENERATION MODEL
# ==========================================================
@st.cache_resource
def load_generator():
    device = -1  # Force CPU for Streamlit Cloud

    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return generator

generator = load_generator()

# ==========================================================
# RETRIEVAL
# ==========================================================
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query]).astype("float32")
    D, I = index.search(q_emb, top_k)
    return [metadata[i] for i in I[0]]

# ==========================================================
# PROMPT BUILDER
# ==========================================================
def build_prompt(context_chunks, question):

    context = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
You are an AI tutor specializing in NCERT textbooks.
Answer clearly and concisely using the context provided.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt

# ==========================================================
# GENERATE ANSWER
# ==========================================================
def generate_answer(query):

    retrieved = retrieve(query)

    if not retrieved:
        return "No relevant information found.", []

    prompt = build_prompt(retrieved, query)

    output = generator(
        prompt,
        max_length=256,
        do_sample=False
    )[0]["generated_text"]

    sources = [
        f"{r['doc_id']} ({r['chunk_id']})"
        for r in retrieved
    ]

    return output.strip(), sources

# ==========================================================
# USER INTERFACE
# ==========================================================
query = st.text_input("Ask a question from NCERT:")

if query:
    with st.spinner("Generating answer..."):
        answer, sources = generate_answer(query)

    st.markdown("### 📖 Answer")
    st.write(answer)

    st.markdown("### 📚 Sources")
    for s in sources:
        st.write("-", s)
