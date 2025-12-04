import os
import zipfile
import streamlit as st
from pathlib import Path
import numpy as np
import gdown
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📚 AI NCERT Tutor", layout="wide")
st.title("📘 AI NCERT Tutor (Google Drive → RAG)")

# Put your file ID here
GOOGLE_FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"

ZIP_PATH = "/tmp/ncrt.zip"
EXTRACT_DIR = "/tmp/ncert_extracted"

# ---------------- OpenAI API ----------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------- Download ZIP From Google Drive ----------------
@st.cache_resource
def download_from_drive():
    url = f"https://drive.google.com/uc?id={GOOGLE_FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)
    return ZIP_PATH


with st.spinner("Downloading NCERT ZIP from Google Drive..."):
    download_from_drive()


# ---------------- Extract ZIP ----------------
@st.cache_resource
def extract_zip():
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR


with st.spinner("Extracting files..."):
    extract_zip()


# ---------------- Load PDFs ----------------
@st.cache_resource
def load_documents(folder):
    texts = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                try:
                    reader = PdfReader(pdf_path)
                    text = "\n".join([p.extract_text() or "" for p in reader.pages])
                    texts.append(text)
                except:
                    pass
    return texts


with st.spinner("Reading PDF documents..."):
    documents = load_documents(EXTRACT_DIR)

st.success(f"Loaded {len(documents)} PDF documents.")


# ---------------- Split Text Into Chunks ----------------
@st.cache_resource
def split_chunks(text_list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    all_chunks = []
    for t in text_list:
        all_chunks.extend(splitter.split_text(t))
    return all_chunks


chunks = split_chunks(documents)
st.success(f"Total chunks created: {len(chunks)}")


# ---------------- Create Embeddings & FAISS ----------------
@st.cache_resource
def build_faiss(chunks_list):
    embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    vectors = embedder.encode(chunks_list, convert_to_numpy=True).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, embedder


with st.spinner("Building FAISS index..."):
    index, embed_model = build_faiss(chunks)

st.success("FAISS index ready.")


# ---------------- Retrieve Relevant Chunks ----------------
def retrieve_context(query):
    q = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q, 5)
    return "\n\n".join([chunks[i] for i in I[0]])


# ---------------- Generate Answer Using OpenAI ----------------
def get_answer(question):
    context = retrieve_context(question)

    prompt = f"""
You are an NCERT expert. Answer the question ONLY using the context below.
Be accurate and clear.

Context:
{context}

Question: {question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )
    return response.choices[0].message["content"]


# ---------------- User Input ----------------
query = st.text_input("Ask your NCERT question:")

if query:
    with st.spinner("Generating answer..."):
        answer = get_answer(query)

    st.subheader("Answer:")
    st.write(answer)
