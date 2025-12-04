import os
import zipfile
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gdown

st.set_page_config(page_title="📚 AI NCERT Tutor", layout="wide")
st.title("📚 AI NCERT Tutor")

# ---------------- Google Drive CONFIG ----------------
FILE_ID = "YOUR_FILE_ID_HERE"  # Replace with your Google Drive file ID
ZIP_PATH = "/tmp/ncert.zip"
EXTRACT_FOLDER = "/tmp/ncert_extracted"

# ---------------- Download ZIP from Drive ----------------
if not os.path.exists(ZIP_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    try:
        gdown.download(url, ZIP_PATH, quiet=False)
    except Exception as e:
        st.error(f"Failed to download ZIP from Google Drive: {e}")
        st.stop()

# ---------------- Extract ZIP ----------------
os.makedirs(EXTRACT_FOLDER, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_FOLDER)

# Extract nested ZIPs
for root, dirs, files in os.walk(EXTRACT_FOLDER):
    for file in files:
        if file.endswith(".zip"):
            nested_zip_path = os.path.join(root, file)
            nested_extract_path = os.path.join(root, file.replace(".zip", ""))
            os.makedirs(nested_extract_path, exist_ok=True)
            with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                zip_ref.extractall(nested_extract_path)

# ---------------- Load PDFs ----------------
def load_documents(folder):
    texts = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            if file.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    if text.strip():
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

texts = load_documents(EXTRACT_FOLDER)
if len(texts) == 0:
    st.warning("No PDF/TXT files found!")
    st.stop()

# ---------------- Split documents ----------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(" ".join(texts))

# ---------------- Embeddings & FAISS ----------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(chunks)

embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)
embedding_matrix = np.array(embeddings).astype("float32")
index.add(embedding_matrix)

# ---------------- User query ----------------
query = st.text_input("Ask a question about NCERT content:")

if query:
    # Retrieve top 5 relevant chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding.astype("float32"), k=5)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    # ---------------- LLM Answer Generation ----------------
    llm_model_name = "facebook/opt-125m"  # CPU-friendly
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
    llm_model.to("cpu")

    input_text = f"Answer the question based on the following context:\n{context}\nQuestion: {query}"
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Answer:")
    st.write(answer)
