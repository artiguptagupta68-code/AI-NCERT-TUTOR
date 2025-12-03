
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
st.title("🤖 NCERT AI Tutor")
#st.text("Downloading and extracting NCERT ZIP from Google Drive...")

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

#st.text("All nested ZIPs extracted.")

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

    # ---------------- Load documents ----------------
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

    texts = load_documents(extract_folder)
   # st.success(f"Loaded {len(texts)} documents.")

    if len(texts) == 0:
        st.warning("No PDF/TXT files found!")
    else:
        # ---------------- Split documents ----------------
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(" ".join(texts))
       # st.success(f"Created {len(chunks)} text chunks.")

        # ---------------- Embeddings & FAISS ----------------
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedding_model.embed_documents(chunks)

        embedding_dim = len(embeddings[0])
        index = faiss.IndexFlatL2(embedding_dim)
        embedding_matrix = np.array(embeddings).astype("float32")
        index.add(embedding_matrix)

        faiss_index = {
            "index": index,
            "chunks": chunks
        }
        #st.success("FAISS index created!")

        # ---------------- User query ----------------
        query = st.text_input("Ask a question about the content:")

        if query:
            # Retrieve top 5 relevant chunks
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode([query], convert_to_numpy=True)
            D, I = index.search(query_embedding.astype("float32"), k=5)
            retrieved_chunks = [chunks[i] for i in I[0]]
            context = "\n\n".join(retrieved_chunks)

            # ---------------- LLM Answer Generation ----------------
            # CPU-friendly model
            llm_model_name = "facebook/opt-125m"  # CPU-friendly

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

            # Load model (no device_map)
            llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

            # Force CPU
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
