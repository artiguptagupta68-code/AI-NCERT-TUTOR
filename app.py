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


st.set_page_config(page_title="📚 AI NCERT Tutor", layout="wide")
st.title("📚 AI NCERT Tutor")

# ---------------- Path to your ZIP (already on server) ----------------
ZIP_FILE = "/content/ncrt subject.zip"  # change this to your actual ZIP path
EXTRACT_FOLDER = "/tmp/ncert_extracted"
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

# ---------------- Extract ZIP ----------------
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
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
                except:
                    pass
            elif file.lower().endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                except:
                    pass
    return texts

texts = load_documents(EXTRACT_FOLDER)

if len(texts) == 0:
    st.error("No documents found in ZIP!")
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

faiss_index = {"index": index, "chunks": chunks}

# ---------------- Load CPU-friendly LLM ----------------
llm_model_name = "facebook/opt-125m"  # CPU-friendly

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

# Load model (no device_map)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

# Force CPU
llm_model.to("cpu")
# ---------------- User Query ----------------
query = st.text_input("Write your query here:")

if query:
    # Retrieve top 5 relevant chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding.astype("float32"), k=5)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    # ---------------- Generate Answer ----------------
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
