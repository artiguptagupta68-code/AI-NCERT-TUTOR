import streamlit as st
import zipfile
import os
import fitz  # PyMuPDF
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="AI NCERT Tutor", layout="wide")
st.title("📚 AI NCERT Tutor")

# ---------------- Constants ----------------
ZIP_FILE = "ncrt subject.zip"  # Path to your ZIP file on server
EXTRACT_FOLDER = "ncert_pdfs"

# ---------------- Extract ZIP ----------------
os.makedirs(EXTRACT_FOLDER, exist_ok=True)
with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_FOLDER)

# ---------------- Load PDFs & TXT ----------------
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
                    print(f"Failed to read PDF {path}: {e}")
            elif file.lower().endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                except Exception as e:
                    print(f"Failed to read TXT {path}: {e}")
    return texts

texts = load_documents(EXTRACT_FOLDER)
if not texts:
    st.error("No documents found! Make sure the ZIP file has PDFs or TXT files.")
    st.stop()

# ---------------- Split documents ----------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_text(" ".join(texts))

# ---------------- Embeddings & FAISS ----------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(chunks)

embedding_dim = len(embeddings[0])
index = FAISS(embedding_model.embed_query, index=None)
index.index = FAISS.build_faiss_index(np.array(embeddings).astype("float32"))
index.chunks = chunks

# ---------------- Load CPU LLM ----------------
llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.float32, device_map="cpu")
llm_model.to("cpu")

# ---------------- User Query ----------------
query = st.text_input("Write your question here:")
if query:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.index.search(query_embedding.astype("float32"), k=5)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    # Generate answer
    input_text = f"Answer the question based on the following context:\n{context}\nQuestion: {query}"
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Answer:")
    st.write(answer)
