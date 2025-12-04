import os
import zipfile
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import gdown

# ---------------- CONFIG ----------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "/tmp/ncrt.zip"
EXTRACT_DIR = "/tmp/ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "facebook/opt-125m"
TOP_K = 4

st.set_page_config(page_title="📚 AI NCERT Tutor", layout="wide")
st.title("📚 AI NCERT Tutor")

# ---------------- Download ZIP from Drive ----------------
if not os.path.exists(ZIP_PATH):
    download_url = f"https://drive.google.com/uc?id={FILE_ID}"
    st.info("Downloading NCERT ZIP from Google Drive...")
    try:
        gdown.download(download_url, ZIP_PATH, quiet=False)
    except Exception as e:
        st.error(
            f"Failed to download ZIP from Google Drive: {e}\n\n"
            "Please check:\n"
            "1. The file is shared with 'Anyone with the link can view'\n"
            "2. The FILE_ID is correct\n"
            "3. You can manually download the file and place it in /tmp/ncert.zip"
        )
        st.stop()

# ---------------- Extract ZIP ----------------
if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Extract nested ZIPs
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for file in files:
            if file.endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                nested_extract_path = os.path.join(root, file.replace(".zip", ""))
                os.makedirs(nested_extract_path, exist_ok=True)
                with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(nested_extract_path)

# ---------------- Load PDFs ----------------
st.info("Loading PDF documents...")
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

if len(documents) == 0:
    st.error("No PDF files found in ZIP!")
    st.stop()

st.success(f"Loaded {len(documents)} PDF documents.")

# ---------------- Chunk Text ----------------
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
    for chunk in chunk_text(doc["text"]):
        all_chunks.append(chunk)

st.success(f"Total chunks created: {len(all_chunks)}")

# ---------------- Embeddings & FAISS ----------------
st.info("Building FAISS index...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = embed_model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True).astype("float32")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)
st.success("FAISS index built.")

# ---------------- Load LLM ----------------
device = 0 if torch.cuda.is_available() else -1
st.info("Loading generator model...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
if device == 0:
    llm_model = llm_model.to("cuda")
generator = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer, device=device)
st.success("Generator model loaded.")

# ---------------- RAG Functions ----------------
def retrieve(query, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    return [all_chunks[i] for i in I[0]]

def build_prompt(retrieved_chunks, question):
    context = "\n---\n".join([c.strip() for c in retrieved_chunks])
    prompt = (
        "You are an AI tutor specialized in NCERT content. Use the provided context to answer the question accurately.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and clearly:"
    )
    return prompt

def generate_answer(query):
    retrieved = retrieve(query)
    if not retrieved:
        return "No relevant documents found."
    prompt = build_prompt(retrieved, query)
    out = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return out.strip()

# ---------------- Streamlit UI ----------------
query = st.text_input("Ask a question about NCERT content:")
if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)
        st.subheader("Answer:")
        st.write(answer)
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
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "/tmp/ncrt.zip"
EXTRACT_FOLDER = "/tmp/ncert_extracted"

# ---------------- Download ZIP from Drive ----------------
if not os.path.exists(ZIP_PATH):
    download_url = f"https://drive.google.com/uc?id={FILE_ID}"
    st.info("Downloading NCERT ZIP from Google Drive...")
    try:
        gdown.download(download_url, ZIP_PATH, quiet=False)
    except Exception as e:
        st.error(
            f"Failed to download ZIP from Google Drive: {e}\n\n"
            "Please check:\n"
            "1. The file is shared with 'Anyone with the link can view'\n"
            "2. The FILE_ID is correct\n"
            "3. You can manually download the file and place it in /tmp/ncert.zip"
        )
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
