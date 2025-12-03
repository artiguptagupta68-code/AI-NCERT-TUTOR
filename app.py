import os
import zipfile
import gdown
import faiss
from langchain_community.embeddings import SentenceTransformerEmbeddings
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# ----------------------------
# STEP 0: Streamlit page setup
# ----------------------------
st.set_page_config(page_title="NCERT AI Tutor", layout="wide")
st.title("NCERT AI Tutor")

# ----------------------------
# STEP 1: Download NCERT ZIP from Google Drive
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"  # your file ID
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

st.write("Downloading NCERT ZIP from Google Drive...")
url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
gdown.download(url, ZIP_PATH, quiet=False)

# Verify ZIP
if not zipfile.is_zipfile(ZIP_PATH):
    st.error(f"{ZIP_PATH} is not a valid ZIP file. Check Google Drive link or permissions.")
    st.stop()

# Extract ZIP
os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
st.success("ZIP extracted successfully!")

# ----------------------------
# STEP 2: List PDFs in folder
# ----------------------------
pdf_files = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(root, file))

if not pdf_files:
    st.warning("No PDF files found in the extracted folder.")
    st.stop()

st.write(f"Found {len(pdf_files)} PDF files.")

# ----------------------------
# STEP 3: Load PDFs and split into chunks
# ----------------------------
st.write("Loading PDFs and splitting into chunks...")
all_chunks = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    all_chunks.extend(pages)

st.write(f"Total chunks: {len(all_chunks)}")

# ----------------------------
# STEP 4: Create embeddings and FAISS vector store
# ----------------------------
st.write("Creating embeddings and FAISS vector store...")
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(all_chunks, embedding_model)
st.success("Vector store created successfully!")

# ----------------------------
# STEP 5: Query interface
# ----------------------------
st.header("Ask NCERT AI Tutor")
query = st.text_input("Enter your question:")

if query:
    docs = vectordb.similarity_search(query, k=3)
    st.write("Top results:")
    for i, doc in enumerate(docs, start=1):
        st.write(f"**Result {i}:**")
        st.write(doc.page_content)
