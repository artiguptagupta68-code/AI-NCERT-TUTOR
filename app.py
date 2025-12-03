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
import gdown

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"          # ZIP will be downloaded to app folder
EXTRACT_DIR = "ncert_extracted" # folder to extract PDFs
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4

# ----------------------------
# Ensure extraction folder exists
# ----------------------------
os.makedirs(EXTRACT_DIR, exist_ok=True)

# ----------------------------
# Download ZIP using gdown
# ----------------------------
if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) == 0:
    print("Downloading ZIP from Google Drive using gdown...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", ZIP_PATH, quiet=False)
    print("Download complete:", ZIP_PATH)

# ----------------------------
# Verify ZIP and extract
# ----------------------------
if zipfile.is_zipfile(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print("ZIP extracted to:", EXTRACT_DIR)
else:
    raise Exception(f"{ZIP_PATH} is not a valid ZIP. Check the Drive link or permissions.")

# ----------------------------
# STEP 2: Read PDFs
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
                print("Failed to read PDF:", path, e)

print(f"Loaded {len(documents)} PDF documents.")

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
    for i, chunk in
