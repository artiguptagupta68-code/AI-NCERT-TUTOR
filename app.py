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

st.set_page_config(page_title="📚 AI NCERT Tutor", layout="wide")
st.title("📚 AI NCERT Tutor")

# ---------------- Upload ZIP ----------------
uploaded_file = st.file_uploader("Upload NCERT ZIP file", type="zip")

if uploaded_file:
    zip_path = os.path.join("/tmp", uploaded_file.name)
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ---------------- Extract ZIP ---------
