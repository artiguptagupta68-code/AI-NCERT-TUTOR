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

ZIP_PATH = "/tmp/ncert.zip"
EXTRACT_DIR = "/tmp/ncert_extracted"

# ---------------- OpenAI API ----------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------- Download ZIP From Google Drive ----------------
@st.cache_resource
def download_from_drive():
    url = f"http
