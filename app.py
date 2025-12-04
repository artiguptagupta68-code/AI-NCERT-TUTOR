
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

# ---------------- FIXED BACKEND ZIP PATH ----------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
DIRECT_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

ZIP_PATH = "/tmp/ncrt.zip"

def download_from_drive():
    with st.spinner("Downloading NCERT ZIP from Google Drive..."):
        response = requests.get(DIRECT_URL, stream=True)
        if response.status_code != 200:
            st.error("❌ Failed to download ZIP from Google Drive.")
            st.stop()

        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

st.info("Click the button to load NCERT content.")

if st.button("Load NCERT NCERT ZIP"):
    download_from_drive()

    # ---------------- Extract ZIP ----------------
    extract_folder = "/tmp/ncert_extracted"
    os.makedirs(extract_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Extract nested ZIPs
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            if file.endswith(".zip"):
                nested_path = os.path.join(root, file)
                nested_extract = os.path.join(root, file.replace(".zip", ""))
                os.makedirs(nested_extract, exist_ok=True)
                with zipfile.ZipFile(nested_path, 'r') as z:
                    z.extractall(nested_extract)

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

    texts = load_documents(extract_folder)

    if len(texts) == 0:
        st.error("No PDF/TXT files found in backend ZIP!")
        st.stop()

    # ---------------- Split text ----------------
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(" ".join(texts))

    # ---------------- Embeddings & FAISS ----------------
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedding_model.embed_documents(chunks)

    embed_dim = len(embeddings[0])
    index = faiss.IndexFlatL2(embed_dim)
    embedding_matrix = np.array(embeddings).astype("float32")
    index.add(embedding_matrix)

    st.success("📦 Loaded successfully! You can ask questions now.")

    st.session_state["index"] = index
    st.session_state["chunks"] = chunks

# ---------------- QUESTION ANSWERING ----------------
if "index" in st.session_state:
    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Searching best answer..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_embed = model.encode([query], convert_to_numpy=True)
            D, I = st.session_state["index"].search(q_embed.astype("float32"), k=5)
            retrieved = [st.session_state["chunks"][i] for i in I[0]]
            context = "\n\n".join(retrieved)

        with st.spinner("Generating answer..."):
            llm_name = "facebook/opt-125m"
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            llm = AutoModelForCausalLM.from_pretrained(llm_name)
            llm.to("cpu")

            inp = f"Answer ONLY from this context:\n{context}\n\nQuestion: {query}"
            inputs = tokenizer(inp, return_tensors="pt")

            with torch.no_grad():
                output = llm.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.6,
                    do_sample=True
                )

            answer = tokenizer.decode(output[0], skip_special_tokens=True)

        st.subheader("Answer:")
        st.write(answer)
