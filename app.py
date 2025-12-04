
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

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📚 AI NCERT Tutor", layout="wide")
st.title("📚 AI NCERT Tutor")

# ---------------- Google Drive Direct Download ----------------
import streamlit as st
import zipfile
import os
import shutil

# Local ZIP file path (no Google Drive, no download shown)
BACKEND_ZIP = "/mount/src/ai-ncert-tutor/data/ncrt.zip"
EXTRACT_DIR = "/mount/src/ai-ncert-tutor/extracted"


def load_local_zip():
    if not os.path.exists(BACKEND_ZIP):
        st.error("NCERT ZIP file not found in backend.")
        return False

    try:
        # Clean old directory silently
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)

        os.makedirs(EXTRACT_DIR, exist_ok=True)

        # Extract without displaying anything on UI
        with zipfile.ZipFile(BACKEND_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        return True

    except Exception as e:
        st.error("Failed to load NCERT content.")
        return False


st.title("NCERT Tutor")

if st.button("Load NCERT Content"):
    success = load_local_zip()
    if success:
        st.success("NCERT content loaded successfully.")
    else:
        st.error("Failed to load NCERT ZIP.")


# ---------------- Load ZIP Automatically ----------------
silent_drive_download()

if not os.path.exists(ZIP_PATH):
    st.error("Failed to load NCERT ZIP from Drive.")
    st.stop()

# ---------------- Extract ZIP ----------------
extract_folder = "/tmp/ncert_extracted"
os.makedirs(extract_folder, exist_ok=True)

try:
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
except:
    st.error("Error extracting NCERT ZIP.")
    st.stop()

# Extract nested ZIPs quietly
for root, dirs, files in os.walk(extract_folder):
    for file in files:
        if file.endswith(".zip"):
            nested = os.path.join(root, file)
            nested_dest = os.path.join(root, file.replace(".zip", ""))
            os.makedirs(nested_dest, exist_ok=True)
            try:
                with zipfile.ZipFile(nested, 'r') as z:
                    z.extractall(nested_dest)
            except:
                pass

# ---------------- Load PDF/TXT ----------------
def load_documents(folder):
    all_text = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)

            if file.endswith(".pdf"):
                try:
                    doc = fitz.open(path)
                    text = "".join([p.get_text() for p in doc])
                    all_text.append(text)
                except:
                    pass

            if file.endswith(".txt"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        all_text.append(f.read())
                except:
                    pass
    return all_text

texts = load_documents(extract_folder)

if len(texts) == 0:
    st.error("No PDF or TXT found in the ZIP.")
    st.stop()

# ---------------- Text Split ----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(" ".join(texts))

# ---------------- Embedding + FAISS ----------------
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embed_model.embed_documents(chunks)

dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# ---------------- QnA ----------------
query = st.text_input("Ask your question:")

if query:
    q_model = SentenceTransformer("all-MiniLM-L6-v2")
    q_embed = q_model.encode([query], convert_to_numpy=True)

    D, I = index.search(q_embed.astype("float32"), k=5)
    retrieved = [chunks[i] for i in I[0]]

    context = "\n\n".join(retrieved)

    # LLM
    llm_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name)
    model.to("cpu")

    prompt = f"Use ONLY this context:\n{context}\n\nQuestion: {query}"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.6,
            do_sample=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Answer:")
    st.write(answer)
