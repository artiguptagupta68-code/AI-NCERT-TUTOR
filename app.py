import streamlit as st
import os
import gdown
import zipfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ----------------------------
# CONFIG
# ----------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncert.zip"
EXTRACT_DIR = "ncert_extracted"

# Ensure clean environment
if not os.path.exists(ZIP_PATH):
    st.write("Downloading NCERT ZIP from Google Drive…")

    # FIXED: strong download link
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)

# Validate ZIP
if not zipfile.is_zipfile(ZIP_PATH):
    raise Exception(f"{ZIP_PATH} is not a valid ZIP file. "
                    f"Check if the Google Drive file is a real ZIP and shared publicly.")

# Extract safely
if not os.path.exists(EXTRACT_DIR):
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

st.success("NCERT ZIP extracted successfully!")

# ----------------------------
# STEP 2: Load PDFs
# ----------------------------
st.write("Reading PDFs…")

all_docs = []
for root, _, files in os.walk(EXTRACT_DIR):
    for f in files:
        if f.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, f)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            all_docs.extend(pages)

st.success(f"Loaded {len(all_docs)} PDF pages")

# ----------------------------
# STEP 3: Split into chunks
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150
)
chunks = splitter.split_documents(all_docs)
st.write(f"Created {len(chunks)} text chunks")

# ----------------------------
# STEP 4: Vector store
# ----------------------------
embd = OpenAIEmbeddings()
vectordb = FAISS.from_documents(chunks, embd)

# ----------------------------
# STEP 5: Chatbot UI
# ----------------------------
st.title("📘 NCERT Tutor AI")

query = st.text_input("Ask a question from NCERT:")

if query:
    llm = ChatOpenAI(model="gpt-4o-mini")

    results = vectordb.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
    You are an NCERT expert tutor.
    Use ONLY the context below to answer.

    Context:
    {context}

    Question: {query}
    """

    answer = llm.invoke(prompt)
    st.markdown("### ✅ Answer")
    st.write(answer.content)
