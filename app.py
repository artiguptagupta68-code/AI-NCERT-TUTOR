import os
import streamlit as st


from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4


st.title("📘 NCERT AI Tutor (RAG)")

# ---------------------------------------------------------
# LOAD PDF DOCUMENTS
# ---------------------------------------------------------
@st.cache_resource
def load_docs():
    docs = []
    if not os.path.exists(PDF_FOLDER):
        st.error("❌ NCERT folder not found!")
        return docs

    for root, dirs, files in os.walk(PDF_FOLDER):
        for f in files:
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, f))
                docs.extend(loader.load())
    return docs

docs = load_docs()
st.success(f"Loaded {len(docs)} PDF pages")

# ---------------------------------------------------------
# SPLIT INTO CHUNKS
# ---------------------------------------------------------
@st.cache_resource
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(docs)

chunks = split_docs(docs)
st.success(f"Created {len(chunks)} chunks")

# ---------------------------------------------------------
# BUILD FAISS INDEX
# ---------------------------------------------------------
@st.cache_resource
def create_faiss(chunks):
    embedder = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vectordb = FAISS.from_documents(chunks, embedder)
    return vectordb

vectordb = create_faiss(chunks)
retriever = vectordb.as_retriever()

st.success("FAISS index ready")

# ---------------------------------------------------------
# LOAD GENERATOR MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()
st.success("Generator model loaded")

# ---------------------------------------------------------
# BUILD RETRIEVAL QA
# ---------------------------------------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ---------------------------------------------------------
# STREAMLIT UI INPUT
# ---------------------------------------------------------
query = st.text_input("Ask an NCERT question:")

if query:
    with st.spinner("Thinking..."):
        response = qa({"query": query})

        st.write("### ✅ Answer:")
        st.write(response["result"])

        st.write("### 📚 Sources:")
        for src in response["source_documents"]:
            st.write(src.metadata.get("source", "Unknown PDF"))
