import os
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# CONFIG
FILE_ID = "1gdiCsGOeIyaDlJ--9qon8VTya3dbjr6G"
ZIP_PATH = "ncrt.zip"
EXTRACT_DIR = "ncert_extracted"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-base"
TOP_K = 4 

st.title("📘 NCERT AI Tutor (LangChain RAG)")

# ----------------------------
# STEP 1: Load PDFs
# ----------------------------
@st.cache_resource
def load_docs():
    docs = []
    if not os.path.exists(EXTRACT_DIR):
        st.error(f"NCERT folder '{EXTRACT_DIR}' not found!")
        return docs

    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, f))
                docs.extend(loader.load())
    return docs

docs = load_docs()
st.success(f"Loaded {len(docs)} PDF pages")

# ----------------------------
# STEP 2: Split documents
# ----------------------------
@st.cache_resource
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

chunks = split_docs(docs)
st.success(f"Created {len(chunks)} text chunks")

# ----------------------------
# STEP 3: Build FAISS index
# ----------------------------
@st.cache_resource
def build_faiss(chunks):
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = FAISS.from_documents(chunks, embedder)
    return vectordb

vectordb = build_faiss(chunks)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
st.success("FAISS vector index ready")

# ----------------------------
# STEP 4: Load Generator Model
# ----------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()
st.success("Generator model loaded")

# ----------------------------
# STEP 5: Build LangChain RetrievalQA
# ----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ----------------------------
# STEP 6: Streamlit UI
# ----------------------------
query = st.text_input("Ask a question from NCERT content:")

if query:
    with st.spinner("Generating answer..."):
        response = qa_chain({"query": query})
        st.subheader("✅ Answer:")
        st.write(response["result"])

        st.subheader("📚 Sources:")
        for src in response["source_documents"]:
            st.write(src.metadata.get("source", "Unknown PDF"))
