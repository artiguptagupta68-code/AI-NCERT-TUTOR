import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------------------------
# CONFIG
# --------------------------------------
GEN_MODEL_NAME = "google/flan-t5-large"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PDF_FOLDER = "ncert_extracted"   # folder containing extracted NCERT PDFs

st.title("📘 NCERT AI Tutor (LangChain RAG)")

# --------------------------------------
# STEP 1 — Load NCERT PDFs
# --------------------------------------
@st.cache_resource
def load_documents():
    docs = []
    for root, dirs, files in os.walk(PDF_FOLDER):
        for file in files:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(root, file))
                docs.extend(loader.load())
    return docs

st.write("📄 Loading NCERT PDFs...")
documents = load_documents()
st.success(f"Loaded {len(documents)} documents.")

# --------------------------------------
# STEP 2 — Split into chunks
# --------------------------------------
@st.cache_resource
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

st.write("✂️ Splitting documents...")
chunks = split_docs(documents)
st.success(f"Created {len(chunks)} text chunks.")

# --------------------------------------
# STEP 3 — Build FAISS index
# --------------------------------------
@st.cache_resource
def build_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL_NAME)
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

st.write("📌 Building vector index...")
vectordb = build_faiss_index(chunks)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
st.success("Vector search ready!")

# --------------------------------------
# STEP 4 — Load Generator Model
# --------------------------------------
@st.cache_resource
def load_generator_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
    )
    return HuggingFacePipeline(pipeline=pipe)

st.write("⚙️ Loading AI generator model...")
llm = load_generator_pipeline()
st.success("Generation model ready!")

# --------------------------------------
# STEP 5 — Build RAG QA Chain
# --------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# --------------------------------------
# STEP 6 — Streamlit UI
# --------------------------------------
query = st.text_input("Ask a question from NCERT:")

if query:
    with st.spinner("Generating answer..."):
        response = qa_chain({"query": query})

        st.write("### ✅ Answer:")
        st.write(response["result"])

        st.write("### 📚 Sources:")
        for src in response["source_documents"]:
            st.write(src.metadata.get("source", "Unknown Source"))
