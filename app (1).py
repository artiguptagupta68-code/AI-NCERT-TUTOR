!pip install streamlit
import streamlit as st
import glob
import os
import fitz # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.fake import FakeListLLM
from langchain_community.vectorstores import FAISS

# --- Configuration ---
# NOTE: Ensure your NCERT PDFs are in a directory named 'ncert_pdfs'
# in the same location as this app.py file.
PDF_DIR = "ncert_pdfs"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# Function to load and index documents (runs only once due to caching)
@st.cache_resource
def setup_rag_pipeline():
    """
    Performs data loading, chunking, embedding, and indexing.
    This function is cached to run only once.
    """

    # 1. Load Documents
    @st.cache_data
    def load_all_pdfs_with_pymupdf(directory):
        docs = []
        if not os.path.exists(directory):
             st.error(f"Directory not found: {directory}. Please ensure your PDFs are there.")
             return []

        for filepath in glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True):
            filename = os.path.basename(filepath)
            try:
                doc = fitz.open(filepath)
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text:
                        docs.append({
                            "source_file": filename,
                            "page": page_num + 1,
                            "content": text
                        })
                doc.close()
            except Exception as e:
                st.warning(f"--- ⚠️ PyMuPDF Error reading {filename}: {e}. Skipping. ---")

        st.success(f"Total pages successfully loaded: {len(docs)}")
        return [Document(page_content=d["content"], metadata={"source_file": d["source_file"], "page": d["page"]}) for d in docs]

    documents = load_all_pdfs_with_pymupdf(PDF_DIR)

    if not documents:
        st.stop()

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    splits = text_splitter.split_documents(documents)
    st.info(f"Total chunks created: {len(splits)}")

    # 3. Keyless Embeddings (HuggingFace)
    with st.spinner("Initializing Local Embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # 4. Keyless Vector Store (FAISS)
    with st.spinner("Creating Local FAISS Index..."):
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # 5. Keyless LLM (Dummy Model)
    # This LLM confirms retrieval works without generating a real answer.
    dummy_llm = FakeListLLM(
        responses=[
            "**[DUMMY LLM RESPONSE - NO EXTERNAL API USED]**\n\nI have successfully received your question and the relevant context from the local vector store (FAISS). This confirms the data retrieval step (RAG) is working perfectly without any keys.\n\nTo get a real answer, you must replace this `dummy_llm` with a functional Large Language Model.\n\n**Received Context:**\n{context}"
        ]
    )

    # 6. RAG Chain Definition
    SYSTEM_PROMPT = """
    You are a helpful NCERT Tutor for classes 6–10.
    Explain concepts in:
    - Simple language first
    - Step-by-step reasoning
    - Include easy examples
    - Mention book + page numbers
    Only use the context provided.
    If the answer is not in the context, say you don't find it in the given NCERT pages.
    """

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt_template = (
        SystemMessage(content=SYSTEM_PROMPT) +
        HumanMessage(content="Question: {question}\n\nContext:\n{context}")
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | dummy_llm
        | StrOutputParser()
    )

    return rag_chain

# --- Streamlit Application ---
st.title("📚 Keyless NCERT RAG Tutor (Local Demo)")
st.caption("This app runs entirely without OpenAI or Pinecone keys. It uses HuggingFace embeddings and FAISS.")

# Setup the RAG chain
rag_chain = setup_rag_pipeline()

# User Input
question = st.text_input(
    "Ask a question about your NCERT content:",
    placeholder="e.g., What are the four stages of the French Revolution?"
)

if st.button("Get Answer"):
    if question:
        with st.spinner("Retrieving context and running dummy LLM..."):
            try:
                # Invoke the RAG chain
                response = rag_chain.invoke(question)
                st.markdown("---")
                st.subheader("Answer (Retrieval Confirmed)")
                st.write(response)

            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
    else:
        st.warning("Please enter a question to get an answer.")
