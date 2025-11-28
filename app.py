import os
import glob
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Setup for Local (Key-Free) RAG ---
# 1. Install necessary libraries (replace pinecone and openai dependencies)
!pip -q install langchain-community langchain-text-splitters pypdf faiss-cpu sentence-transformers transformers torch accelerate

# 2. Re-run PDF extraction (assuming original steps 1 & 2 were successful)
# This directory should contain the extracted PDFs:
PDF_DIR = "/content/ncert_pdfs"
pdf_files = glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True)

# 3. Load Documents (same as original logic)
print(f"Loading {len(pdf_files)} PDF documents...")
docs = []
for file_path in pdf_files:
    loader = PyMuPDFLoader(file_path)
    loaded_pages = loader.load()
    # Add source and page to metadata
    for page in loaded_pages:
        page.metadata['source_file'] = os.path.basename(file_path)
    docs.extend(loaded_pages)

# 4. Split Documents into Chunks (same as original logic)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
splits = text_splitter.split_documents(docs)
print(f"\nTotal chunks created: {len(splits)}")

# --- Key-Free Components ---

# 5. Initialize Local Embeddings (Replaces OpenAIEmbeddings)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Note: 'all-MiniLM-L6-v2' is a small, fast 384-dimensional model.
# This does NOT require an API key and runs locally.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("\nInitialized local HuggingFace Embeddings (Key-Free)")

# 6. Initialize Local Vector Store (Replaces Pinecone)
from langchain_community.vectorstores import FAISS

# This step uses the local embeddings to create a FAISS index in memory.
db = FAISS.from_documents(splits, embeddings)
print("Created local FAISS vector store (Key-Free)")

# 7. Placeholder for Local LLM (Replaces ChatOpenAI)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Note: You need a model to run locally. This is a placeholder for a small
# open-source model like 'TinyLlama-1.1B-Chat-v1.0'.
# For a practical result, you might need a stronger model like Llama 2 7B,
# which requires significant resources (RAM/GPU) or a more complex setup.
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Downloading the model for local inference (this can take a few minutes)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Create a local text generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512
)

def local_llm_invoke(messages):
    """Simple wrapper to invoke the local HuggingFace pipeline."""
    # Convert chat messages (system/user role) to a single prompt string
    prompt_template = ""
    for msg in messages:
        if msg["role"] == "system":
            # For local models, integrate system prompt into the user query
            prompt_template += f"SYSTEM: {msg['content']}\n\n"
        elif msg["role"] == "user":
            prompt_template += f"USER: {msg['content']}\n"
    
    # Use the pipeline to generate the response
    response = llm_pipeline(prompt_template)[0]['generated_text']
    
    # Clean up the output to only return the model's answer
    # This is a simplification; a full chat template is usually better.
    # We find the first instance of 'USER' or 'SYSTEM' after generation and crop.
    try:
        start_index = response.index(prompt_template) + len(prompt_template)
        return response[start_index:].strip()
    except:
        return response.strip()

print("Initialized local LLM pipeline (Key-Free)")

# 8. Define RAG functions (Replaces the original semantic_search and answer_doubt)

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

def build_context(matches):
    """Builds a formatted context string from FAISS search results."""
    ctx = []
    for doc in matches:
        metadata = doc.metadata
        page = metadata.get("page", "N/A")
        source_file = metadata.get("source_file", "Unknown Source")
        
        ctx.append(
            f"--- Source: {source_file}, Page: {page} ---\n"
            f"{doc.page_content}\n"
        )
    return "\n\n---\n\n".join(ctx)


def answer_doubt(question):
    # 1) Retrieve relevant chunks from local FAISS store
    # FAISS uses a similarity search method (e.g., Euclidean, but HuggingFace embeddings usually use Cosine similarity).
    matches = db.similarity_search(question, k=6)

    # 2) Build context string from those chunks
    context = build_context(matches)

    # 3) Ask the local LLM with system + user messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}"
        }
    ]

    # Use the local LLM wrapper instead of llm.invoke
    reply = local_llm_invoke(messages)

    # 4) Collect sources (optional, for verification)
    sources = []
    for doc in matches:
        metadata = doc.metadata
        sources.append({
            "source_file": metadata.get("source_file", ""),
            "page": metadata.get("page", 0),
            # FAISS does not return a 'score' easily, so we omit it or set to N/A
            "score": "N/A (Local FAISS)" 
        })

    return reply, sources

# --- Example Usage (Key-Free) ---
print("\n--- Testing Key-Free RAG System ---")

# Example question: 'What were the causes of the French Revolution?'
question = "What were the causes of the French Revolution?"

# The function call is the same, but the implementation is local:
answer, sources_used = answer_doubt(question)

print(f"\n✅ Question: {question}")
print(f"\n--- Tutor's Answer (from Local LLM) ---\n{answer}")
print("\n--- Sources Used (from Local FAISS) ---")
for src in sources_used:
    print(f"File: {src['source_file']}, Page: {src['page']}")
