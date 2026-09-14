import os
import re
import uuid
import hashlib
import io
import zipfile
from typing import List, Dict, Any, Tuple
import streamlit as st
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from groq import Groq

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")
CHROMA_COLLECTION_NAME = "pdf_rag_collection"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
SUPPORTED_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# ==========================================
# STREAMLIT PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="Smart PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .metric-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 10px;
        border-radius: 6px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# RAG CORE FUNCTIONS & CLASSES
# ==========================================
def clean_text(text: str) -> str:
    """Clean and normalize raw extracted PDF text."""
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = text.replace('\u200b', '').replace('\ufeff', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Recursively split text using word boundaries."""
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start_idx = 0
    total_words = len(words)

    while start_idx < total_words:
        end_idx = min(start_idx + chunk_size, total_words)
        chunk_words = words[start_idx:end_idx]
        chunks.append(" ".join(chunk_words))

        if end_idx == total_words:
            break
        step = max(1, chunk_size - chunk_overlap)
        start_idx += step

    return chunks

def process_pdf_bytes(file_bytes: bytes, filename: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> Tuple[List[Dict[str, Any]], int, bool, str]:
    """Extract, clean, and chunk PDF document content safely from bytes buffer."""
    chunks_data = []
    doc = None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        if total_pages == 0:
            return [], 0, False, "The PDF file contains no pages."

        total_extracted_chars = 0
        doc_id = hashlib.md5(f"{filename}_{len(file_bytes)}".encode()).hexdigest()[:12]

        for page_num in range(total_pages):
            page = doc[page_num]
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            total_extracted_chars += len(cleaned)

            if not cleaned:
                continue

            page_chunks = split_text_into_chunks(cleaned, chunk_size, chunk_overlap)
            for c_idx, chunk_text in enumerate(page_chunks):
                chunk_id = f"{doc_id}_p{page_num + 1}_c{c_idx}_{uuid.uuid4().hex[:6]}"
                chunks_data.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "total_pages": total_pages
                    }
                })

        if total_extracted_chars < 50 and total_pages > 0:
            return [], total_pages, True, "This PDF does not contain machine-readable text. It appears to be scanned or image-based. OCR is required."

        return chunks_data, total_pages, False, ""
    except Exception as e:
        return [], 0, False, f"Failed to parse PDF document: {str(e)}"
    finally:
        if doc:
            doc.close()

class HuggingFaceEmbeddingFunction:
    """Wrapper class for HuggingFace Sentence Transformers compatible with ChromaDB."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

class VectorStoreManager:
    """Manager class handling ChromaDB persistent vector storage and similarity searches."""
    def __init__(self):
        self.embedding_fn = HuggingFaceEmbeddingFunction(EMBEDDING_MODEL_NAME)
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
        documents, metadatas, ids = [], [], []
        for chunk in chunks:
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            ids.append(chunk["metadata"]["chunk_id"])

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
        return len(documents)

    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.get_count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.get_count()),
            include=["documents", "metadatas", "distances"]
        )
        formatted_results = []
        if results and results.get("documents") and results["documents"][0]:
            docs, metas, distances = results["documents"][0], results["metadatas"][0], results["distances"][0]
            for doc, meta, dist in zip(docs, metas, distances):
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "score": max(0.0, 1.0 - float(dist))
                })
        return formatted_results

    def get_count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def reset_database(self):
        try:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

class RAGPipeline:
    """RAG execution pipeline using Groq LLM API."""
    def __init__(self, groq_api_key: str, vector_store_manager: VectorStoreManager, model_name: str = DEFAULT_GROQ_MODEL):
        self.client = Groq(api_key=groq_api_key)
        self.vector_store = vector_store_manager
        self.model_name = model_name

    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.35) -> List[Dict[str, Any]]:
        cleaned_query = re.sub(r'\s+', ' ', query).strip()
        if not cleaned_query:
            return []
        raw_results = self.vector_store.similarity_search(cleaned_query, top_k=top_k)
        return [item for item in raw_results if item["score"] >= similarity_threshold]

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        if not retrieved_chunks:
            return "I couldn't find enough information in the uploaded document(s) to answer that question.", []

        context_str, citation_sources = "", []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            text, meta, score = chunk["text"], chunk["metadata"], chunk.get("score", 0.0)
            src_file, page_num = meta.get("source", "Unknown PDF"), meta.get("page", "?")
            context_str += f"--- CONTEXT BLOCK {idx} [File: {src_file} | Page: {page_num}] ---\n{text}\n\n"
            citation_sources.append({
                "source": src_file,
                "page": page_num,
                "score": score,
                "text": text[:150] + "..." if len(text) > 150 else text
            })

        system_prompt = (
            "You are an expert AI document assistant called 'Smart PDF RAG Assistant'.\n"
            "Answer the user's question using ONLY the provided document context.\n\n"
            "STRICT RULES:\n"
            "1. Base your answer strictly on the provided Context passages.\n"
            "2. Do NOT use outside knowledge or extrapolate beyond the text.\n"
            "3. If context is insufficient, state: 'I couldn't find enough information in the uploaded document(s) to answer that question.'\n"
            "4. Cite source filename and page numbers inline (e.g., [doc.pdf - Page 4])."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": f"DOCUMENT CONTEXT:\n{context_str}\nUSER QUESTION: {query}\n\nANSWER:"})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip(), citation_sources
        except Exception as e:
            return f"❌ Groq API Error: {str(e)}", []

# ==========================================
# SESSION STATE & HELPER FUNCTIONS
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "processed_file_hashes" not in st.session_state:
    st.session_state.processed_file_hashes = set()

@st.cache_resource(show_spinner=False)
def get_vector_store():
    return VectorStoreManager()

def resolve_api_key(user_key: str = "") -> str:
    if user_key.strip():
        return user_key.strip()
    try:
        if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

def create_project_zip() -> bytes:
    """Generate a downloadable ZIP file containing app.py and requirements.txt dynamically."""
    req_content = (
        "streamlit>=1.32.0\n"
        "pymupdf>=1.23.22\n"
        "sentence-transformers>=2.5.1\n"
        "chromadb>=0.4.24\n"
        "groq>=0.4.2\n"
        "python-dotenv>=1.0.1\n"
    )
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("requirements.txt", req_content)
        if os.path.exists(__file__):
            zip_file.write(__file__, arcname="app.py")
            
    return zip_buffer.getvalue()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("⚙️ RAG Settings")
    
    # Download Source Code Option
    st.download_button(
        label="📦 Download Source Files (ZIP)",
        data=create_project_zip(),
        file_name="pdf_rag_app.zip",
        mime="application/zip",
        use_container_width=True
    )
    st.divider()

    detected_key = resolve_api_key()
    if not detected_key:
        st.warning("⚠️ Groq API Key required.")
        user_key = st.text_input("Enter Groq API Key:", type="password")
        active_api_key = resolve_api_key(user_key)
    else:
        st.success("🔑 API Key Loaded")
        active_api_key = detected_key

    st.divider()
    selected_model = st.selectbox("Groq LLM Model", options=SUPPORTED_GROQ_MODELS, index=0)
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=15, value=5)
    similarity_threshold = st.slider("Min Similarity Score", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    
    st.divider()
    chunk_size = st.number_input("Chunk Size (words)", min_value=100, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (words)", min_value=0, max_value=500, value=150, step=20)

    st.divider()
    if st.button("🗑️ Reset Application & DB", use_container_width=True):
        v_store = get_vector_store()
        v_store.reset_database()
        st.session_state.messages = []
        st.session_state.indexed_files = {}
        st.session_state.total_chunks = 0
        st.session_state.processed_file_hashes = set()
        st.success("Database cleared!")
        st.rerun()

# ==========================================
# MAIN INTERFACE
# ==========================================
st.title("📚 Smart PDF RAG Assistant")
st.caption("Upload PDFs and ask questions using local open-source embeddings + Groq LLM inference.")

uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
v_store = get_vector_store()

if uploaded_files:
    files_to_process = [f for f in uploaded_files if f"{f.name}_{f.size}" not in st.session_state.processed_file_hashes]
    
    if files_to_process and st.button("⚙️ Process Uploaded Documents", type="primary", use_container_width=True):
        all_new_chunks = []
        with st.status("Processing PDFs...", expanded=True) as status:
            for file_obj in files_to_process:
                status.write(f"📖 Extracting `{file_obj.name}`...")
                chunks, total_pages, is_scanned, err_msg = process_pdf_bytes(
                    file_bytes=file_obj.read(),
                    filename=file_obj.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                if is_scanned:
                    st.error(f"⚠️ `{file_obj.name}`: {err_msg}")
                    continue
                elif err_msg:
                    st.error(f"❌ `{file_obj.name}` error: {err_msg}")
                    continue
                
                all_new_chunks.extend(chunks)
                st.session_state.indexed_files[file_obj.name] = total_pages
                st.session_state.processed_file_hashes.add(f"{file_obj.name}_{file_obj.size}")
                status.write(f"✅ Indexed `{file_obj.name}` ({total_pages} pages, {len(chunks)} chunks)")

            if all_new_chunks:
                status.write("🧠 Generating embeddings...")
                added_count = v_store.add_documents(all_new_chunks)
                st.session_state.total_chunks += added_count
                status.update(label="🎉 Processing Complete!", state="complete", expanded=False)
                st.toast(f"Added {added_count} chunks to DB!", icon="✅")

# Stats Summary Header
if st.session_state.indexed_files:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><b>Documents</b><h3>{len(st.session_state.indexed_files)}</h3></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><b>Pages</b><h3>{sum(st.session_state.indexed_files.values())}</h3></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><b>Chunks</b><h3>{st.session_state.total_chunks}</h3></div>", unsafe_allow_html=True)

st.divider()

# Chat History Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 Referenced Context Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {idx}:** `{src['source']}` — **Page {src['page']}** *(Relevance: {src['score']:.2%})*")
                    st.caption(f"_{src['text']}_")

# User Query Handler
user_query = st.chat_input("Ask a question about your uploaded PDFs...")

if user_query:
    if not active_api_key:
        st.error("❌ Groq API Key is missing.")
        st.stop()
    if v_store.get_count() == 0:
        st.warning("⚠️ Upload and process at least one PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            pipeline = RAGPipeline(active_api_key, v_store, selected_model)
            retrieved_chunks = pipeline.retrieve(user_query, top_k, similarity_threshold)
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            
            answer, sources = pipeline.generate_answer(user_query, retrieved_chunks, history)
            st.markdown(answer)
            
            if sources:
                with st.expander("🔍 Referenced Context Sources"):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**Source {idx}:** `{src['source']}` — **Page {src['page']}** *(Relevance: {src['score']:.2%})*")
                        st.caption(f"_{src['text']}_")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
