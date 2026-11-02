import os
import hashlib 
# We use hashlib in Python to generate a unique hash of data for checking file integrity, detecting duplicates, and avoiding unnecessary processing in RAG projects.
import html
import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_SESSION_STATES, CUSTOM_CSS, MAX_HISTORY_MESSAGES
from utils.ui_helpers import chat_bubble_html
from core.document_loader import process_multiple_documents, get_embedding_model, EmptyPDFError
from core.retriever import (
    collection_exists, 
    load_existing_vector_store, 
    build_new_vector_store, 
    create_hybrid_retriever
)
from core.rag_engine import get_llm, RAGChatbot

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
for key, val in DEFAULT_SESSION_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "batch_id" not in st.session_state:
    st.session_state.batch_id = None

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def friendly_error(e: Exception) -> str:
    text = str(e)
    if any(s in text for s in ("403", "401", "Unauthorized", "invalid_api_key")):
        return "Could not reach the Groq API. Check that your GROQ_API_KEY is valid."
    if "429" in text or "rate" in text.lower():
        return "Rate limit reached on Groq. Please wait a moment and try again."
    return f"Error generating response: {text}"


def reset_document():
    for key in ("vector_store", "rag_chain", "documents_metadata"):
        st.session_state[key] = DEFAULT_SESSION_STATES.get(key)
    st.session_state.batch_id = None
    st.session_state.pdf_processed = False
    st.session_state.messages = []
    st.session_state.last_context = ""


# Sidebar
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.session_state.pdf_processed:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_context = ""
            st.rerun()

# Files removed via the uploader's clear/cross icon -> reset the state
if not uploaded_files and st.session_state.pdf_processed:
    reset_document()
    st.rerun()

if uploaded_files:
    # Compute combined hash across all files to detect changes/re-uploads
    combined_bytes = b"".join(f.getvalue() for f in uploaded_files)
    batch_id = hashlib.sha256(combined_bytes).hexdigest()

    if st.session_state.batch_id != batch_id:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.sidebar.error("GROQ_API_KEY not found in .env file.")
        else:
            collection_name = f"batch_{batch_id[:16]}"
            embeddings = get_embedding_model()

            with st.sidebar.status("Processing Documents...", expanded=True) as status:
                ok = False
                try:
                    # 1. Check karo collection pehle se bani hai ya nahi
                    if collection_exists(collection_name):
                        status.update(label="Found cached index! Loading...", state="running")
                        vector_store, all_chunks = load_existing_vector_store(collection_name, embeddings)
                        
                        metadata_by_file = {}
                        for f in uploaded_files:
                            file_chunks = [c for c in all_chunks if c.metadata.get("source_file") == f.name]
                            pages = 0
                            if file_chunks:
                                pages = file_chunks[0].metadata.get(
                                    "total_pages",
                                    max((c.metadata.get("page_number", c.metadata.get("page", 0) + 1) for c in file_chunks), default=1)
                                )
                            metadata_by_file[f.name] = {
                                "num_pages": int(pages),
                                "num_chunks": len(file_chunks) if file_chunks else len(all_chunks),
                            }
                        status.update(label="Loaded existing embeddings from ChromaDB!", state="complete", expanded=False)
                    
                    # 2. Agar collection NAHI mili, toh nayi banayein
                    else:
                        status.update(label="Chunking and embedding documents...", state="running")
                        all_chunks, metadata_by_file, _ = process_multiple_documents(uploaded_files)
                        
                        status.update(label="Building Chroma vector index...", state="running")
                        vector_store = build_new_vector_store(all_chunks, embeddings, collection_name)
                        status.update(label="Index built and saved successfully!", state="complete", expanded=False)

                    # 3. Hybrid retriever initialize karein (k=35 default pool use hoga high recall ke liye)
                    v_retriever, bm25_retriever = create_hybrid_retriever(vector_store, all_chunks, k=35)
                    llm = get_llm(groq_api_key)

                    reset_document()
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = RAGChatbot(v_retriever, bm25_retriever, llm)
                    st.session_state.documents_metadata = metadata_by_file
                    st.session_state.batch_id = batch_id
                    st.session_state.pdf_processed = True
                    ok = True

                except EmptyPDFError as e:
                    status.update(label="No text found", state="error")
                    st.sidebar.error(str(e))
                except Exception as e:
                    status.update(label=f"Error: {e}", state="error")

            if ok:
                st.rerun()
                
# Main UI
st.markdown('<div class="main-title">📚 PDF Chatbot</div>', unsafe_allow_html=True)

if not st.session_state.pdf_processed:
    st.info("👈 Upload one or more PDF files in the sidebar to start asking questions.")
else:
    # Render metadata summary for all loaded documents
    total_docs = len(st.session_state.documents_metadata)
    total_pages = sum(d["num_pages"] for d in st.session_state.documents_metadata.values())
    total_chunks = sum(d["num_chunks"] for d in st.session_state.documents_metadata.values())

    meta_html = f"""
    <div style="background: rgba(30, 41, 59, 0.4); padding: 14px 20px; border-radius: 10px; margin-bottom: 20px; font-size: 0.9rem;">
        <div style="display: flex; gap: 24px; margin-bottom: 8px;">
            <div><b>Total Files:</b> {total_docs}</div>
            <div><b>Total Pages:</b> {total_pages}</div>
            <div><b>Total Chunks:</b> {total_chunks}</div>
        </div>
        <div style="font-size: 0.85rem; color: #94a3b8;">
            <b>Active:</b> {", ".join(html.escape(name) for name in st.session_state.documents_metadata.keys())}
        </div>
    </div>
    """
    st.markdown(meta_html, unsafe_allow_html=True)

    # Render Chat History
    for message in st.session_state.messages:
        st.markdown(chat_bubble_html(message["role"], message["content"]), unsafe_allow_html=True)
        
        # Display the source context used for assistant responses
        if message.get("role") == "assistant" and message.get("context"):
            with st.expander("🔍 View Retrieved Sources & Chunks"):
                st.markdown(f"```text\n{message['context']}\n```")

    # Render Errors (if any)
    if st.session_state.error:
        st.error(st.session_state.error)
        st.session_state.error = None

    # Chat Input
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.rerun()

    # Inference & Streaming Response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        question = st.session_state.messages[-1]["content"]
        history = st.session_state.messages[:-1][-MAX_HISTORY_MESSAGES:]
        placeholder = st.empty()
        full_response, new_context, error = "", None, None

        try:
            gen, new_context = st.session_state.rag_chain.stream(
                question, history, st.session_state.last_context
            )
            for token in gen:
                full_response += token
                placeholder.markdown(chat_bubble_html("assistant", full_response + " ▌"), unsafe_allow_html=True)
            if not full_response.strip():
                raise RuntimeError("The model returned an empty response.")
        except Exception as e:
            error = friendly_error(e)

        if error:
            st.session_state.messages.pop()
            st.session_state.error = f"{error} Please try asking again."
        else:
            st.session_state.last_context = new_context
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "context": new_context  # Preserves retrieved context chunks for each answer
            })
        st.rerun()