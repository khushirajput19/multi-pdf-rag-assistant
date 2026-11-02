import os
import hashlib
import streamlit as st
from dotenv import load_dotenv

from config import DEFAULT_SESSION_STATES, CUSTOM_CSS, RETRIEVER_K, MAX_HISTORY_MESSAGES
from utils.ui_helpers import chat_bubble_html
from core.document_loader import process_uploaded_document, EmptyPDFError
from core.retriever import build_vector_store, create_hybrid_retriever
from core.rag_engine import get_llm, RAGChatbot

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, val in DEFAULT_SESSION_STATES.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def friendly_error(e: Exception) -> str:
    text = str(e)
    if any(s in text for s in ("403", "401", "Unauthorized", "invalid_api_key")):
        return ("Could not reach the Groq API. Check that your GROQ_API_KEY is valid.")
    if "429" in text or "rate" in text.lower():
        return "Rate limit reached on Groq. Please wait a moment and try again."
    return f"Error generating response: {text}"


def reset_document():
    vs = st.session_state.vector_store
    if vs is not None:
        try:
            vs.delete_collection()
        except Exception:
            pass
    for key in ("vector_store", "rag_chain", "document_name", "document_id"):
        st.session_state[key] = DEFAULT_SESSION_STATES[key]
    st.session_state.num_pages = 0
    st.session_state.num_chunks = 0
    st.session_state.pdf_processed = False
    st.session_state.messages = []
    st.session_state.last_context = ""


# Sidebar
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

    if st.session_state.pdf_processed:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_context = ""
            st.rerun()

# File removed with the uploader's X -> drop the old document
if uploaded_file is None and st.session_state.pdf_processed:
    reset_document()
    st.rerun()

# Pipeline
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.document_id != file_id:
        hf_api_key = os.getenv("GROQ_API_KEY")
        if not hf_api_key:
            st.sidebar.error("GROQ_API_KEY not found in .env file.")
        else:
            with st.sidebar.status("Processing PDF...", expanded=True) as status:
                try:
                    num_pages, chunks, embeddings = process_uploaded_document(uploaded_file)
                    vector_store = build_vector_store(chunks, embeddings)
                    v_retriever, bm25_retriever = create_hybrid_retriever(vector_store, chunks, k=RETRIEVER_K)
                    llm = get_llm(hf_api_key)

                    reset_document()  # remove the previous document's index
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = RAGChatbot(v_retriever, bm25_retriever, llm)
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.document_id = file_id
                    st.session_state.num_pages = num_pages
                    st.session_state.num_chunks = len(chunks)
                    st.session_state.pdf_processed = True

                    status.update(label="PDF Processed Successfully!", state="complete", expanded=False)
                    ok = True
                except EmptyPDFError as e:
                    status.update(label="No text found", state="error")
                    st.sidebar.error(str(e))
                    ok = False
                except Exception as e:
                    status.update(label=f"Error: {e}", state="error")
                    ok = False
            if ok:
                st.rerun()

# Main UI
st.markdown('<div class="main-title">📚 PDF Chatbot</div>', unsafe_allow_html=True)

if not st.session_state.pdf_processed:
    st.info("👈 Upload a PDF file in the sidebar to start asking questions.")
else:
    import html
    st.markdown(
        f"""
    <div style="display: flex; gap: 20px; background: rgba(30, 41, 59, 0.4); padding: 12px 20px; border-radius: 10px; margin-bottom: 20px; font-size: 0.9rem;">
        <div><b>Document:</b> {html.escape(st.session_state.document_name)}</div>
        <div><b>Pages:</b> {st.session_state.num_pages}</div>
        <div><b>Chunks:</b> {st.session_state.num_chunks}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        st.markdown(chat_bubble_html(message["role"], message["content"]), unsafe_allow_html=True)

    if st.session_state.error:
        st.error(st.session_state.error)
        st.session_state.error = None

    user_question = st.chat_input("Ask a question about your document...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.rerun()

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
            # Errors are shown once, never stored as chat history.
            st.session_state.messages.pop()
            st.session_state.error = f"{error} Please try asking again."
        else:
            st.session_state.last_context = new_context
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
