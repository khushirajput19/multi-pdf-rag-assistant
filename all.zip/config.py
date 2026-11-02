DEFAULT_SESSION_STATES = {
    "messages": [],
    "vector_store": None,
    "rag_chain": None,
    "document_name": None,
    "document_id": None,
    "num_pages": 0,
    "num_chunks": 0,
    "last_context": "",
    "pdf_processed": False,
    "error": None,
}

# Chunking / retrieval
MAX_CHUNK_CHARS = 1200      # hard cap applied after semantic chunking
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 30        # drop tiny fragments (page numbers, headers)
RETRIEVER_K = 4
MAX_CONTEXT_CHARS = 6000

# LLM
MAX_NEW_TOKENS = 1024
MAX_HISTORY_MESSAGES = 6    # last N messages sent to the LLM

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #080808;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #0b0c14;
    border-right: 1px solid #1f2937;
}

.main-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 2.2rem;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.chat-bubble {
    padding: 16px 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    line-height: 1.6;
}

.chat-bubble.user {
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    color: #ffffff;
    margin-left: auto;
    max-width: 80%;
}

.chat-bubble.assistant {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #e2e8f0;
    margin-right: auto;
    max-width: 85%;
}

.chat-bubble table { border-collapse: collapse; margin: 8px 0; }
.chat-bubble th, .chat-bubble td { border: 1px solid rgba(255,255,255,0.15); padding: 6px 10px; }
.chat-bubble p:last-child { margin-bottom: 0; }

.chat-avatar {
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 6px;
}

.user-avatar { color: #a5b4fc; }
.assistant-avatar { color: #f472b6; }
</style>
"""
