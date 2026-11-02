import re
from uuid import uuid4
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from config import MAX_CONTEXT_CHARS


def build_vector_store(chunks, embeddings):
    # Unique collection per upload: sessions never share or delete each other's index.
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{uuid4().hex}",
    )


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def create_hybrid_retriever(vector_store, chunks, k=4):
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k},
    )
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=_tokenize)
    bm25_retriever.k = k
    return vector_retriever, bm25_retriever


def hybrid_search(query, vector_retriever, bm25_retriever):
    """Reciprocal Rank Fusion of vector and BM25 results."""
    content_to_doc = {}
    scores = {}
    for retriever in (vector_retriever, bm25_retriever):
        seen = set()
        for rank, doc in enumerate(retriever.invoke(query), start=1):
            content = doc.page_content
            if content in seen:
                continue
            seen.add(content)
            content_to_doc.setdefault(content, doc)
            scores[content] = scores.get(content, 0.0) + 1.0 / (60.0 + rank)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [content_to_doc[c] for c in ranked]


def format_docs(docs, max_context_chars=MAX_CONTEXT_CHARS):
    """Join chunks with [Page X] markers. Whole chunks only; a chunk that
    doesn't fit is skipped (smaller lower-ranked ones may still fit)."""
    parts = []
    used = 0
    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        text = f"[Page {page_num}]\n{doc.page_content}"
        sep = 2 if parts else 0
        if used + sep + len(text) > max_context_chars:
            if not parts:  # first chunk alone is too big: truncate it
                parts.append(text[:max_context_chars])
            continue
        parts.append(text)
        used += sep + len(text)
    return "\n\n".join(parts)
