import os
import re # Raw text se unwanted characters, formatting aur spaces clean karne ke liye[cite: 9].
import chromadb
from uuid import uuid4 # Vector database me har ek chunk ko unique document ID dene ke liye taaki duplicate na bane[cite: 9].
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder 
from langchain_chroma import Chroma
from langchain_core.documents import Document
import streamlit as st
from config import MAX_CONTEXT_CHARS # context ki length limit karke token cost aur window overflow rokne ke liye[cite: 9].


# Caching cross-encoder so it loads only once taaki har search par model baar baar reload na ho[cite: 9]
@st.cache_resource(show_spinner=False)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


CHROMA_PERSIST_DIR = "./chroma_db"

def collection_exists(collection_name: str) -> bool:
    """Check karta hai ki kya collection disk par pehle se maujood hai[cite: 9]."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    existing_collections = [col.name for col in client.list_collections()]
    return collection_name in existing_collections

def load_existing_vector_store(collection_name: str, embeddings):
    """Disk se existing collection aur uske stored documents fetch karta hai bina re-embed kiye[cite: 9]."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    raw_data = vector_store.get()
    retrieved_chunks = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw_data.get("documents", []), raw_data.get("metadatas", []))
    ]
    return vector_store, retrieved_chunks

def build_new_vector_store(chunks, embeddings, collection_name: str):
    """Naye chunks ko deterministic IDs ke sath ChromaDB me save karta hai[cite: 9]."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # Metadata se chunk_id nikalte hain, agar kisi wajah se na ho toh fallback me safe uuid banate hain
    ids = [c.metadata.get("chunk_id", str(uuid4())) for c in chunks]
    vector_store = Chroma.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        ids=ids,  # Passes chunk IDs explicitly[cite: 9]
    )
    return vector_store


# Yeh mostly BM25Retriever ke preprocessing ke liye use hota hai, kyunki BM25 ko keyword frequency count karne ke liye tokenized words chahiye hote hain[cite: 9]
def _tokenize(text: str): # (_) indicate karta hai ki yeh ek internal/private utility function hai[cite: 9].
    return re.findall(r"\w+", text.lower()) # extract only alphanumeric words, punctuation are skiped[cite: 9]

def create_hybrid_retriever(vector_store, chunks, k: int = 35):
    # k=35 isliye rakha hai taaki shuruat me dense aur keyword dono broad candidate pool capture karein aur high recall mile[cite: 9]
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k},
    )
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=_tokenize)
    bm25_retriever.k = k
    return vector_retriever, bm25_retriever

def hybrid_search(query: str, vector_retriever, bm25_retriever, top_k: int = 5, score_threshold: float = -1.2):
    content_to_doc = {} # Duplicate chunks avoid karne aur text se poore Document object ko map karne ke liye[cite: 9].
    scores = {} # Har chunk ka fused RRF score store karne ke liye[cite: 9].

    # STEP 1: Reciprocal Rank Fusion (Dense + BM25)[cite: 9]
    for retriever in (vector_retriever, bm25_retriever):
        seen = set() # Ek hi retriever agar duplicate text de de, toh rank penalty ya double scoring rokne ke liye deduplication karta hai[cite: 9].
        for rank, doc in enumerate(retriever.invoke(query), start=1):
            content = doc.page_content
            if content in seen:
                continue
            seen.add(content)
            content_to_doc.setdefault(content, doc)
            scores[content] = scores.get(content, 0.0) + 1.0 / (60.0 + rank)
            # Constant 60 ek standard damping factor hai jo higher-ranked items ke outlier impact ko balance karta hai[cite: 9].
            # Agar koi document dono (Vector aur BM25) me aata hai, toh uske scores add ho jaate hain aur wo naturally top par aa jata hai[cite: 9].

    # STEP 2: Broad Candidate Pool Selection (Top 35 candidates RRF se uthate hain taaki recall drop na ho)[cite: 9]
    initial_ranked = sorted(scores, key=scores.get, reverse=True)[:35]
    candidate_docs = [content_to_doc[c] for c in initial_ranked]

    if not candidate_docs:
        return []

    # STEP 3: Cross-Encoder Re-Ranking (Query aur Document text ka full cross-attention)
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs)

    # Score ke descending order me sort karte hain taaki sabse best match Rank 1 par aaye
    scored_docs = sorted(zip(candidate_docs, rerank_scores), key=lambda x: x[1], reverse=True)

    # STEP 4: Context Noise Reduction (Score Thresholding)
    # Sirf wahi chunks prompt context me bhejenge jinka semantic confidence score threshold (-2.5) se upar ho
    filtered_docs = [doc for doc, score in scored_docs if score >= score_threshold]

    # Safety Guardrail: Agar kisi rare query me sabhi documents threshold ke neeche gir jayein, toh at least sabse best top-1 document zarur bhejte hain
    if not filtered_docs:
        filtered_docs = [scored_docs[0][0]]

    # Final top_k chunks return karte hain
    return filtered_docs[:top_k]


def format_docs(docs, max_context_chars=MAX_CONTEXT_CHARS):
    parts = []
    used = 0
    seen_parents = set()

    for doc in docs:
        source_name = doc.metadata.get("source_file", "Document")
        page_num = doc.metadata.get("page_number", 1)
        
        parent_id = doc.metadata.get("parent_id")
        parent_content = doc.metadata.get("parent_content")

        if parent_id and parent_content:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            content_text = parent_content.strip()
        else:
            content_text = doc.page_content.strip()

        text = f"[{source_name}, Page {page_num}]\n{content_text}"

        sep = 2 if parts else 0
        if used + sep + len(text) > max_context_chars:
            if not parts:
                parts.append(text[:max_context_chars])
            break

        parts.append(text)
        used += sep + len(text)

    return "\n\n---\n\n".join(parts)