import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import MAX_CHUNK_CHARS, CHUNK_OVERLAP, MIN_CHUNK_CHARS


class EmptyPDFError(ValueError):
    """Raised when a PDF contains no extractable text."""


def load_pdf_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        documents = PyMuPDFLoader(temp_path).load()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return documents


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_chunker(_embeddings):
    return SemanticChunker(_embeddings)


def process_uploaded_document(uploaded_file):
    """Returns (num_pages, chunks, embeddings)."""
    all_pages = load_pdf_file(uploaded_file)
    pages_with_text = [d for d in all_pages if d.page_content.strip()]
    if not pages_with_text:
        raise EmptyPDFError(
            "No text found in this PDF. It may be a scanned document (images only); "
            "run OCR on it first and upload again."
        )

    embeddings = get_embedding_model()
    chunks = get_chunker(embeddings).split_documents(pages_with_text)

    # Second pass: cap chunk size so semantic chunks can't grow unbounded.
    capper = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP)
    chunks = capper.split_documents(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]

    if not chunks:
        raise EmptyPDFError("The PDF has too little text to build an index.")

    return len(all_pages), chunks, embeddings
