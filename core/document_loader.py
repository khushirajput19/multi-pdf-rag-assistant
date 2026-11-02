import bisect
import os
import re
import tempfile
from typing import List, Dict, Tuple, Any

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import MIN_CHUNK_CHARS


class EmptyPDFError(ValueError):
    """Raised when a PDF contains no extractable text."""

# Avoid problematic characters in filenames, Create safer file paths.
def sanitize_filename(name: str) -> str:
    base_name = os.path.basename(name)
    return re.sub(r"[^\w\.-]", "_", base_name)

# takes an uploaded PDF, temporarily saves it, extracts its text using PyMuPDFLoader, adds the original filename to metadata, and finally deletes the temporary file.
def load_single_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file: # creates a temporary file with a name.
        temp_file.write(uploaded_file.getvalue()) # retrieves the complete binary content of the uploaded PDF, writes those bytes into the temporary PDF file..
        temp_path = temp_file.name  # retrieves the path of that temporary file.

    try:
        documents = PyMuPDFLoader(temp_path).load() # extracts the PDF content and returns a list of LangChain Document objects.
        for doc in documents:
            doc.metadata["source_file"] = uploaded_file.name
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path) # We don't want unnecessary temporary PDFs accumulating on the system and consuming storage.

    return documents

# doesn't have to load the embedding model again every time the app reruns.
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def process_multiple_documents(uploaded_files):
    embeddings = get_embedding_model()

    # add_start_index=True lets us map every chunk back to its exact page
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3200,
        chunk_overlap=800,
        separators=["\n\n", "\n", " "],
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
        add_start_index=True,
    )

    all_child_chunks = [] # store child chunks from every uploaded PDF.
    metadata_by_file = {} # dictionary stores summary information about each uploaded PDF.

    for file in uploaded_files: # processes each uploaded PDF one by one.
        filename = getattr(file, "name", "unknown_document.pdf") # retrieves an attribute from an object, getattr(object, attribute, default_value).
        clean_name = sanitize_filename(filename)

        all_pages = load_single_pdf(file)
        pages_with_text = [d for d in all_pages if d.page_content and d.page_content.strip()] # filters out pages that are empty or contain only whitespace.
        if not pages_with_text:
            continue

        # Stitch pages (so bullets spanning pages stay together) but remember
        # where each page starts, instead of embedding markers in the text.
        stitched_text = "" # Stores the combined text of all readable pages
        page_starts, page_nums = [], [] # Stores the starting character position of each page, Stores the actual PDF page numbers
        for d in pages_with_text:
            page_starts.append(len(stitched_text))
            page_nums.append(d.metadata.get("page", 0) + 1)
            stitched_text += d.page_content.strip() + "\n\n"

        def page_at(pos: int) -> int:
            return page_nums[bisect.bisect_right(page_starts, pos) - 1]

        file_child_chunks = []
        global_child_idx = 0

        for p_idx, parent in enumerate(parent_splitter.create_documents([stitched_text])):
            parent_start = parent.metadata["start_index"]
            parent_page = page_at(parent_start)
            parent_id = f"{clean_name}_p{parent_page}_parent_{p_idx}"
            parent_content = parent.page_content.strip()

            for child in child_splitter.create_documents([parent.page_content]):
                c_text = child.page_content
                if len(c_text.strip()) < MIN_CHUNK_CHARS:
                    continue

                # Page of the CHILD itself (not the parent's last page)
                child_page = page_at(parent_start + child.metadata["start_index"])

                file_child_chunks.append(Document(
                    page_content=c_text,
                    metadata={
                        "chunk_id": f"{clean_name}_p{child_page}_c{global_child_idx}",
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "source_file": filename,
                        "clean_file_name": clean_name,
                        "page_number": child_page,
                    },
                ))
                global_child_idx += 1

        all_child_chunks.extend(file_child_chunks)
        metadata_by_file[filename] = {
            "num_pages": len(all_pages),
            "readable_pages": len(pages_with_text),
            "num_chunks": len(file_child_chunks),
            "clean_name": clean_name,
        }

    if not all_child_chunks:
        raise EmptyPDFError("Uploaded PDFs contain no readable text to build an index.")

    return all_child_chunks, metadata_by_file, embeddings