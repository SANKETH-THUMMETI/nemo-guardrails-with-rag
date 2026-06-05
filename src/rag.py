"""
RAG pipeline: build a Chroma vector store from HR documents and retrieve relevant chunks.
"""

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

from src.hr_docs import HR_DOCUMENTS

EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100


def _make_documents():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = []
    for item in HR_DOCUMENTS:
        chunks = splitter.create_documents(
            texts=[item["content"]],
            metadatas=[{"source": item["title"]}],
        )
        docs.extend(chunks)
    return docs


@st.cache_resource(show_spinner="Building HR policy index… (first run downloads ~40 MB embedding model)")
def build_vectorstore():
    embeddings = FastEmbedEmbeddings(model_name=EMBED_MODEL)
    docs = _make_documents()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="hr_policies",
    )
    return vectorstore


def retrieve(query: str, vectorstore, k: int = 3):
    """Return top-k chunks with relevance scores."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content,
            "score": round(score, 3),
        }
        for doc, score in results
    ]
