#  LLM Data Pipeline – Retrieval-Augmented Generation (RAG) Lab

This project implements a complete **Retrieval-Augmented Generation (RAG) pipeline** for unstructured documents.  
It demonstrates how to transform raw files into a system that can **retrieve relevant information** and **answer questions using an LLM**.

This lab follows modern MLOps concepts such as modular design, traceability, reproducibility, and clean data flows.

---

## Pipeline Overview

               ┌─────────────────────┐
               │   Raw Documents     │
               │  (PDF / TXT / MD)   │
               └──────────┬──────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  Ingestion Layer │
                │ (PDF/MD/TXT Read)│
                └──────────┬───────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Preprocessing    │
                │ Clean / Normalize│
                └──────────┬───────┘
                          │
                          ▼
                ┌──────────────────┐
                │     Chunking     │
                │ Overlap Strategy │
                └──────────┬───────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ Embedding Model         │
              │ all-MiniLM-L6-v2        │
              └──────────┬──────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ FAISS Vector DB  │
                │   (IndexFlatL2)  │
                └──────────┬───────┘
                          │
                          ▼
                  ┌──────────────┐
                  │ Retrieval    │
                  │  Top-K Chunks│
                  └───────┬──────┘
                          │
                          ▼
               ┌─────────────────────┐
               │   LLM Answering     │
               │ Local HF / OpenAI   │
               └─────────────────────┘


---

##  Features

###  1.Multi-Format Document Ingestion
Supports:
- `.pdf`
- `.txt`
- `.md`

### 2.Preprocessing & Cleaning
- Unicode and whitespace normalization  
- Removal of empty/invalid pages  
- Consistent cleaning for embeddings

### 3.Smart Chunking
Using `RecursiveCharacterTextSplitter` with:
- `chunk_size` (default 400)
- `chunk_overlap` (default 60)

### 4.High-Quality Embeddings
Model used:
sentence-transformers/all-MiniLM-L6-v2
- Fast, lightweight, and great semantic performance.

### 5.FAISS Vector Store
Efficient similarity search using: faiss.IndexFlatL2


---

##  How to Run

### 1. Install Dependencies
```bash
pip install sentence-transformers faiss-cpu pypdf langchain langchain-text-splitters python-dotenv transformers
```

### 2.Add Your Documents

Create a folder named documents/ and add:

- PDFs

- Text files

- Markdown notes

### 3.Execute Notebook Cells

In order:

- Ingestion

- Preprocessing

- Chunking

- Embedding

- FAISS Indexing

- Retrieval

- LLM Answering

### 4. Ask Questions

Example:
```bash
answer_query("What is the key idea of this document?", index, chunks, metadata)
```
