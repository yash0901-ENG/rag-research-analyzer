RAG-based Research PDF Retrieval
Overview

This project is a simple implementation of a Retrieval-Augmented Generation (RAG) style pipeline for academic PDFs.

It takes a folder of research papers in PDF format, extracts the text, splits it into chunks, converts each chunk into a vector embedding, stores those vectors in a FAISS index, and retrieves the most relevant chunks for a given query.

The goal is to make it easier to search research papers by meaning rather than exact keywords.

Why this exists

Searching research papers is hard because:

Papers are long and dense

Keyword search fails when wording is different

Manual skimming is slow

This project uses semantic embeddings so that queries like “what is self-attention” still find relevant content even if those exact words are not used in the paper.

How it works

PDFs are loaded from a folder and converted to text

Text is split into overlapping chunks

Each chunk is embedded using a sentence embedding model

Embeddings are stored in a FAISS vector index

User queries are embedded and matched against stored vectors

Setup
Requirements

Python 3.10 or 3.11

Tested on Windows, should also work on Linux/Mac

Quick start
git clone https://github.com/yash0901-ENG/rag-research-analyzer.git
cd rag-research-analyzer

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
python rag_pipeline.py

Example

Query:

what are transformers


Output:

You will see the most relevant chunks from the papers that discuss Transformers and self-attention.

Tech used

Python

SentenceTransformers

FAISS

LangChain (community modules)

Limitations

This is retrieval only — it does not generate answers

No ranking evaluation metrics are implemented

Performance depends on chunk size and embedding model

Possible next steps

Add an LLM on top to generate answers

Add evaluation metrics like precision@k

Support hybrid keyword + semantic search

Notes

Put your PDFs inside data/pdfs/ before running

Large PDFs will take longer to process on first run
