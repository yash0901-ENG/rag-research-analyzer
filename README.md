📚 Retrieval-Augmented Generation (RAG) for Research PDFs
📌 Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline for academic research documents.

The system ingests PDF research papers, converts them into machine-readable text, splits them into meaningful chunks, embeds each chunk into a semantic vector space, stores them in a vector database, and retrieves the most relevant document segments for a user query using semantic similarity.

The goal is to enable meaning-based retrieval from long and complex research papers instead of relying on simple keyword matching.

🧠 Problem

Research papers are:

Long and dense

Difficult to search manually

Poorly served by keyword-based search when queries are phrased differently from the original text

This system addresses these issues by retrieving content based on semantic meaning, not just surface-level word matches.

🏗️ Architecture
PDFs
  ↓
Text Extraction
  ↓
Chunking (overlapping)
  ↓
Semantic Embeddings
  ↓
FAISS Vector Store
  ↓
Query-time Retrieval

⚙️ Methodology
1. Document Ingestion

PDF files are loaded and converted into plain text so they can be processed by NLP models.

2. Chunking

Long texts are split into smaller overlapping segments.

Chunk size: 1000 characters

Overlap: 200 characters

This preserves contextual continuity across chunk boundaries.

3. Embeddings

Each chunk is converted into a numerical vector using a sentence embedding model.

Model: all-MiniLM-L6-v2

Vector dimension: 384

4. Vector Store

All embeddings are stored in a FAISS index for efficient semantic similarity search.

5. Retrieval

A user query is embedded into the same vector space and compared against stored vectors.
The system retrieves the top-k most similar chunks as relevant context.

🧪 Example

Query

What is self-attention?


Retrieved Output (excerpt)

The Transformer model is based entirely on self-attention mechanisms, dispensing with recurrence and convolution...

🧰 Tech Stack

Python

LangChain (community modules)

SentenceTransformers

FAISS

NumPy

⚠️ Limitations

Retrieval-only system (no LLM-based answer generation).

Not designed for statistical text analysis.

Retrieval quality depends on chunking and embedding model.

🚀 Possible Extensions

Add an LLM for answer generation.

Add evaluation metrics (precision@k, recall@k).

Support hybrid keyword + semantic search.

Scale to very large document collections.

⚙️ Environment Requirements

Python 3.10 or 3.11 recommended

Not tested on Python 3.12+ or Python 3.9

⚡ Quick Start
1. Clone the repository
git clone https://github.com/yash0901-ENG/rag-research-analyzer.git
cd rag-research-analyzer

2. Create and activate virtual environment

Windows

python -m venv venv
venv\Scripts\activate


Mac/Linux

python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the pipeline
python rag_pipeline.py
