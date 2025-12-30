📚 Retrieval-Augmented Generation (RAG) for Research PDFs

📌 Overview



This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline for academic research documents.

The system ingests PDF research papers, converts them into machine-readable text, splits them into meaningful chunks, embeds each chunk into a semantic vector space, stores them in a vector database, and retrieves the most relevant document segments for a user query using semantic similarity.



The goal is to enable meaning-based retrieval from long and complex research papers instead of relying on simple keyword matching.



🧠 Problem



Research papers are:



Long and dense



Hard to search manually



Poorly served by keyword search when queries are phrased differently from the original text



This system solves that by retrieving content based on semantic meaning, not surface-level word matches.



🏗️ Architecture

PDFs

&nbsp;↓

Text Extraction

&nbsp;↓

Chunking (overlapping)

&nbsp;↓

Semantic Embeddings

&nbsp;↓

FAISS Vector Store

&nbsp;↓

Query-time Retrieval



⚙️ Methodology

1\. Document Ingestion



PDF files are loaded and converted into plain text so they can be processed by NLP models.



2\. Chunking



Long texts are split into smaller overlapping segments.



Chunk size: 1000 characters



Overlap: 200 characters

This preserves context across boundaries and avoids loss of meaning.



3\. Embeddings



Each chunk is converted into a numerical vector using a sentence embedding model.



Model: all-MiniLM-L6-v2



Vector dimension: 384



4\. Vector Store



All embeddings are stored in a FAISS index to enable fast semantic similarity search.



5\. Retrieval



A user query is embedded into the same vector space and compared to all stored vectors.

The system retrieves the top-k most similar chunks as relevant context.



🧪 Example

Query

What is self-attention?



Retrieved Output (excerpt)

The Transformer model is based entirely on self-attention mechanisms, 

dispensing with recurrence and convolution...





This shows that the system retrieves conceptually relevant passages, even without exact keyword matching.



🧰 Tech Stack



Python



LangChain (community modules)



SentenceTransformers



FAISS



NumPy



⚠️ Limitations



Designed for semantic retrieval, not statistical analysis (e.g., counting word frequencies).



Does not generate final answers using an LLM (retrieval only).



Retrieval quality depends on embedding model and chunking parameters.



🚀 Possible Extensions



Add an LLM to generate natural-language answers from retrieved chunks.



Add retrieval evaluation metrics (precision@k, recall@k).



Support hybrid keyword + semantic search.



Scale to large document collections.



🎯 Summary



This project demonstrates:



Document ingestion and preprocessing



Recursive chunking with overlap



Semantic embedding generation



Vector database construction



Query-time semantic retrieval



It reflects a research-oriented approach to building explainable and modular RAG systems.

