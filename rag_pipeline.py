import os
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def load_pdfs(pdf_dir):
    documents = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            documents.extend(loader.load())
    return documents


if __name__ == "__main__":

    # -------------------------
    # 1. LOAD PDF DOCUMENTS
    # -------------------------
    pdf_directory = "data/pdfs"
    docs = load_pdfs(pdf_directory)
    print(f"Total pages loaded: {len(docs)}")

    # -------------------------
    # 2. CHUNKING
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    # -------------------------
    # 3. EMBEDDINGS (FREE, LOCAL)
    # -------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)

    print(f"Embedding vector size: {embeddings.shape[1]}")

    # -------------------------
    # 4. FAISS VECTOR STORE
    # -------------------------
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    print(f"Vectors stored in FAISS index: {index.ntotal}")

    # -------------------------
    # 5. USER QUERY (RETRIEVAL)
    # -------------------------
    query = input("\nEnter your query: ")

    query_embedding = model.encode([query])
    k = 3  # number of chunks to retrieve

    distances, indices = index.search(query_embedding, k)

    print("\nRETRIEVED CONTEXT:\n")

    for i, idx in enumerate(indices[0], 1):
        print(f"--- Result {i} ---\n")
        print(chunks[idx].page_content)
        print("\n")


