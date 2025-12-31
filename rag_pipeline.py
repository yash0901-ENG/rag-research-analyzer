import os
import numpy as np

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

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
    # 4. FAISS VECTOR STORE (LangChain)
    # -------------------------
    documents = [Document(page_content=text) for text in texts]
    vectorstore = FAISS.from_documents(documents, embedding=model)

    print(f"Vectors stored in FAISS index: {vectorstore.index.ntotal}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # -------------------------
    # 5. USER QUERY (RETRIEVAL)
    # -------------------------
    query = input("\nEnter your query: ")

    results = retriever.get_relevant_documents(query)

    print("\nRETRIEVED CONTEXT:\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---\n")
        print(doc.page_content)
        print("\n")

