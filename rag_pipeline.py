import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    """Adapter to make SentenceTransformer compatible with LangChain."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


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
    embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

    texts = [chunk.page_content for chunk in chunks]

    print("Encoding embeddings...")
    sample_vec = embedding_model.embed_documents([texts[0]])[0]
    print(f"Embedding vector size: {len(sample_vec)}")

    # -------------------------
    # 4. FAISS VECTOR STORE
    # -------------------------
    documents = [Document(page_content=text) for text in texts]

    vectorstore = FAISS.from_documents(documents, embedding=embedding_model)

    print(f"Vectors stored in FAISS index: {vectorstore.index.ntotal}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # -------------------------
    # 5. USER QUERY (RETRIEVAL)
    # -------------------------
    query = input("\nEnter your query: ")

    results = retriever.invoke(query)

    print("\nRETRIEVED CONTEXT:\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---\n")
        print(doc.page_content)
        print("\n")

