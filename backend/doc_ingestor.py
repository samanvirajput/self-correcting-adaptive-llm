import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

import docx2txt

from embedding_wrapper import SentenceTransformerWrapper


class DocumentIngestor:
    """
    Reads, chunks, and embeds user documents (PDF, TXT, MD, PY, DOCX)
    into a local Chroma vector database for semantic retrieval.
    """

    def __init__(self, persist_directory="./data/docs_store"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.embedder = SentenceTransformerWrapper("all-MiniLM-L6-v2")

    def load_documents(self, file_path):
        """
        Load a document based on its extension.
        Supported: .pdf, .txt, .md, .py, .docx
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()

        elif ext in [".txt", ".md", ".py"]:
            loader = TextLoader(file_path)
            return loader.load()

        elif ext == ".docx":
            text = docx2txt.process(file_path)
            # Convert text into a proper LangChain Document object
            return [Document(page_content=text, metadata={"source": file_path})]

        else:
            raise ValueError(f"❌ Unsupported file type: {ext}")

    def chunk_and_embed(self, file_path):
        """
        Split document into chunks and embed them into Chroma vector store.
        """
        print(f"📄 Processing {file_path}...")
        docs = self.load_documents(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)

        # Create or append to Chroma vector DB
        Chroma.from_documents(
            chunks,
            embedding=self.embedder,  # ✅ updated for new API
            persist_directory=self.persist_directory
        )

        print(f"✅ Embedded and stored {len(chunks)} chunks from {file_path} into vector DB.")

    def build_index(self, folder_path):
        """
        Iterate through user_docs folder and embed all supported files.
        """
        for file in os.listdir(folder_path):
            if file.endswith((".pdf", ".txt", ".md", ".py", ".docx")):
                try:
                    self.chunk_and_embed(os.path.join(folder_path, file))
                except Exception as e:
                    print(f"⚠️ Skipping {file}: {e}")
        print(f"✅ Document store built at {self.persist_directory}")


if __name__ == "__main__":
    ingestor = DocumentIngestor()
    user_docs_dir = "./user_docs"
    if not os.path.exists(user_docs_dir):
        os.makedirs(user_docs_dir, exist_ok=True)
        print("📁 Created ./user_docs directory — add your PDFs, DOCXs, or TXT files there.")
    else:
        print(f"📂 Scanning {user_docs_dir} for documents...")
        ingestor.build_index(user_docs_dir)
