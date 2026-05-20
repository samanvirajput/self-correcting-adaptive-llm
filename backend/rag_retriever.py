# backend/rag_retriever.py
import os
import re
from backend.embedding_wrapper import SentenceTransformerWrapper
from langchain_community.vectorstores import Chroma

def normalize_path_string(s: str) -> str:
    return s.replace("\\", "/").lower().replace(" ", "")

def tokenize_filename_like(s: str):
    # remove common verbs/phrases users add, keep words and extension
    s = s.lower()
    s = re.sub(r'\b(refer|refer to|please|summarize|doc|document|file|the|a|an|please|thanks|thank you)\b', ' ', s)
    s = re.sub(r'[^a-z0-9\.\-_ ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

class RAGRetriever:
    def __init__(self, persist_directory="./data/docs_store"):
        self.persist_directory = persist_directory
        self.embedder = SentenceTransformerWrapper("all-MiniLM-L6-v2")
        # NOTE: different langchain/chroma versions accept different arg names.
        # Try both to be robust.
        try:
            self.db = Chroma(persist_directory=persist_directory, embedding=self.embedder)
        except TypeError:
            self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embedder)

    def extract_filename(self, query: str):
        """Return normalized filename candidate, or None."""
        # Look for patterns like 'something.docx' first
        m = re.search(r'([\w\s\-\_]+?\.(?:docx|pdf|txt|md))', query, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            return cand.lower().replace(" ", "")
        # Otherwise tokenize and attempt to keep last words that may indicate filename
        tokens = tokenize_filename_like(query)
        if not tokens:
            return None
        # If tokens contain a dotless extension word, attempt to append common exts in search (handled later)
        return tokens.replace(" ", "")

    def _metadata_matches_target(self, metadata_source: str, file_target: str) -> bool:
        if not metadata_source or not file_target:
            return False
        meta_norm = normalize_path_string(metadata_source)
        target_norm = file_target.lower().replace(" ", "")
        # direct substring match
        if target_norm in meta_norm:
            return True
        # match by basename
        basename = os.path.basename(meta_norm)
        if target_norm in basename:
            return True
        # fuzzy: check if all words in target appear in basename
        target_words = re.sub(r'\W+', ' ', file_target).split()
        if target_words and all(w in basename for w in target_words):
            return True
        return False

    def search(self, query: str, top_k: int = 3):
        """
        Perform semantic search. If the user mentions a filename, prefer chunks from that file.
        Returns list[str] of page_content.
        """
        # make sure DB exists
        if not os.path.exists(os.path.join(self.persist_directory)):
            print("⚠️ Document DB not found.")
            return []

        file_target = self.extract_filename(query)
        # fetch a larger pool to filter from
        try:
            all_results = self.db.similarity_search(query, k=max(top_k * 3, 6))
        except Exception as e:
            # fallback to small pool
            try:
                all_results = self.db.similarity_search(query, k=top_k)
            except Exception:
                print(f"⚠️ Retriever error: {e}")
                return []

        if not all_results:
            return []

        # If user mentioned a file-like token, filter by metadata
        if file_target:
            filtered = []
            for r in all_results:
                # r is a LangChain Document w/ r.page_content and r.metadata
                src = None
                try:
                    src = r.metadata.get("source") if isinstance(r.metadata, dict) else None
                except Exception:
                    src = None
                if src and self._metadata_matches_target(src, file_target):
                    filtered.append(r)
            if filtered:
                # return top_k page contents
                return [d.page_content for d in filtered[:top_k]]
            else:
                return [d.page_content for d in all_results[:top_k]]

        # No file mentioned — return top_k global snippets
        return [d.page_content for d in all_results[:top_k]]
