import os
import json
from datetime import datetime
from backend.embeddings import EmbeddingEngine
from backend.vector_memory import VectorMemory
from sentence_transformers import util

class ContextManager:
    def __init__(self, context_dir: str = "./data/context_memory"):
        os.makedirs(context_dir, exist_ok=True)
        self.context_dir = context_dir
        self.embedder = EmbeddingEngine()
        self.memory = VectorMemory()
        self.session_history = []

    def save_interaction(self, user_input: str, model_output: str):
        """Save user-model interaction to disk + memory DB"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": model_output
        }
        self.session_history.append(record)

        # Persist to file
        file_path = os.path.join(self.context_dir, "session_log.jsonl")
        with open(file_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Add to vector memory
        self.memory.add_entry(user_input, model_output)

    def summarize_history(self, threshold: int = 10):
        """Summarize and compress old history entries into one vector"""
        if len(self.session_history) < threshold:
            return None

        combined_text = "\n".join([
            f"User: {m['user']}\nAssistant: {m['assistant']}"
            for m in self.session_history[-threshold:]
        ])
        summary_prompt = (
            "Summarize the following conversation into a short factual memory "
            "without losing context or key facts:\n\n" + combined_text
        )

        summary_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_embedding = self.embedder.embed(summary_prompt)

        # Save to summaries
        summary = {
            "id": summary_id,
            "summary_text": summary_prompt,
            "timestamp": datetime.now().isoformat()
        }

        with open(os.path.join(self.context_dir, f"summary_{summary_id}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Store summary in vector memory
        self.memory.collection.add(
            documents=[summary_prompt],
            embeddings=[summary_embedding],
            ids=[summary_id],
        )

        print(f"🧠 Summarized {threshold} interactions into memory block {summary_id}")
        # Trim old entries
        self.session_history = self.session_history[-3:]

    def retrieve_context(self, query: str, top_k: int = 3):
        """Retrieve the most semantically similar past summaries"""
        results = self.memory.search(query, top_k)
        if not results:
            return ""
        return "\n\n".join([r["text"] for r in results])
