import os
import json
from datetime import datetime
from backend.vector_memory import VectorMemory

class ForgetManager:
    def __init__(self,
                 reflection_dir="./data/reflection_data",
                 context_dir="./data/context_memory",
                 chroma_path="./data/chroma_store"):
        self.reflection_dir = reflection_dir
        self.context_dir = context_dir
        self.vector_db = VectorMemory(persist_directory=chroma_path)

    # -------- Inspection --------
    def list_reflections(self):
        files = [f for f in os.listdir(self.reflection_dir) if f.endswith(".json")]
        for i, f in enumerate(sorted(files)):
            with open(os.path.join(self.reflection_dir, f)) as j:
                data = json.load(j)
                print(f"{i+1}. [{f}]  Query: {data['query'][:40]}  ->  Correction: {data['correct_response'][:40]}")
        return files

    def list_contexts(self):
        files = [f for f in os.listdir(self.context_dir) if f.startswith("summary_")]
        for i, f in enumerate(sorted(files)):
            print(f"{i+1}. {f}")
        return files

    # -------- Deletion --------
    def forget_reflection(self, keyword):
        deleted = 0
        for file in os.listdir(self.reflection_dir):
            if not file.endswith(".json"): 
                continue
            path = os.path.join(self.reflection_dir, file)
            with open(path, "r") as f:
                data = json.load(f)
            if any(keyword.lower() in str(v).lower() for v in data.values()):
                os.remove(path)
                deleted += 1
        print(f"🗑️  Deleted {deleted} reflection record(s) containing '{keyword}'.")
        return deleted

    def forget_context(self, summary_id):
        path = os.path.join(self.context_dir, f"{summary_id}.json")
        if os.path.exists(path):
            os.remove(path)
            print(f"🗑️  Deleted summary context {summary_id}.")
            # Also remove from vector DB if present
            try:
                self.vector_db.collection.delete(ids=[summary_id])
            except Exception:
                pass
            return True
        else:
            print(f"⚠️  Summary {summary_id} not found.")
            return False

    # -------- Full wipe (dangerous) --------
    def wipe_all(self):
        confirm = input("⚠️  Type 'CONFIRM' to delete all memory, reflections, and summaries: ")
        if confirm == "CONFIRM":
            for directory in [self.reflection_dir, self.context_dir]:
                for f in os.listdir(directory):
                    os.remove(os.path.join(directory, f))
            self.vector_db.client.delete_collection("memory")
            print("🧹 All local data erased.")
        else:
            print("❌  Wipe canceled.")
