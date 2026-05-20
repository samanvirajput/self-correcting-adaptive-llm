import os
import json
from datetime import datetime
from backend.embeddings import EmbeddingEngine

class SelfReflection:
    def __init__(self, dataset_dir: str = "./data/reflection_data"):
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.embedder = EmbeddingEngine()

    def log_feedback(self, query: str, wrong_response: str, correct_response: str):
        """Store correction data and compute reflection weight"""
        wrong_emb = self.embedder.embed(wrong_response)
        corr_emb = self.embedder.embed(correct_response)
        similarity = self.embedder.cosine_similarity(wrong_emb, corr_emb)
        reflection_weight = round(1 - similarity, 4)

        record = {
            "query": query,
            "wrong_response": wrong_response,
            "correct_response": correct_response,
            "reflection_weight": reflection_weight,
            "timestamp": datetime.now().isoformat()
        }

        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.dataset_dir, file_name)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"✅ Logged correction with reflection weight {reflection_weight:.3f}")
        return reflection_weight

    def get_reflection_dataset(self):
        """Load all feedback samples"""
        dataset = []
        for file in os.listdir(self.dataset_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.dataset_dir, file), "r") as f:
                    dataset.append(json.load(f))
        return dataset
    

