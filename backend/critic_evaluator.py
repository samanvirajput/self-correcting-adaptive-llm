# backend/critic_evaluator.py
import numpy as np
from backend.embeddings import EmbeddingEngine

class CriticEvaluator:
    """
    Evaluates each model response along multiple dimensions:
    - factual alignment (semantic similarity to retrieved docs/context)
    - helpfulness (alignment with past feedback)
    - linguistic quality (clarity & diversity)
    Returns a critic_score between 0 and 1.
    """

    def __init__(self):
        self.embedder = EmbeddingEngine()

    def evaluate(self, query: str, response: str, retrieved_context: str = "", feedback_weight: float = 0.5):
        # 1️⃣ Factual alignment
        if retrieved_context:
            q_vec = self.embedder.embed(query)
            r_vec = self.embedder.embed(response)
            c_vec = self.embedder.embed(retrieved_context)
            factual_score = (self.embedder.cosine_similarity(r_vec, c_vec) + self.embedder.cosine_similarity(q_vec, c_vec)) / 2
        else:
            factual_score = 0.6  # default if no docs available

        # 2️⃣ Helpfulness (use last feedback trend weight if available)
        helpful_score = np.clip(0.5 + feedback_weight / 2, 0, 1)

        # 3️⃣ Linguistic quality
        words = response.split()
        diversity = len(set(words)) / max(len(words), 1)
        clarity = min(1.0, len(words) / 800)  # penalize overly short or long
        ling_score = (diversity * 0.6 + clarity * 0.4)

        # combine (weighted mean)
        critic_score = round(float((0.5*factual_score + 0.3*helpful_score + 0.2*ling_score)), 3)
        return {
            "factuality": round(factual_score, 3),
            "helpfulness": round(helpful_score, 3),
            "linguistic_quality": round(ling_score, 3),
            "critic_score": critic_score
        }
