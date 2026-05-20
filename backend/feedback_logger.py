import os
import json
from datetime import datetime

class FeedbackLogger:
    def __init__(self, feedback_dir: str = "./data/feedback_buffer"):
        self.feedback_dir = feedback_dir
        os.makedirs(self.feedback_dir, exist_ok=True)

    def log_feedback(self, query: str, response: str, reward: int):
        """Store user feedback as reinforcement signal (+1 or -1)."""
        record = {
            "query": query,
            "response": response,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        }
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.feedback_dir, filename)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"💾 Feedback logged (reward={reward}) -> {filename}")

    def get_feedback_dataset(self):
        """Load all feedback entries."""
        dataset = []
        for file in os.listdir(self.feedback_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.feedback_dir, file), "r") as f:
                    dataset.append(json.load(f))
        return dataset
