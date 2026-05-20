import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import Dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RewardFineTuner:
    def __init__(self, feedback_dir="./data/feedback_buffer", output_dir="./data/reward_adapters"):
        self.feedback_dir = feedback_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model_name = "gpt2"  # lightweight for on-device training
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # LoRA config for efficient fine-tuning
        self.peft_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["c_attn", "c_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, self.peft_config)

    def load_feedback(self):
        data = []
        for file in os.listdir(self.feedback_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.feedback_dir, file), "r") as f:
                    record = json.load(f)
                    reward = record.get("reward", 0)
                    label_text = "Good Response:" if reward > 0 else "Bad Response:"
                    combined = f"{label_text} {record['query']} -> {record['response']}"
                    data.append({"text": combined})
        return Dataset.from_list(data)

    def train(self):
        dataset = self.load_feedback()
        def tokenize(batch): return self.tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
        dataset = dataset.map(tokenize, batched=True)

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
            num_train_epochs=1,
            logging_steps=2,
            save_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
        )

        trainer.train()
        self.model.save_pretrained(self.output_dir)
        print(f"✅ Reward-based fine-tuning complete. Saved to {self.output_dir}")

if __name__ == "__main__":
    trainer = RewardFineTuner()
    trainer.train()
