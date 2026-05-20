
import os
import glob
import json
import math
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datasets import Dataset
import evaluate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_NAME = os.getenv("FT_MODEL_NAME", "gpt2")  

OUTPUT_DIR = os.getenv("FT_OUTPUT_DIR", "./data/adapters/reflect_lora")
BATCH_SIZE = int(os.getenv("FT_BS", "4"))
EPOCHS = int(os.getenv("FT_EPOCHS", "3"))
LR = float(os.getenv("FT_LR", "1e-4"))
MAX_LENGTH = int(os.getenv("FT_MAX_LEN", "256"))
WEIGHT_COLUMN = "reflection_weight"  

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_reflection_jsons(reflect_dir: str = "./data/reflection_data"):
    files = sorted(glob.glob(os.path.join(reflect_dir, "*.json")))
    examples = []
    for f in files:
        try:
            j = json.load(open(f, "r"))
            # Expect fields: query, wrong_response, correct_response, reflection_weight
            query = j.get("query", "")
            correct = j.get("correct_response") or j.get("correct", j.get("correct_response", ""))
            wrong = j.get("wrong_response", "")
            weight = float(j.get("reflection_weight", 1.0))
            # Build an instruction-style example:
            instruction = f"User: {query}\nAssistant (corrected): {correct}"
            examples.append({"text": instruction, "weight": weight})
        except Exception as e:
            print(f"⚠️ Skipping invalid file {f}: {e}")
    return examples


def prepare_dataset(examples: List[Dict[str, Any]], tokenizer: AutoTokenizer):
    # Tokenize and create a huggingface Dataset
    texts = [ex["text"] for ex in examples]
    weights = [ex.get("weight", 1.0) for ex in examples]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokenized["weights"] = weights
    # Dataset expects list of dicts
    ds = Dataset.from_dict(tokenized)
    return ds


class WeightedTrainer(Trainer):
    """Trainer subclass that supports per-sample weighting in loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        inputs contains:
          - input_ids, attention_mask, labels, weights
        We compute token-level CE and multiply by a scalar sample weight.
        """
        labels = inputs.get("labels")
        weights = inputs.pop("weights", None)
        outputs = model(**inputs)
        loss = outputs.loss  # this is averaged over tokens (default)
        if weights is not None:
            # weights is shape (batch,) - need to scale loss per example
            # Here we assume labels are padded equally; for simplicity multiply scalar mean weight across batch.
            # For more exact per-token weighting you'd compute token-level losses manually.
            batch_weight = torch.mean(weights)
            loss = loss * batch_weight
        return (loss, outputs) if return_outputs else loss


def main():
    print("🔹 Loading reflection examples...")
    examples = load_reflection_jsons()
    if len(examples) == 0:
        print("⚠️ No reflection examples found in ./data/reflection_data. Exiting.")
        return

    print(f"🔹 Found {len(examples)} reflection examples. Preparing tokenizer & model: {MODEL_NAME}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # If causal LM tokenizer missing pad token, add it
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Model loading strategy (try to be memory-efficient)
    use_kbit = False
    model = None
    try:
        # If you have bitsandbytes and want 4-bit fine-tuning, enable prepare_model_for_kbit_training & bnb config.
        # For Mac M3, you likely won't have bitsandbytes compiled. We fallback to CPU FP32 if not present.
        print("🔹 Loading model (this may be slow on CPU)...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"⚠️ Could not load model {MODEL_NAME} normally: {e}")
        print("Attempting to load with low_cpu_mem_usage=True")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)

    # Apply LoRA (PEFT)
    peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj"]  # compatible with Phi and GPT-style models
)

    # Prepare model for k-bit if needed (placeholder; will be safe if not using kbit)
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Dataset
    ds = prepare_dataset(examples, tokenizer)

    # Prepare labels (for causal LM, labels = input_ids with -100 for padding)
    def labelize(example):
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        labels = [iid if am == 1 else -100 for iid, am in zip(input_ids, attention_mask)]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": example["weights"]}

    ds = ds.map(labelize, remove_columns=ds.column_names)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    print("🔹 Starting fine-tuning...")
    trainer.train()

    # Save adapters
    print(f"🔹 Saving adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    print("✅ Fine-tuning complete. Adapter saved.")

if __name__ == "__main__":
    main()
