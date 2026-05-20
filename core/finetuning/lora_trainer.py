import os
import glob
import json
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

FINETUNE_THRESHOLD = int(os.getenv("FINETUNE_THRESHOLD", "50"))
FINETUNE_OUTPUT_DIR = os.getenv("FINETUNE_OUTPUT_DIR", "./data/adapters")


@dataclass
class TrainingPair:
    instruction: str
    response: str
    weight: float = 1.0


def prepare_training_data(corrections) -> list[TrainingPair]:
    """Convert CorrectionPattern list + reflection JSON files into training pairs."""
    pairs: list[TrainingPair] = []

    # Load from reflection_data JSON files (existing format)
    reflect_dir = "./data/reflection_data"
    if os.path.isdir(reflect_dir):
        for fpath in sorted(glob.glob(os.path.join(reflect_dir, "*.json"))):
            try:
                with open(fpath) as f:
                    rec = json.load(f)
                query = rec.get("query", "")
                correct = rec.get("correct_response", "")
                weight = float(rec.get("reflection_weight", 1.0))
                if query and correct and not correct.startswith("[AUTO-REFLECTED]"):
                    pairs.append(TrainingPair(
                        instruction=f"User: {query}",
                        response=f"Assistant: {correct}",
                        weight=weight,
                    ))
            except Exception:
                pass

    # Also convert correction patterns into guidance pairs
    for p in (corrections or []):
        inst = f"User: [correction] Avoid {p.pattern_type}: {p.description}"
        resp = f"Assistant: Understood. I will avoid {p.pattern_type} going forward."
        pairs.append(TrainingPair(instruction=inst, response=resp, weight=p.confidence))

    return pairs


def train(
    model_name: str,
    training_data: list[TrainingPair],
    output_dir: str = FINETUNE_OUTPUT_DIR,
    epochs: int = 1,
) -> None:
    if len(training_data) < FINETUNE_THRESHOLD:
        print(f"[lora_trainer] Only {len(training_data)} samples — threshold is {FINETUNE_THRESHOLD}. Skipping.")
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as e:
        print(f"[lora_trainer] Missing dependency: {e}. Install transformers, peft, datasets.")
        return

    print(f"[lora_trainer] Starting LoRA fine-tune on {model_name} with {len(training_data)} samples...")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # 4-bit config (bitsandbytes, GPU only)
    load_kwargs: dict = {"low_cpu_mem_usage": True}
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["quantization_config"] = bnb_config
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    texts = [f"{p.instruction}\n{p.response}" for p in training_data]
    tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
    tokenized["weights"] = [p.weight for p in training_data]

    ds = Dataset.from_dict(tokenized)

    def labelize(ex):
        labels = [iid if am == 1 else -100 for iid, am in zip(ex["input_ids"], ex["attention_mask"])]
        return {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"],
                "labels": labels, "weights": ex["weights"]}

    ds = ds.map(labelize, remove_columns=ds.column_names)

    from transformers import Trainer

    class _WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("weights", None)
            outputs = model(**inputs)
            loss = outputs.loss
            if weights is not None:
                loss = loss * torch.mean(weights)
            return (loss, outputs) if return_outputs else loss

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    trainer = _WeightedTrainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"[lora_trainer] Adapter saved to {output_dir}")


def load_adapter(base_model, adapter_path: str):
    try:
        from peft import PeftModel
        return PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"[lora_trainer] Could not load adapter: {e}")
        return base_model


def correction_count_from_data() -> int:
    reflect_dir = "./data/reflection_data"
    if not os.path.isdir(reflect_dir):
        return 0
    return len([f for f in os.listdir(reflect_dir) if f.endswith(".json")])
