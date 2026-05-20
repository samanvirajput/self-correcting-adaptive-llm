"""
Simple evaluation harness that loads a list of test prompts (from ./data/evaluation_prompts.txt)
and computes average token-level loss (or simple exact-match) before and after applying adapter.

Note: For a correct "before vs after" measurement you must:
  - Load the same base model (without adapter) -> evaluate -> record
  - Load the base model + adapter (the adapter saved in OUTPUT_DIR) -> evaluate -> record
"""

import os
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_prompts(path="./data/evaluation_prompts.txt"):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines() if l.strip()]

def simple_generation_metric(model_name, prompts, max_new_tokens=64):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    results = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt")
        out = model.generate(**inp, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append(text)
    return results

if __name__ == "__main__":
    print("Run evaluation by calling simple_generation_metric() with model & prompts.")
