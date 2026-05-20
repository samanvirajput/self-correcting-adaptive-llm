# Self-Correcting Adaptive LLM Framework

> Offline-first LLM with LoRA/QLoRA fine-tuning and a generate → critique → revise self-reflection loop. arXiv paper forthcoming (q-fin.ST / stat.AP).

## Overview
An end-to-end framework for running and fine-tuning LLMs entirely offline, with a built-in self-reflection loop that iteratively improves outputs without any API dependency.

## Technical Details
- **Vector store**: FAISS (hot tier) + ChromaDB (cold tier) dual retrieval architecture
- **Fine-tuning**: LoRA / QLoRA pipeline with 4-bit quantisation and gradient checkpointing
- **Self-reflection loop**: generate → critique → revise → output
- **Offline-first**: zero API dependency — runs fully local
- **Paper**: arXiv submission forthcoming — q-fin.ST / stat.AP

## Stack
`Python` `FAISS` `ChromaDB` `LoRA` `QLoRA` `HuggingFace Transformers`
