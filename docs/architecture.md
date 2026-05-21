# Architecture

## System Overview

The framework consists of 7 loosely coupled modules that form a closed
self-improvement loop. Each module is independently testable.

```
User Input
     │
     ▼
┌─────────────────────┐
│   Adaptive Prompt   │  ← injects memories + corrections + prefs
│      Builder        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│    Local LLM Layer  │  ← Ollama / llama.cpp (quantized, offline)
│  (inference engine) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Reflection Engine  │  ← generate → critique → revise
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Correction Engine  │  ← pattern detection, error logging
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│   Memory Store      │────▶│  Forget Mechanism    │
│  (FAISS + ChromaDB) │     │  (decay + explicit)  │
└────────┬────────────┘     └──────────────────────┘
         │
         ▼
┌─────────────────────┐
│  LoRA Fine-Tuner    │  ← triggered at correction_count > threshold
│  (background job)   │
└─────────────────────┘
```

## Module Responsibilities

### Local LLM Layer (`core/llm/`)
- Detects runtime: Ollama (preferred) → llama.cpp fallback
- Manages sliding context window to avoid token overflow
- Supports any GGUF-quantized model

### Reflection Engine (`core/reflection/`)
- Sends initial response through a self-critique prompt
- Structured critique: checks for factual errors, tone mismatch, repetition, ignored context
- Only triggers revision if critique identifies actionable issues
- Returns: original, critique, revised (if changed), was_corrected flag

### Correction Engine (`core/correction/`)
- Extracts behavioral patterns from reflection results
- Pattern taxonomy: verbosity, tone, repetition, factual_error, ignored_memory
- Confidence-weighted logging — low-confidence patterns don't persist

### Dual-Tier Memory (`core/memory/`)
- **Hot tier (FAISS)**: recent memories, in-memory ANN index, fast retrieval
- **Cold tier (ChromaDB)**: persistent on-disk, older memories promoted from hot
- Promotion triggered when hot tier exceeds size threshold
- Retrieval: searches hot first, supplements from cold if needed

### Forget Mechanism (`core/memory/forget.py`)
- Three forgetting modes:
  - **Explicit**: user command `/forget <id>`
  - **Implicit**: decay scoring — recency × access_frequency × confidence
  - **Contradiction**: new behavioral pattern replaces conflicting old one
- Pruning runs on a background schedule, not blocking inference

### Adaptive Prompt Builder (`core/prompt/`)
- Assembles layered context: system → corrections → memories → history → query
- Active corrections injected as soft rules (not hard constraints)
- Memory injection limited to top-3 by relevance to avoid context bloat

### LoRA Fine-Tuner (`core/finetuning/`)
- Only triggers when correction_count crosses FINETUNE_THRESHOLD (default: 50)
- 4-bit QLoRA with gradient checkpointing (fits in 8GB RAM)
- Target modules: q_proj, v_proj (attention layers only)
- Adapter saved separately — base model untouched

## Data Flow: Single Inference Cycle

```
1. User sends message
2. Short-term buffer adds message to session context
3. Embed message → query hot+cold memory → retrieve top-5
4. Load active corrections for user
5. Build prompt: system + corrections + memories + history + message
6. LLM generates response
7. Reflection engine critiques response
8. If corrected: log pattern to correction engine
9. Embed + store both turns in hot memory
10. Return final response to user
11. (Background) Check if finetune threshold crossed → schedule LoRA job
```
