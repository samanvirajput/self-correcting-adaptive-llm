# Self-Correcting Adaptive LLM

A fully local, privacy-preserving LLM system that learns from every conversation. No cloud. No API keys. All inference, memory, and fine-tuning run on-device.

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │              app.py (CLI)            │
                        │   [N corrections | M memories] >     │
                        └────────────────┬────────────────────┘
                                         │
              ┌──────────────────────────▼───────────────────────────┐
              │  llm/loader  ──▶  llm/inference (sliding ctx window) │
              │  Ollama (auto-detect) │ llama-cpp-python (fallback)   │
              └──────────────────────────┬───────────────────────────┘
                                         │
              ┌──────────────────────────▼───────────────────────────┐
              │          reflection/engine  (3-pass loop)             │
              │    generate ──▶ structured critique ──▶ revise        │
              └──────────────────────────┬───────────────────────────┘
                                         │
              ┌──────────────────────────▼───────────────────────────┐
              │    correction/engine  (CorrectionPattern detection)   │
              │    verbosity · tone · repetition · factual · memory   │
              └──────────────────────────┬───────────────────────────┘
                                         │
              ┌──────────────────────────▼───────────────────────────┐
              │                      memory/                          │
              │   short_term: per-session sliding buffer              │
              │   long_term:  FAISS hot tier + ChromaDB cold tier     │
              │   forget:     decay score · explicit · contradiction  │
              └──────────────────────────┬───────────────────────────┘
                                         │
              ┌──────────────────────────▼───────────────────────────┐
              │  embeddings/pipeline  ──▶  retrieval/retriever        │
              │  sentence-transformers    cosine ANN over both tiers  │
              └──────────────────────────┬───────────────────────────┘
                                         │
              ┌──────────────────────────▼───────────────────────────┐
              │  prompt/builder  (adaptive prompt assembly)           │
              │  system + correction rules + memories + history       │
              └──────────────────────────┬───────────────────────────┘
                                         │
              ┌──────────────────────────▼───────────────────────────┐
              │  finetuning/lora_trainer  (4-bit QLoRA)               │
              │  fires only when correction_count > FINETUNE_THRESHOLD│
              └──────────────────────────┬───────────────────────────┘
                                         │
              ┌──────────────────────────▼───────────────────────────┐
              │  privacy/manager  — 100% local, no network calls      │
              │  export · wipe · list · forget                        │
              └───────────────────────────────────────────────────────┘
```

## Module Reference

| Module | What it does |
|--------|-------------|
| `core/llm/loader.py` | Pings Ollama `:11434`; falls back to llama-cpp-python + GGUF |
| `core/llm/inference.py` | Sliding context window; approx token counting per session |
| `core/reflection/engine.py` | Structured critique → targeted revision loop |
| `core/correction/engine.py` | Extracts `CorrectionPattern` from issues; persists JSON to disk |
| `core/memory/short_term.py` | In-session buffer per session_id; configurable max_turns |
| `core/memory/long_term.py` | FAISS hot (in-memory ANN) + ChromaDB cold (persistent); auto-promotes at 200 entries |
| `core/memory/forget.py` | Decay score (recency + access + confidence); prune + explicit + contradiction forget |
| `core/embeddings/pipeline.py` | sentence-transformers singleton; `embed_text()` + `embed_batch()` |
| `core/retrieval/retriever.py` | Cosine ANN search over both tiers; min_score filter |
| `core/prompt/builder.py` | Assembles: system + active corrections + top-3 memories + last-5 turns + query |
| `core/finetuning/lora_trainer.py` | 4-bit QLoRA, `WeightedTrainer`, reflection-weighted loss; threshold-gated |
| `privacy/manager.py` | export/wipe/list/forget — all local, zero network |

## Quick Start

```bash
# 1. Install Ollama
brew install ollama
ollama serve &
ollama pull llama3

# 2. Clone and install
git clone https://github.com/samanvirajput/self-correcting-adaptive-llm
cd self-correcting-adaptive-llm
pip install -r requirements.txt
cp .env.example .env

# 3. Run CLI
python app.py
```

```
Loading model...
  backend : ollama
  model   : llama3

[0 corrections | 0 memories] > Hello, what can you do?
```

## CLI Commands

| Command | Effect |
|---------|--------|
| `/memories` | List stored memories with tier (hot/cold) and preview |
| `/forget <id>` | Delete specific memory by id |
| `/export` | Dump all user data to `./data/export_<user>.json` |
| `/stats` | Token count, memory counts, corrections, fine-tune progress |
| `/quit` | Exit |

## Per-Turn Pipeline

```
user input
  → embed (sentence-transformers)
  → retrieve top-5 (FAISS hot → ChromaDB cold fallback)
  → build adaptive prompt
  → generate (Ollama / llama-cpp)
  → reflect: critique → revise
  → detect correction patterns → log JSON
  → store in short_term + long_term (hot tier)
  → periodic decay prune (every 20 turns)
```

## Fine-Tuning

Triggers when `correction_count >= FINETUNE_THRESHOLD` (default: 50).

- Source data: `./data/reflection_data/*.json` (auto-generated each time a response is revised)
- Method: **4-bit QLoRA** — `r=8`, `lora_alpha=32`, `target_modules=["q_proj","v_proj"]`
- Loss: reflection-weighted CE (`WeightedTrainer`) — samples weighted by cosine distance between original and corrected response
- Output: adapter saved to `./data/adapters/`

## Hardware

- **Recommended**: Apple Silicon M-series, 8 GB+ unified memory
- **Minimum**: 8 GB RAM for 3B–7B models via Ollama
- **GPU**: bitsandbytes 4-bit works on CUDA; Apple Silicon uses Ollama's Metal backend

## Privacy

All data stays local:
- Inference: Ollama (localhost) or llama-cpp-python (on-disk GGUF)
- Memory: `./data/chroma/` (ChromaDB), in-memory FAISS
- Corrections: `./data/corrections/*.json`
- Reflections: `./data/reflection_data/*.json`
- Zero network calls after initial model download
- `/export` writes a local JSON file — never uploaded

## Paper

arXiv preprint forthcoming — q-fin.ST / stat.AP
