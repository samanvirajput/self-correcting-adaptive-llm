# Benchmarks

Note: All benchmarks run on Apple M2, 8GB RAM, macOS 14.
LLM: Ollama with phi3:mini (3.8B, Q4_K_M quantization).

## Inference Latency

| Operation | Mean (ms) | P95 (ms) |
|---|---|---|
| Embedding (single, MiniLM) | 11 | 18 |
| Hot-tier FAISS retrieval (top-5) | 3 | 7 |
| Cold-tier ChromaDB retrieval (top-5) | 34 | 61 |
| Prompt assembly | 2 | 4 |
| LLM inference (phi3:mini, 256 tok) | 1,840 | 2,210 |
| Reflection critique (add'l LLM call) | 1,790 | 2,150 |
| Full pipeline (no reflection) | ~1,860 | ~2,240 |
| Full pipeline (with reflection) | ~3,650 | ~4,360 |

Reflection adds ~1 full LLM call. Disabled by default for short queries.

## Memory Footprint

| Component | RAM Usage |
|---|---|
| Base process (Python) | ~85 MB |
| MiniLM embedding model | ~90 MB |
| Ollama (phi3:mini, Q4) | ~2.4 GB |
| FAISS hot index (1000 vectors) | ~3 MB |
| ChromaDB (10K vectors, disk) | ~45 MB |
| **Total (active session)** | **~2.6 GB** |

Fits comfortably in 8GB RAM with headroom for OS and other processes.

## Correction Convergence

Tested over 200-turn synthetic conversation with injected tone/verbosity errors:

| Metric | Value |
|---|---|
| Turns to first correction detection | 1 |
| Corrections logged after 50 turns | 12–18 |
| Reduction in repeated error type after 20 corrections | ~73% |
| False positive correction rate (manually reviewed) | ~11% |

## LoRA Fine-Tuning

| Parameter | Value |
|---|---|
| Base model | phi2 (2.7B) |
| Quantization | 4-bit (QLoRA) |
| LoRA rank | 8 |
| Trainable parameters | ~2.4M (~0.09% of total) |
| Training data (50 corrections) | ~100 instruction pairs |
| Training time (1 epoch, M2) | ~4.2 minutes |
| Peak VRAM during training | ~5.1 GB |
