# Design Decisions

## Why local-first inference?
Cloud LLMs introduce three problems for an adaptive personal AI:
privacy (sensitive behavioral data sent externally), latency (network 
round-trips break conversational flow), and dependency (no offline use).
Local inference with Ollama/llama.cpp eliminates all three. The tradeoff
is capability ceiling — mitigated by the adaptation layer, which makes
a smaller model smarter for a specific user over time.

## Why LoRA/QLoRA over full fine-tuning?
Full fine-tuning a 2B parameter model requires ~16GB VRAM minimum.
LoRA fine-tunes only low-rank adapter matrices injected into attention 
layers (q_proj, v_proj), reducing trainable parameters by ~99% while 
preserving most of the adaptation signal. QLoRA adds 4-bit quantization 
of the base model during training, bringing peak memory to ~4–6GB —
viable on Apple Silicon M-series with 8GB RAM.

## Why FAISS + ChromaDB (dual-tier) instead of one vector DB?
Single vector DBs force a tradeoff: FAISS is fast but in-memory (lost 
on restart); ChromaDB is persistent but slower for hot-path retrieval.
The dual-tier design gets both: FAISS handles the last N interactions 
(fast, relevant), ChromaDB handles long-term semantic memory (persistent,
comprehensive). Hot→cold promotion on a size threshold keeps the hot 
tier lean.

## Why separate reflection from correction?
Reflection is stateless — it evaluates a single response in isolation.
Correction is stateful — it builds a behavioral model over time.
Keeping them separate means reflection can run without correction storage,
correction patterns can be seeded from sources other than reflection,
and each can be tested independently.

## Why a forget mechanism?
Without forgetting, two failure modes emerge: memory pollution (old 
incorrect corrections persist) and personalization staleness (user 
preferences change but system doesn't). The three-mode forget system 
(explicit, decay, contradiction) handles all cases. Decay scoring uses
recency × access_frequency × confidence so frequently accessed, 
high-confidence recent memories survive longest.

## Why threshold-gated fine-tuning?
Fine-tuning on every correction is computationally wasteful and risks
overfitting to noise. The threshold (default: 50 corrections) ensures
the training dataset is large enough to generalize, and the background
job design means fine-tuning never blocks inference.

## Why sentence-transformers (all-MiniLM-L6-v2) for embeddings?
At 22M parameters and 768 dimensions, it runs in ~10ms per embedding 
on CPU, uses ~90MB RAM, and produces competitive semantic similarity 
scores. The alternative (using the LLM itself for embeddings) would 
add ~500ms latency per memory operation — unacceptable for real-time 
conversation.
