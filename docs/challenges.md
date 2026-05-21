# Challenges

## Distinguishing noise from signal in corrections
The hardest problem: a user might phrase something awkwardly once (noise)
or consistently dislike a behavior (signal). Early versions logged every
reflection output as a correction, polluting the behavioral model.
Solution: confidence thresholding — patterns below 0.6 confidence are 
logged but not injected into prompts until confirmed by 3+ occurrences.

## Context window overflow
With memories + corrections + history all injected into a single prompt,
context overflow was a constant problem — especially with 2K-token models.
Solution: strict token budgeting in the prompt builder. Each section has
a max token allocation: corrections (150), memories (300), history (400),
leaving the majority for the query and response.

## FAISS index persistence
FAISS in-memory indices are lost on process exit. Early versions lost
all hot-tier memories on restart, defeating the personalization goal.
Solution: periodic FAISS index serialization to disk (every N writes),
with hot→cold promotion ensuring ChromaDB always has a complete copy.

## LoRA adapter compatibility
Loading a LoRA adapter trained on one quantization level onto a 
differently-quantized base model causes silent degradation.
Solution: adapter metadata stores base model hash — mismatches raise 
an explicit error rather than loading silently.

## Reflection loop latency
Reflection adds a full LLM call (~1.8s) to every turn, doubling 
perceived latency. Users found this unacceptable for casual exchanges.
Solution: reflection is now gated — only triggered for responses above
a length threshold (>150 tokens) or when the query contains explicit
reasoning markers (why, explain, compare).
