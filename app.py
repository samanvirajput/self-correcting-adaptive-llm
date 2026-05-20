"""
app.py — Interactive CLI for the self-correcting adaptive LLM.

Commands:
  /forget <id>   — delete a specific memory by id
  /memories      — list all stored memories
  /export        — export all user data to JSON
  /stats         — show session stats
  /quit          — exit
"""
import os
import sys
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from core.llm.loader import load_model, get_model_info
from core.llm.inference import generate, get_session_token_count
from core.embeddings.pipeline import embed_text
from core.reflection.engine import reflect
from core.correction.engine import detect_patterns, log_correction, get_active_corrections
from core.memory.short_term import add_turn, get_context
from core.memory.long_term import LongTermMemory
from core.memory.forget import prune_low_value
from core.prompt.builder import build_prompt
from core.finetuning.lora_trainer import correction_count_from_data, FINETUNE_THRESHOLD
from privacy.manager import export_user_data, list_memories, forget_memory

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USER_ID = os.getenv("USER_ID", "default")
SESSION_ID = "main"


def _prompt_prefix(memory: LongTermMemory) -> str:
    n_corr = len(get_active_corrections(USER_ID))
    n_mem = memory.count()["total"]
    return f"[{n_corr} corrections | {n_mem} memories] > "


def _handle_command(cmd: str, memory: LongTermMemory) -> bool:
    """Handle /commands. Returns True if handled."""
    cmd = cmd.strip()

    if cmd.startswith("/forget "):
        mem_id = cmd[8:].strip()
        ok = forget_memory(USER_ID, mem_id, memory)
        print(f"  {'Deleted' if ok else 'Not found'}: {mem_id}")
        return True

    if cmd == "/memories":
        mems = list_memories(USER_ID, memory)
        if not mems:
            print("  No memories stored yet.")
        for m in mems:
            print(f"  [{m['tier']}] {m['id']} — {m['content']}")
        return True

    if cmd == "/export":
        data = export_user_data(USER_ID, memory)
        out_path = f"./data/export_{USER_ID}.json"
        os.makedirs("./data", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Exported to {out_path}")
        return True

    if cmd == "/stats":
        counts = memory.count()
        tokens = get_session_token_count(SESSION_ID)
        corrections = get_active_corrections(USER_ID)
        n_reflect = correction_count_from_data()
        print(f"  tokens this session : {tokens}")
        print(f"  hot memories        : {counts['hot']}")
        print(f"  cold memories       : {counts['cold']}")
        print(f"  active corrections  : {len(corrections)}")
        print(f"  reflection logs     : {n_reflect} / {FINETUNE_THRESHOLD} (fine-tune threshold)")
        return True

    if cmd in ("/quit", "/exit", "/q"):
        print("Goodbye.")
        sys.exit(0)

    return False


def main():
    print("Loading model...")
    model = load_model()
    info = get_model_info(model)
    print(f"  backend : {info.get('backend')}")
    print(f"  model   : {info.get('model')}")

    print("Loading long-term memory...")
    memory = LongTermMemory(user_id=USER_ID, dim=384)
    counts = memory.count()
    print(f"  hot={counts['hot']}  cold={counts['cold']}")

    print("\nReady. Type /quit to exit.\n")

    while True:
        try:
            user_input = input(_prompt_prefix(memory)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if _handle_command(user_input, memory):
            continue

        # --- 1. Build context ---
        short_ctx = get_context(SESSION_ID, max_turns=5)
        qvec = embed_text(user_input, model_name=EMBEDDING_MODEL)
        retrieved = memory.retrieve(qvec, top_k=5)
        corrections = get_active_corrections(USER_ID)

        # --- 2. Build prompt + generate ---
        prompt = build_prompt(
            query=user_input,
            short_term_context=short_ctx,
            retrieved_memories=retrieved,
            active_corrections=corrections,
        )
        response = generate(prompt, model, max_tokens=512, session_id=SESSION_ID)

        # --- 3. Reflect ---
        result = reflect(user_input, response, model)
        final_response = result.revised if result.was_corrected else response

        print(f"\n{final_response}\n")
        if result.was_corrected:
            print(f"  [self-corrected — {len(result.issues)} issue(s) found]\n")

        # --- 4. Detect + log correction patterns ---
        if result.was_corrected:
            patterns = detect_patterns(result, USER_ID)
            for p in patterns:
                log_correction(p)

        # --- 5. Store in short-term + long-term memory ---
        add_turn(SESSION_ID, "user", user_input)
        add_turn(SESSION_ID, "assistant", final_response)

        mem_content = f"User: {user_input}\nAssistant: {final_response}"
        rvec = embed_text(mem_content, model_name=EMBEDDING_MODEL)
        memory.store(mem_content, rvec, metadata={"type": "conversation"})

        # Periodic pruning (every 20 turns)
        total_turns = len(get_context(SESSION_ID))
        if total_turns % 40 == 0 and total_turns > 0:
            pruned = prune_low_value(USER_ID, memory, threshold=0.2)
            if pruned:
                print(f"  [pruned {pruned} low-value memories]")


if __name__ == "__main__":
    main()
