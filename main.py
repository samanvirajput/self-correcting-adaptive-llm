import os, warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*parallelism.*", category=UserWarning)
from backend.llm_engine import LLMEngine
from backend.embeddings import EmbeddingEngine
from backend.vector_memory import VectorMemory
from backend.self_correction import SelfReflection
from backend.context_manager import ContextManager
from backend.forget_manager import ForgetManager
from backend.feedback_logger import FeedbackLogger
from backend.rag_retriever import RAGRetriever
from backend.critic_evaluator import CriticEvaluator




# --- INITIAL SETUP ---
print("🔹 Initializing system...")

# Core components
llm = LLMEngine(model_name="phi3:mini")
embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
memory = VectorMemory(persist_directory="./data/chroma_store")
reflector = SelfReflection()
context_manager = ContextManager()
forget_manager = ForgetManager()
feedback_logger = FeedbackLogger()
rag_retriever = RAGRetriever()


# Rolling chat memory (short-term)
CONTEXT_WINDOW = 6
conversation_history = []

# --- Feedback cadence (ask user every N assistant replies) ---
FEEDBACK_INTERVAL = 5                # change this number to any interval you prefer
prompts_since_feedback = 0           # runtime counter (do NOT persist)



# --- Start background fine-tuning scheduler ---
import threading
from backend.scheduler import FineTuneScheduler

def start_scheduler():
    scheduler = FineTuneScheduler(interval=600)  # every 10 minutes
    t = threading.Thread(target=scheduler.run, daemon=True)
    t.start()

start_scheduler()


print("✅ System ready.\n")
print("🚀 Adaptive Contextual LLM Chat (Type 'exit' to quit)\n")


# --- MAIN LOOP ---
while True:
    try:
        user_input = input("🧠 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # --- Privacy / Forget Commands ---
        if user_input.startswith("/forget "):
            keyword = user_input.replace("/forget ", "").strip()
            forget_manager.forget_reflection(keyword)
            continue

        if user_input.startswith("/wipe"):
            forget_manager.wipe_all()
            continue

        # --- 1️⃣ Retrieve hybrid contextual memory (conversation + documents) ---
        retrieved_context = context_manager.retrieve_context(user_input)
        
        # Try retrieving additional grounded info from your local document stor
        try:
            doc_hits = rag_retriever.search(user_input, top_k=2)
        except Exception as e:
            doc_hits = []
            print(f"⚠️ Document retrieval skipped: {e}")
        context_segments = []
        if retrieved_context.strip():
            context_segments.append(f"[Conversation Memory]\n{retrieved_context}")
        if doc_hits:
            joined_docs = "\n---\n".join(doc_hits)
            context_segments.append(f"[Document Knowledge]\n{joined_docs}")

        if context_segments:
            context_prefix = (
                    "You have access to relevant background information below. "
                    "Use it only for context — do NOT copy it directly.\n\n"
                    + "\n\n".join(context_segments)
                    + "\n\n"
                )
        else:
            context_prefix = ""


        # --- 2️⃣ Build short-term conversation window (no explicit roles) ---
        recent_context = " ".join(
            [msg["content"] for msg in conversation_history[-CONTEXT_WINDOW * 2:]]
            )


        # --- Construct final prompt cleanly ---
        system_note = (
             "You are a conversational assistant. "
             "Respond naturally to the user's latest message. "
             "Do NOT narrate or invent new instructions. "
             "Ignore unrelated or old context text.\n\n"
             )
        # Compact past dialogue into readable natural context
        recent_context = " ".join(
             [msg["content"] for msg in conversation_history[-CONTEXT_WINDOW * 2:]]
             )
        
        # Clean prompt: background info, chat, then user request
        full_prompt = (
            f"{system_note}"
            f"{context_prefix}"
            f"Previous context (for background only): {recent_context}\n\n"
            f"User: {user_input}\nAssistant:"
        )





        # --- 4️⃣ Generate model response (extended context) ---
        response = llm.generate(full_prompt, max_new_tokens=800)
        print(f"\n🤖 LLM: {response}\n")

        # --- 5️⃣ Save this turn to long-term and context memory ---
        context_manager.save_interaction(user_input, response)
        context_manager.summarize_history()  # summarizes every ~10 turns

        # Add assistant response to in-memory rolling window
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})

        # --- 🔍 Critic Evaluation ---
        # --- 🔍 Critic Evaluation & Auto-Regeneration ---
        critic = CriticEvaluator()
        critic_result = critic.evaluate(user_input, response, retrieved_context)

        print(
            f"🧩 Critic Scores → "
            f"Factuality={critic_result['factuality']:.2f}, "
            f"Helpfulness={critic_result['helpfulness']:.2f}, "
            f"Linguistic={critic_result['linguistic_quality']:.2f}, "
            f"Overall={critic_result['critic_score']:.2f}"
        )

        # Compute reflection weight
        reflection_weight = round(max(0.0, min(1.0, 1 - critic_result["critic_score"])),3)


        # ✅ If the answer is poor, self-reflect and regenerate once
        if reflection_weight > 0.6:
            print("🤔 Low critic score detected — initiating self-reflection and regeneration...")

            regen_prompt = (
                f"The previous answer scored poorly (critic={critic_result['critic_score']:.2f}). "
                f"Reflect on how to improve clarity, factual accuracy, and helpfulness. "
                f"Rewrite your response to the user query below, making it more accurate and natural.\n\n"
                f"User: {user_input}\n"
                f"Previous Response: {response}\n"
                f"Improved Assistant Response:"
            )

            improved_response = llm.generate(regen_prompt, max_new_tokens=800)
            print(f"\n🤖 Self-Improved Response: {improved_response}\n")

            # Log both old and new versions
            reflector.log_feedback(
                query=user_input,
                wrong_response=response,
                correct_response=improved_response,
            )

            # Replace response for display & memory
            response = improved_response
            print(f"✅ Auto-reflection applied with weight {reflection_weight:.3f}\n")

        else:
            # Normal logging (no regeneration)
            reflector.log_feedback(
                query=user_input,
                wrong_response=response,
                correct_response=f"[AUTO-REFLECTED] Critic Score={critic_result['critic_score']:.2f}",
            )
            print(f"✅ Logged auto-reflection with weight {reflection_weight:.3f}\n")




        # --- 6️⃣ Feedback & reflection (asked every FEEDBACK_INTERVAL replies) ---
        prompts_since_feedback += 1

# Allow user to force feedback immediately via command
        if user_input.strip().startswith("/feedback"):
            force_feedback = True
        else:
            force_feedback = False

        if prompts_since_feedback >= FEEDBACK_INTERVAL or force_feedback:
            # Reset counter if we actually prompt
            prompts_since_feedback = 0

            feedback = input("👍 Was this response helpful? (y/n): ").strip().lower()
            if feedback == "y":
                feedback_logger.log_feedback(user_input, response, reward=+1)
                print("✅ Great! Logged positive reward.\n")
            elif feedback == "n":
                correct_response = input("💡 Please enter your corrected or ideal response:\n> ").strip()
                reflector.log_feedback(user_input, response, correct_response)
                feedback_logger.log_feedback(user_input, response, reward=-1)
                print("✅ Logged correction and negative reward.\n")
            else:
                # user pressed enter or something else — skip politely
                print("ℹ️ Skipping feedback.\n")
        else:
            # Not asking this turn — continue silently
            # (optional) print a small hint every few prompts — currently we stay silent
            pass



    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully.")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")
