import streamlit as st
import os
from backend.llm_engine import LLMEngine
from backend.context_manager import ContextManager
from backend.self_correction import SelfReflection
from backend.embeddings import EmbeddingEngine
from backend.vector_memory import VectorMemory
from backend.critic_evaluator import CriticEvaluator
from backend.forget_manager import ForgetManager
from backend.feedback_logger import FeedbackLogger
from backend.rag_retriever import RAGRetriever

# ==========================================================
# INITIALIZE BACKEND SYSTEMS
# ==========================================================
@st.cache_resource
def init_system():
    llm = LLMEngine(model_name="phi3:mini")
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    memory = VectorMemory(persist_directory="./data/chroma_store")
    context_manager = ContextManager()
    reflector = SelfReflection()
    critic = CriticEvaluator()
    forget_manager = ForgetManager()
    feedback_logger = FeedbackLogger()
    rag = RAGRetriever()
    return llm, embedder, memory, context_manager, reflector, critic, forget_manager, feedback_logger, rag


llm, embedder, memory, context_manager, reflector, critic, forget_manager, feedback_logger, rag = init_system()

# ==========================================================
# PAGE CONFIG & CSS
# ==========================================================
st.set_page_config(page_title="Adaptive Local LLM", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    body, .main { background-color: #0e1117; color: #f0f0f0; font-family: 'Inter', sans-serif; }
    header, footer { visibility: hidden; }

    .user-bubble {
        background-color: #1f1f29;
        padding: 0.9rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #3a7bd5;
    }
    .assistant-bubble {
        background-color: #252532;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border-left: 3px solid #ff4b4b;
        margin: 0.7rem 0 1.2rem 0;
    }

    .block-container h1, h2, h3, h4 { color: #ff4b4b; font-weight: 600; }
    code {
        background-color: #1b1f27;
        color: #ffcc66;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    pre {
        background-color: #1b1f27;
        color: #ffcc66;
        padding: 8px;
        border-radius: 6px;
        overflow-x: auto;
    }
    ul, ol { margin-left: 1.2rem; }

    .metric-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.3rem;
        margin-bottom: 1rem;
    }
    .metric-badge {
        font-size: 0.8rem;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
    }
    .fact { background-color: #007acc33; color: #66ccff; }
    .help { background-color: #00993333; color: #33ff88; }
    .ling { background-color: #ffcc0033; color: #ffcc33; }
    .overall { background-color: #ff444433; color: #ff7777; }
    .reflect { background-color: #6633ff33; color: #aa88ff; }

    section[data-testid="stSidebar"] {
        background-color: #161a1f;
        border-right: 1px solid #2b2b36;
    }

    .stChatInput textarea {
        background-color: #1c1f25 !important;
        color: #f0f0f0 !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption("Manage model behavior and data.")
    if st.button("🧹 Forget All Memory"):
        forget_manager.wipe_all()
        st.success("All memory cleared.")

    uploaded_file = st.file_uploader("📄 Upload Document", type=["pdf", "docx", "txt"])
    if uploaded_file:
        temp_path = f"./user_docs/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

# ==========================================================
# HEADER
# ==========================================================
st.markdown("## ◆ Adaptive Self-Learning LLM Interface")
st.caption("Built for full offline personalization and continuous self-reflection.")

# ==========================================================
# CHAT INPUT HANDLING
# ==========================================================
user_input = st.chat_input("Type your message or command...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    st.markdown(f"<div class='user-bubble'>◉ {user_input}</div>", unsafe_allow_html=True)

    # Forget command
    if user_input.startswith("/forget "):
        keyword = user_input.replace("/forget ", "").strip()
        forget_manager.forget_reflection(keyword)
        st.markdown(f"<div class='assistant-bubble'>■ Forgotten entries containing '{keyword}'.</div>", unsafe_allow_html=True)
    else:
        retrieved_context = context_manager.retrieve_context(user_input)
        full_prompt = f"Context: {retrieved_context}\n\nUser: {user_input}\nAssistant:"

        with st.spinner("Thinking..."):
            response = llm.generate(full_prompt, max_new_tokens=800)

        # Render formatted markdown
        st.markdown("<div class='assistant-bubble'>◆</div>", unsafe_allow_html=True)
        st.markdown(response, unsafe_allow_html=False)
        st.session_state.history.append({"role": "assistant", "content": response})

        # Critic evaluation
        critic_result = critic.evaluate(user_input, response, retrieved_context)
        reflection_weight = round(max(0.0, min(1.0, 1 - critic_result["critic_score"])), 3)

        st.markdown(
            f"<div class='metric-container'>"
            f"<span class='metric-badge fact'>Factual: {critic_result['factuality']:.2f}</span>"
            f"<span class='metric-badge help'>Helpful: {critic_result['helpfulness']:.2f}</span>"
            f"<span class='metric-badge ling'>Linguistic: {critic_result['linguistic_quality']:.2f}</span>"
            f"<span class='metric-badge overall'>Overall: {critic_result['critic_score']:.2f}</span>"
            f"<span class='metric-badge reflect'>Reflect: {reflection_weight:.2f}</span>"
            f"</div>", unsafe_allow_html=True
        )

        reflector.log_feedback(user_input, response, f"[AUTO-REFLECTED] Critic={critic_result['critic_score']:.2f}")

        # ==================================================
        # SELF-REFLECTION LOOP (with new critic evaluation)
        # ==================================================
        if reflection_weight > 0.6:
            st.warning("Low critic score — regenerating improved response...")
            regen_prompt = (
                f"The previous answer was weak (score={critic_result['critic_score']:.2f}). "
                f"Rewrite it to be clearer, more accurate, and engaging.\n\n"
                f"User: {user_input}\nOld Response: {response}\nImproved Response:"
            )
            improved = llm.generate(regen_prompt, max_new_tokens=800)

            # Display regenerated response
            st.markdown("<div class='assistant-bubble'>◇</div>", unsafe_allow_html=True)
            st.markdown(improved, unsafe_allow_html=False)

            # Evaluate improved response
            improved_score = critic.evaluate(user_input, improved, retrieved_context)
            improved_reflect = round(max(0.0, min(1.0, 1 - improved_score["critic_score"])), 3)

            # Display improved metrics
            st.markdown(
                f"<div class='metric-container'>"
                f"<span class='metric-badge fact'>Factual: {improved_score['factuality']:.2f}</span>"
                f"<span class='metric-badge help'>Helpful: {improved_score['helpfulness']:.2f}</span>"
                f"<span class='metric-badge ling'>Linguistic: {improved_score['linguistic_quality']:.2f}</span>"
                f"<span class='metric-badge overall'>Overall: {improved_score['critic_score']:.2f}</span>"
                f"<span class='metric-badge reflect'>Reflect: {improved_reflect:.2f}</span>"
                f"</div>", unsafe_allow_html=True
            )

            reflector.log_feedback(user_input, response, improved)
            st.success("✅ Self-reflection applied and re-evaluated!")

# ==========================================================
# DISPLAY CHAT STREAM
# ==========================================================
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>◉ {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'>◆</div>", unsafe_allow_html=True)
        st.markdown(msg["content"], unsafe_allow_html=False)
