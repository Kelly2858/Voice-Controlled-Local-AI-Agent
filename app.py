"""
Memo — Voice-Controlled Local AI Agent.

A premium voice AI agent that transcribes speech locally using HuggingFace Whisper,
classifies intent with Ollama, and executes actions — all running on your machine.

Run with:  streamlit run app.py
"""

import streamlit as st
from config import OLLAMA_MODEL, OUTPUT_DIR, WHISPER_MODEL
from stt import transcribe_audio, get_whisper_model_name
from intent import classify_intent, get_available_models
from tools import execute_intent
from ui_components import (
    render_pipeline_step,
    render_step_connector,
    render_result_card,
    render_history_item,
    render_confirmation_dialog,
    render_transcript_box,
    render_intent_badge,
)

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Memo — Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Animations ────────────────────────────────────── */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ── Hero Header ───────────────────────────────────── */
    .hero-container {
        text-align: center;
        padding: 20px 0 10px 0;
        animation: fadeSlideIn 0.6s ease-out;
    }

    .hero-icon {
        font-size: 3rem;
        margin-bottom: 4px;
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
    }

    .main-title {
        background: linear-gradient(135deg, #8B5CF6 0%, #A78BFA 30%, #C084FC 60%, #E879F9 100%);
        background-size: 200% 200%;
        animation: gradientShift 4s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }

    .sub-title {
        color: #64748B;
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 24px;
        letter-spacing: 0.02em;
    }

    /* ── Sidebar ───────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1120 0%, #0F172A 100%);
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* ── Button styling ────────────────────────────────── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.01em;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Primary button glow */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%);
        border: none;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25);
    }

    /* ── Audio input ───────────────────────────────────── */
    .stAudioInput > div {
        border-radius: 12px;
        border: 1px solid #334155;
    }

    /* ── Tab styling ───────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #0F172A;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #1E293B;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* ── Divider ───────────────────────────────────────── */
    .divider {
        border: none;
        border-top: 1px solid #1E293B;
        margin: 20px 0;
    }

    /* ── Status indicator ──────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s ease-in-out infinite;
    }

    .status-dot.active { background: #10B981; box-shadow: 0 0 8px #10B98180; }
    .status-dot.inactive { background: #EF4444; }
    .status-dot.warning { background: #F59E0B; }

    /* ── Card containers ───────────────────────────────── */
    .glass-card {
        background: linear-gradient(135deg, #1E293B60 0%, #0F172A80 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: #475569;
    }

    /* ── Processing indicator ──────────────────────────── */
    .processing-bar {
        height: 3px;
        background: linear-gradient(90deg, transparent, #8B5CF6, #C084FC, #8B5CF6, transparent);
        background-size: 200% 100%;
        animation: shimmer 1.5s linear infinite;
        border-radius: 2px;
        margin: 8px 0;
    }

    /* ── Hide default Streamlit elements ────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State ────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "action_history" not in st.session_state:
    st.session_state.action_history = []
if "pending_intent" not in st.session_state:
    st.session_state.pending_intent = None
if "pending_audio_text" not in st.session_state:
    st.session_state.pending_audio_text = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 16px 0 8px 0;">
            <span style="font-size: 2rem;">🎙️</span>
            <div style="
                font-size: 1.2rem;
                font-weight: 700;
                background: linear-gradient(135deg, #8B5CF6, #C084FC);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-top: 4px;
            ">Memo</div>
            <div style="
                font-size: 0.65rem;
                color: #475569;
                margin-top: 2px;
                letter-spacing: 0.05em;
            ">100% LOCAL · NO API KEYS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Models")

    # ── Whisper STT Model ────────────────────────────────────────────────
    whisper_models = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3",
    ]
    default_whisper_idx = 0
    for i, m in enumerate(whisper_models):
        if WHISPER_MODEL == m:
            default_whisper_idx = i
            break

    selected_whisper = st.selectbox(
        "🎤 Speech-to-Text Model",
        whisper_models,
        index=default_whisper_idx,
        help="HuggingFace Whisper model for local transcription. Larger models are more accurate but slower.",
    )

    # Show model size hint
    size_hints = {
        "openai/whisper-tiny": "~40 MB · Fastest",
        "openai/whisper-base": "~75 MB · Fast",
        "openai/whisper-small": "~250 MB · Balanced",
        "openai/whisper-medium": "~800 MB · Accurate",
        "openai/whisper-large-v3": "~1.6 GB · Best (GPU recommended)",
    }
    st.caption(f"📦 {size_hints.get(selected_whisper, '')}")

    st.markdown(
        '<div><span class="status-dot active"></span> <span style="color:#94A3B8; font-size:0.85rem;">Whisper: Local Model</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("", unsafe_allow_html=True)

    # ── Ollama LLM Model ────────────────────────────────────────────────
    available_models = get_available_models()
    if available_models:
        default_idx = 0
        for i, m in enumerate(available_models):
            if OLLAMA_MODEL in m:
                default_idx = i
                break
        selected_model = st.selectbox(
            "🤖 LLM Model (Ollama)",
            available_models,
            index=default_idx,
            help="Local Ollama model for intent classification and execution.",
        )
        st.markdown(
            '<div><span class="status-dot active"></span> <span style="color:#94A3B8; font-size:0.85rem;">Ollama: Running</span></div>',
            unsafe_allow_html=True,
        )
    else:
        selected_model = st.text_input(
            "🤖 LLM Model (Ollama)",
            value=OLLAMA_MODEL,
            help="Ollama not detected. Enter model name manually.",
        )
        st.markdown(
            '<div><span class="status-dot inactive"></span> <span style="color:#94A3B8; font-size:0.85rem;">Ollama: Not detected</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Run `ollama serve` in a terminal")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Output folder info
    st.markdown(
        f"""
        <div style="
            background: #1E293B40;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 0.8rem;
        ">
            <span style="color: #64748B;">📂 Output folder</span><br>
            <code style="color: #94A3B8; font-size: 0.75rem;">{OUTPUT_DIR}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Session history
    st.markdown("### 📜 History")
    if st.session_state.action_history:
        for idx, item in enumerate(reversed(st.session_state.action_history)):
            render_history_item(item, len(st.session_state.action_history) - 1 - idx)
    else:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 20px;
                color: #475569;
                font-size: 0.82rem;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">🎤</div>
                No actions yet.<br>Record or upload audio to get started.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.action_history = []
        st.session_state.pending_intent = None
        st.session_state.pending_audio_text = None
        st.session_state.last_results = None
        st.session_state.processing_complete = False
        st.rerun()

# ── Main Content ─────────────────────────────────────────────────────────────

# Hero header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-icon">🎙️</div>
        <h1 class="main-title">Memo</h1>
        <p class="sub-title">Speak a command. Watch it execute. 100% local — powered by HuggingFace Whisper + Ollama.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Audio Input ──────────────────────────────────────────────────────────────

tab_record, tab_upload = st.tabs(["🎤 Record", "📁 Upload"])

audio_bytes = None
audio_filename = "audio.wav"

with tab_record:
    st.markdown(
        """
        <div style="
            color: #64748B;
            font-size: 0.82rem;
            margin-bottom: 8px;
            text-align: center;
        ">Click the microphone to start recording your voice command</div>
        """,
        unsafe_allow_html=True,
    )
    recorded_audio = st.audio_input(
        "Record your voice command",
        key="mic_input",
        label_visibility="collapsed",
    )
    if recorded_audio:
        audio_bytes = recorded_audio.getvalue()
        audio_filename = "recording.wav"

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a .wav, .mp3, .ogg, or .webm file",
        type=["wav", "mp3", "ogg", "webm", "m4a", "flac"],
        key="file_upload",
    )
    if uploaded_file:
        audio_bytes = uploaded_file.getvalue()
        audio_filename = uploaded_file.name

# ── Process Button ───────────────────────────────────────────────────────────

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    col_l, col_center, col_r = st.columns([1, 2, 1])
    with col_center:
        process_clicked = st.button(
            "🚀 Process Audio",
            type="primary",
            use_container_width=True,
        )

    if process_clicked:
        st.session_state.processing_complete = False
        st.session_state.pending_intent = None
        st.session_state.last_results = None

        # ── Step 1: Transcription ────────────────────────────────────────
        with st.status("🎙️ Processing your voice command...", expanded=True) as status:
            st.write(f"📝 **Step 1:** Transcribing with **{selected_whisper}** (local)...")
            st.caption("First run downloads the model — this may take a moment.")

            stt_result = transcribe_audio(audio_bytes, selected_whisper, audio_filename)

            if not stt_result["success"]:
                st.error(f"❌ Transcription failed: {stt_result['error']}")
                status.update(label="❌ Transcription failed", state="error")
                st.stop()

            transcription = stt_result["text"]
            st.write(f"✅ Transcribed: *\"{transcription}\"*")

            # ── Step 2: Intent Classification ────────────────────────────
            st.write(f"🧠 **Step 2:** Classifying intent with **{selected_model}** (Ollama)...")

            intent_result = classify_intent(
                transcription,
                selected_model,
                st.session_state.chat_history,
            )

            if not intent_result["success"]:
                st.write(f"⚠️ Classification issue: {intent_result['error']}")
                if intent_result["intent"] != "general_chat":
                    status.update(label="❌ Intent classification failed", state="error")
                    st.stop()

            intent_label = intent_result["intent"].replace("_", " ").title()
            st.write(f"✅ Detected: **{intent_label}** ({intent_result.get('confidence', 'medium')} confidence)")

            status.update(label="✅ Audio processed successfully", state="complete")

        # ── Display Results Below Status ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        render_transcript_box(transcription)
        render_intent_badge(
            intent_result["intent"],
            intent_result.get("confidence", "medium"),
            intent_result.get("reasoning", ""),
        )

        # ── Step 3: Confirmation or Execute ──────────────────────────────
        needs_confirmation = intent_result["intent"] in ("create_file", "write_code", "compound")

        if needs_confirmation:
            st.session_state.pending_intent = intent_result
            st.session_state.pending_audio_text = transcription
        else:
            # Execute immediately for non-file operations
            with st.spinner("⚙️ Executing action..."):
                results = execute_intent(
                    intent_result,
                    selected_model,
                    st.session_state.chat_history,
                )

            st.session_state.last_results = results
            st.session_state.processing_complete = True

            # Update chat & action history
            st.session_state.chat_history.append({"role": "user", "content": transcription})
            for r in results:
                st.session_state.chat_history.append({"role": "assistant", "content": r.get("output", "")})

            st.session_state.action_history.append({
                "transcription": transcription,
                "intent": intent_result["intent"],
                "confidence": intent_result.get("confidence", ""),
                "results": results,
            })

            # Display results
            st.markdown("<br>", unsafe_allow_html=True)
            render_pipeline_step("⚙️", "Execution", "Action completed successfully.", "success")
            for r in results:
                render_result_card(r)

# ── Pending Confirmation Flow ────────────────────────────────────────────────

if st.session_state.pending_intent and not st.session_state.processing_complete:
    intent_result = st.session_state.pending_intent
    transcription = st.session_state.pending_audio_text

    confirmation = render_confirmation_dialog(intent_result)

    if confirmation is True:
        with st.spinner("⚙️ Executing confirmed action..."):
            results = execute_intent(
                intent_result,
                selected_model,
                st.session_state.chat_history,
            )

        st.session_state.last_results = results
        st.session_state.processing_complete = True
        st.session_state.pending_intent = None

        # Update histories
        st.session_state.chat_history.append({"role": "user", "content": transcription})
        for r in results:
            st.session_state.chat_history.append({"role": "assistant", "content": r.get("output", "")})

        st.session_state.action_history.append({
            "transcription": transcription,
            "intent": intent_result["intent"],
            "confidence": intent_result.get("confidence", ""),
            "results": results,
        })

        render_pipeline_step("⚙️", "Execution", "Action completed successfully.", "success")
        for r in results:
            render_result_card(r)

    elif confirmation is False:
        st.session_state.pending_intent = None
        st.session_state.processing_complete = False
        st.info("❌ Action cancelled by user.")

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="
        text-align: center;
        color: #334155;
        font-size: 0.75rem;
        padding: 8px 0 16px 0;
        letter-spacing: 0.02em;
    ">
        <strong style="color: #475569;">Memo</strong> — Voice AI Agent ·
        100% Local · <span style="color: #475569;">HuggingFace Whisper</span> +
        <span style="color: #475569;">Ollama</span> ·
        Files saved to <code style="color: #64748B;">output/</code>
    </div>
    """,
    unsafe_allow_html=True,
)
