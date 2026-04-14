
import streamlit as st


def render_pipeline_step(icon: str, title: str, content: str, status: str = "success"):
    color_map = {
        "success": ("#10B981", "#064E3B"),
        "error":   ("#EF4444", "#7F1D1D"),
        "info":    ("#8B5CF6", "#4C1D95"),
        "warning": ("#F59E0B", "#78350F"),
    }
    accent, bg = color_map.get(status, color_map["info"])

    st.markdown(
        f"""
        <div style="
            background: {bg}18;
            border-left: 3px solid {accent};
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 10px;
            backdrop-filter: blur(10px);
            animation: fadeSlideIn 0.4s ease-out;
        ">
            <div style="
                font-size: 0.72rem;
                font-weight: 700;
                color: {accent};
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 5px;
                display: flex;
                align-items: center;
                gap: 6px;
            ">{icon} {title}</div>
            <div style="
                color: #CBD5E1;
                font-size: 0.9rem;
                line-height: 1.6;
            ">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_connector():
    st.markdown(
        """
        <div style="
            width: 2px;
            height: 12px;
            background: linear-gradient(180deg, #334155 0%, #475569 100%);
            margin: 0 auto;
            border-radius: 1px;
        "></div>
        """,
        unsafe_allow_html=True,
    )


def render_code_result(code: str, language: str, file_path: str | None = None):
    if file_path:
        st.markdown(
            f"""
            <div style="
                background: #064E3B20;
                border: 1px solid #10B98140;
                border-radius: 8px;
                padding: 10px 14px;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 8px;
            ">
                <span style="font-size: 1.1rem;">📁</span>
                <span style="color: #6EE7B7; font-size: 0.85rem; font-weight: 500;">
                    Saved to: <code style="background: #064E3B40; padding: 2px 6px; border-radius: 4px;">{file_path}</code>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.code(code, language=language.lower() if language else "text")


def render_result_card(result: dict):
    if result["success"]:
        status = "success"
        icon = "✅"
    else:
        status = "error"
        icon = "❌"

    action = result.get("action", "Action")
    details = result.get("details", "")
    detail_html = ""
    if details:
        detail_html = f'<br><span style="color:#94A3B8; font-size:0.8rem;">{details}</span>'

    render_pipeline_step(icon, action, detail_html, status)

    if result.get("file_path") and result.get("output") and result["output"] != "(empty file)":
        ext = result["file_path"].rsplit(".", 1)[-1] if "." in result["file_path"] else "text"
        lang_map = {
            "py": "python", "js": "javascript", "ts": "typescript",
            "java": "java", "c": "c", "cpp": "cpp", "go": "go",
            "rs": "rust", "html": "html", "css": "css", "sh": "bash",
            "txt": "text", "md": "markdown", "json": "json",
            "rb": "ruby", "php": "php", "swift": "swift", "kt": "kotlin",
            "r": "r", "sql": "sql",
        }
        lang = lang_map.get(ext, "text")
        render_code_result(result["output"], lang, result["file_path"])
    elif result.get("output"):
        st.markdown(result["output"])


def render_history_item(item: dict, index: int):
    intent_icons = {
        "create_file": "📄",
        "write_code": "💻",
        "summarize": "📋",
        "general_chat": "💬",
        "compound": "🔗",
    }
    icon = intent_icons.get(item.get("intent", ""), "⚡")
    confidence = item.get("confidence", "medium")

    conf_colors = {
        "high": "#10B981",
        "medium": "#F59E0B",
        "low": "#EF4444",
    }
    conf_color = conf_colors.get(confidence, "#94A3B8")

    transcript_preview = item.get("transcription", "")[:45]
    if len(item.get("transcription", "")) > 45:
        transcript_preview += "…"

    intent_label = item.get("intent", "unknown").replace("_", " ").title()

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 8px;
            border: 1px solid #334155;
            transition: all 0.2s ease;
        ">
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 6px;
            ">
                <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">
                    #{index + 1} · {icon} {intent_label}
                </span>
                <span style="
                    font-size: 0.6rem;
                    color: {conf_color};
                    background: {conf_color}15;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-weight: 600;
                    text-transform: uppercase;
                ">{confidence}</span>
            </div>
            <div style="
                font-size: 0.82rem;
                color: #94A3B8;
                font-style: italic;
            ">"{transcript_preview}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confirmation_dialog(intent_result: dict) -> bool | None:
    intent = intent_result.get("intent", "")
    params = intent_result.get("parameters", {})

    if intent == "create_file":
        filename = params.get("filename", "untitled.txt")
        content_preview = params.get("content", "")[:200]
        msg = f"📄 **Create file:** `{filename}`"
        if content_preview:
            msg += f"\n\n> Content preview: _{content_preview}_"
    elif intent == "write_code":
        msg = (
            f"💻 **Generate code:**\n\n"
            f"- **File:** `{params.get('filename', 'generated.py')}`\n"
            f"- **Language:** {params.get('language', 'python')}\n"
            f"- **Task:** {params.get('description', 'N/A')}"
        )
    elif intent == "compound":
        sub_intents = params.get("sub_intents", [])
        lines = []
        for i, s in enumerate(sub_intents, 1):
            icon = {"create_file": "📄", "write_code": "💻", "summarize": "📋"}.get(s.get("intent", ""), "⚡")
            lines.append(f"{i}. {icon} **{s.get('intent', '?').replace('_', ' ').title()}**")
        msg = "🔗 **Compound command:**\n\n" + "\n".join(lines)
    else:
        return True

    st.markdown(
        """
        <div style="
            background: #78350F15;
            border: 1px solid #F59E0B40;
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
        ">
            <div style="
                font-size: 0.8rem;
                font-weight: 700;
                color: #F59E0B;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 8px;
            ">⚠️ Confirmation Required</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(msg)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm & Execute", key="confirm_btn", use_container_width=True):
            return True
    with col2:
        if st.button("❌ Cancel", key="cancel_btn", use_container_width=True):
            return False

    return None


def render_transcript_box(text: str):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #4C1D9520 0%, #7C3AED10 100%);
            border: 1px solid #7C3AED40;
            border-radius: 12px;
            padding: 16px 20px;
            margin: 8px 0 16px 0;
        ">
            <div style="
                font-size: 0.7rem;
                font-weight: 700;
                color: #8B5CF6;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 8px;
            ">🎤 You said</div>
            <div style="
                color: #E2E8F0;
                font-size: 1.05rem;
                font-style: italic;
                line-height: 1.5;
            ">"{text}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_intent_badge(intent: str, confidence: str, reasoning: str):
    intent_config = {
        "create_file": ("📄", "Create File", "#10B981"),
        "write_code": ("💻", "Write Code", "#3B82F6"),
        "summarize": ("📋", "Summarize", "#F59E0B"),
        "general_chat": ("💬", "Chat", "#8B5CF6"),
        "compound": ("🔗", "Compound", "#EC4899"),
    }

    icon, label, color = intent_config.get(intent, ("⚡", intent.replace("_", " ").title(), "#94A3B8"))

    conf_colors = {"high": "#10B981", "medium": "#F59E0B", "low": "#EF4444"}
    conf_color = conf_colors.get(confidence, "#94A3B8")

    st.markdown(
        f"""
        <div style="
            background: {color}10;
            border: 1px solid {color}30;
            border-radius: 12px;
            padding: 14px 18px;
            margin: 8px 0;
        ">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="font-size: 1.2rem;">{icon}</span>
                <span style="
                    color: {color};
                    font-weight: 700;
                    font-size: 0.95rem;
                ">{label}</span>
                <span style="
                    background: {conf_color}18;
                    color: {conf_color};
                    font-size: 0.65rem;
                    font-weight: 700;
                    padding: 2px 8px;
                    border-radius: 6px;
                    text-transform: uppercase;
                    letter-spacing: 0.06em;
                ">{confidence}</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.82rem;">{reasoning}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
