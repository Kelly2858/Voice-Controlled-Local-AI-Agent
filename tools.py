
import os
import logging
from pathlib import Path

import ollama as ollama_client
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def _safe_path(filename: str) -> Path:
    clean = filename.lstrip("/\\")
    resolved = (OUTPUT_DIR / clean).resolve()
    if not str(resolved).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError(f"Path traversal blocked: {filename}")
    return resolved


def _strip_markdown_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def create_file(filename: str, content: str = "") -> dict:
    try:
        if not filename:
            filename = "untitled.txt"

        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        size_str = f"{len(content)} chars" if content else "empty"
        return {
            "success": True,
            "action": f"Created `{filename}`",
            "output": content if content else "(empty file)",
            "file_path": str(path),
            "details": f"File saved to output/{filename} ({size_str})",
        }
    except Exception as e:
        logger.exception("create_file failed")
        return {
            "success": False,
            "action": f"Failed to create `{filename}`",
            "output": str(e),
            "file_path": None,
            "details": None,
        }


def write_code(filename: str, language: str, description: str, model: str) -> dict:
    try:
        if not filename:
            ext_map = {
                "python": ".py", "javascript": ".js", "typescript": ".ts",
                "java": ".java", "c": ".c", "cpp": ".cpp", "c++": ".cpp",
                "go": ".go", "rust": ".rs", "html": ".html", "css": ".css",
                "bash": ".sh", "shell": ".sh", "ruby": ".rb", "php": ".php",
                "swift": ".swift", "kotlin": ".kt", "r": ".r", "sql": ".sql",
            }
            ext = ext_map.get(language.lower(), ".txt")
            safe_desc = "".join(c if c.isalnum() else "_" for c in description[:30]).strip("_").lower()
            filename = f"{safe_desc or 'generated'}{ext}"

        prompt = (
            f"Write clean, well-commented {language} code that does the following:\n\n"
            f"{description}\n\n"
            f"Requirements:\n"
            f"- Output ONLY the raw code\n"
            f"- No markdown fences (no ```)\n"
            f"- No explanations before or after the code\n"
            f"- Include helpful comments in the code\n"
            f"- Follow {language} best practices and conventions"
        )

        response = ollama_client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are an expert {language} programmer. "
                        f"Output ONLY raw {language} code. "
                        f"No markdown, no explanations, no code fences. Just the code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )

        code = response["message"]["content"].strip()
        code = _strip_markdown_fences(code)

        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")

        line_count = len(code.splitlines())
        return {
            "success": True,
            "action": f"Generated {language} code → `{filename}`",
            "output": code,
            "file_path": str(path),
            "details": f"{line_count} lines of {language} saved to output/{filename}",
        }
    except Exception as e:
        logger.exception("write_code failed")
        return {
            "success": False,
            "action": f"Failed to generate code for `{filename}`",
            "output": str(e),
            "file_path": None,
            "details": None,
        }


def summarize_text(text: str, model: str) -> dict:
    try:
        if not text:
            return {
                "success": False,
                "action": "Summarization failed",
                "output": "No text provided to summarize.",
                "file_path": None,
                "details": None,
            }

        response = ollama_client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise summarization assistant. "
                        "Provide clear, well-structured summaries. "
                        "Use bullet points for key points when appropriate."
                    ),
                },
                {"role": "user", "content": f"Please summarize the following text:\n\n{text}"},
            ],
            options={"temperature": 0.3},
        )

        summary = response["message"]["content"].strip()
        word_count = len(summary.split())

        return {
            "success": True,
            "action": "Text summarized",
            "output": summary,
            "file_path": None,
            "details": f"Summary: {word_count} words",
        }
    except Exception as e:
        logger.exception("summarize_text failed")
        return {
            "success": False,
            "action": "Summarization failed",
            "output": str(e),
            "file_path": None,
            "details": None,
        }


def general_chat(message: str, chat_history: list, model: str) -> dict:
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Memo, a helpful and friendly AI voice assistant. "
                    "Be concise but thorough. Use markdown formatting for better readability. "
                    "If the user's message seems like a misheard transcription, "
                    "try to understand what they likely meant."
                ),
            },
        ]
        for msg in (chat_history or [])[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": message})

        response = ollama_client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.7},
        )

        reply = response["message"]["content"].strip()

        return {
            "success": True,
            "action": "Chat response",
            "output": reply,
            "file_path": None,
            "details": None,
        }
    except Exception as e:
        logger.exception("general_chat failed")
        return {
            "success": False,
            "action": "Chat failed",
            "output": str(e),
            "file_path": None,
            "details": None,
        }


def execute_intent(intent_result: dict, model: str, chat_history: list | None = None) -> list[dict]:
    intent = intent_result.get("intent", "general_chat")
    params = intent_result.get("parameters", {})
    results = []

    if intent == "create_file":
        results.append(create_file(
            filename=params.get("filename", "untitled.txt"),
            content=params.get("content", ""),
        ))

    elif intent == "write_code":
        results.append(write_code(
            filename=params.get("filename", ""),
            language=params.get("language", "python"),
            description=params.get("description", ""),
            model=model,
        ))

    elif intent == "summarize":
        results.append(summarize_text(
            text=params.get("text", ""),
            model=model,
        ))

    elif intent == "general_chat":
        results.append(general_chat(
            message=params.get("message", ""),
            chat_history=chat_history or [],
            model=model,
        ))

    elif intent == "compound":
        sub_intents = params.get("sub_intents", [])
        if not sub_intents:
            results.append(general_chat(
                message="I couldn't parse the compound command. Could you rephrase?",
                chat_history=chat_history or [],
                model=model,
            ))
        else:
            previous_output = ""
            for sub in sub_intents:
                sub_intent = sub.get("intent", "general_chat")
                sub_params = sub.get("parameters", {})

                if sub_intent == "create_file" and not sub_params.get("content") and previous_output:
                    sub_params["content"] = previous_output

                sub_result_obj = {
                    "intent": sub_intent,
                    "parameters": sub_params,
                }
                sub_results = execute_intent(sub_result_obj, model, chat_history)
                results.extend(sub_results)

                if sub_results and sub_results[-1].get("success"):
                    previous_output = sub_results[-1].get("output", "")

    else:
        results.append(general_chat(
            message=params.get("message", intent_result.get("reasoning", "")),
            chat_history=chat_history or [],
            model=model,
        ))

    return results
