
import json
import logging
import time
import ollama as ollama_client

from config import LLM_MAX_RETRIES

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an intent classification assistant for a voice-controlled AI agent called "Memo".
Your job is to analyze the user's transcribed speech and classify their intent.

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no code fences.

## Supported Intents

1. **create_file** — User wants to create a new file (possibly with content).
   Parameters: { "filename": "<name>", "content": "<optional content>" }
   Examples: "create a file called notes.txt", "make a new readme file with hello world"

2. **write_code** — User wants to generate code and save it to a file.
   Parameters: { "filename": "<name with extension>", "language": "<programming language>", "description": "<what the code should do>" }
   Examples: "write a python function to sort a list", "create a javascript file that fetches data from an API"

3. **summarize** — User wants to summarize some text or the conversation so far.
   Parameters: { "text": "<the text to summarize>" }
   Examples: "summarize what we talked about", "give me a summary of this: ..."

4. **general_chat** — General conversation, questions, or anything that doesn't fit above.
   Parameters: { "message": "<the user's message>" }
   Examples: "what is the capital of France?", "explain how neural networks work", "hello"

5. **compound** — The user gave MULTIPLE distinct instructions in one command.
   Parameters: { "sub_intents": [ <array of intent objects, each with "intent" and "parameters"> ] }
   Examples: "summarize this text and save it to a file called summary.txt"

## Classification Rules
- If the user says to create a file AND write code to it → use "write_code" (NOT compound).
- If the user asks to summarize AND save the result → use "compound" with sub_intents: [summarize, create_file].
- For code files, infer a reasonable filename if the user doesn't specify one (e.g., "sort.py" for a sort function in Python).
- If you cannot determine the intent, default to "general_chat".
- Pay attention to keywords: "create", "make", "write", "code", "build" → likely create_file or write_code.
- Keywords like "summarize", "sum up", "brief" → likely summarize.
- Casual greetings, questions, explanations → general_chat.

## Output Format (strict JSON)
{
  "intent": "<one of: create_file, write_code, summarize, general_chat, compound>",
  "confidence": "<high | medium | low>",
  "parameters": { ... },
  "reasoning": "<one-sentence explanation>"
}"""


def classify_intent(text: str, model: str, chat_history: list | None = None) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append(msg)

    messages.append({"role": "user", "content": text})

    last_error = None

    for attempt in range(1, LLM_MAX_RETRIES + 2):
        try:
            response = ollama_client.chat(
                model=model,
                messages=messages,
                format="json",
                options={"temperature": 0.1},
            )

            content = response["message"]["content"]
            result = json.loads(content)

            intent = result.get("intent", "general_chat")
            if intent not in ("create_file", "write_code", "summarize", "general_chat", "compound"):
                intent = "general_chat"

            return {
                "success": True,
                "intent": intent,
                "confidence": result.get("confidence", "medium"),
                "parameters": result.get("parameters", {}),
                "reasoning": result.get("reasoning", ""),
                "error": None,
            }

        except json.JSONDecodeError as e:
            last_error = f"LLM returned invalid JSON (attempt {attempt}): {e}"
            logger.warning(last_error)
            if attempt <= LLM_MAX_RETRIES:
                messages.append({
                    "role": "user",
                    "content": "Your response was not valid JSON. Please respond with ONLY a valid JSON object, no markdown fences.",
                })
                time.sleep(0.5)
                continue

        except Exception as e:
            logger.exception("Intent classification failed")
            error_msg = str(e)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                error_msg = (
                    "Cannot connect to Ollama. Please ensure Ollama is running "
                    "(run `ollama serve` in a terminal) and you have pulled a model "
                    f"(run `ollama pull {model}`)."
                )
            return {
                "success": False,
                "intent": "general_chat",
                "confidence": "low",
                "parameters": {"message": text},
                "reasoning": "Intent classification failed; falling back to general chat.",
                "error": error_msg,
            }

    logger.error("All %d intent classification attempts failed", LLM_MAX_RETRIES + 1)
    return {
        "success": False,
        "intent": "general_chat",
        "confidence": "low",
        "parameters": {"message": text},
        "reasoning": "Failed to parse model output after retries; falling back to general chat.",
        "error": last_error,
    }


def get_available_models() -> list[str]:
    try:
        response = ollama_client.list()
        return [m.model for m in response.models] if response.models else []
    except Exception:
        return []
