# 🎙️ Memo — Voice-Controlled Local AI Agent

A voice-controlled AI agent that accepts audio input, transcribes it, classifies intent using a local LLM, executes appropriate tools, and displays the full pipeline in a polished Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🏗️ Architecture

```
🎤 Audio Input ──→ 📝 Speech-to-Text ──→ 🧠 Intent Classification ──→ ⚙️ Tool Execution ──→ 🖥️ UI Display
  (Mic / File)       (Groq Whisper)         (Ollama LLM)               (Local Actions)       (Streamlit)
```

### Pipeline Flow

1. **Audio Input** — Record via microphone or upload an audio file (.wav, .mp3, .ogg)
2. **Speech-to-Text** — Transcribe audio using Groq's Whisper API
3. **Intent Classification** — Analyze text with a local Ollama LLM to detect intent
4. **Tool Execution** — Execute the appropriate action (create file, generate code, summarize, chat)
5. **UI Display** — Show transcription, intent, confirmation, and results in a clean pipeline view

---

## 📋 Supported Intents

| Intent | Description | Example Voice Command |
|--------|-------------|-----------------------|
| **Create File** | Create a new file with optional content | _"Create a file called notes.txt"_ |
| **Write Code** | Generate code and save to a file | _"Write a Python retry function"_ |
| **Summarize** | Summarize provided text | _"Summarize the following paragraph…"_ |
| **General Chat** | General Q&A and conversation | _"What is machine learning?"_ |
| **Compound** | Multiple actions in one command | _"Summarize this and save to summary.txt"_ |

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** — Install from [ollama.com](https://ollama.com/) and pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. **Groq API Key** — Get a free key (no credit card) at [console.groq.com](https://console.groq.com/)

### Installation

```bash
# Clone the repo
cd "Memo project"

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env and add your Groq API key

# Start Ollama (if not already running)
ollama serve

# Run the app
streamlit run app.py
```

### Configuration

You can configure the app via the `.env` file or the Streamlit sidebar:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key for Whisper STT | _(required)_ |
| `OLLAMA_MODEL` | Ollama model for intent classification | `llama3.2` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |

---

## 📝 Why Groq API for Speech-to-Text?

We chose **Groq's hosted Whisper API** over running Whisper locally for these reasons:

1. **Hardware Constraints** — Running Whisper Large V3 locally requires significant GPU VRAM (≥6 GB) that many consumer machines lack. This makes it impractical for a project that should work on most developer laptops.

2. **Performance** — Groq's LPU (Language Processing Unit) transcribes audio at approximately **10x real-time speed**, returning results in under 1 second for typical voice commands.

3. **Free Tier** — Groq offers a generous free tier with 2,000 requests/day and 25 MB file size limit, requiring no credit card — more than sufficient for development and demo use.

4. **Drop-in Compatible** — Groq's API is OpenAI-compatible, making it trivial to switch to local Whisper or another provider if desired (just swap the client initialization).

---

## ✨ Bonus Features

### 🔗 Compound Commands
Supports multiple actions in one voice command. Example: _"Summarize this text and save it to summary.txt"_ triggers both summarization and file creation, with the summary automatically piped as file content.

### 🛡️ Human-in-the-Loop
File operations (create file, write code) require explicit user confirmation before execution. A confirmation dialog shows exactly what will happen, with Confirm/Cancel buttons.

### 🪂 Graceful Degradation
- **Ollama unreachable** → Clear error message with setup instructions
- **Invalid API key** → Descriptive error in the pipeline display
- **Unintelligible audio** → Shows "no speech detected" message
- **Unmapped intent** → Falls back to general chat gracefully
- **Path traversal** → Blocked by the safety sandbox validation

### 🧠 Session Memory
Maintains chat history and action history within the session. Previous context is passed to the LLM for smarter follow-up responses. History is visible in the sidebar with intent icons and transcript previews.

### 🔒 Safety Constraints
All file operations are strictly sandboxed to the `output/` directory within the repository. Path traversal attacks are explicitly blocked with validation that resolved paths stay within the sandbox.

---

## 📁 Project Structure

```
Memo project/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration & environment variables
├── stt.py                  # Speech-to-Text (Groq Whisper API)
├── intent.py               # Intent classification (Ollama LLM)
├── tools.py                # Tool execution (file ops, code gen, summarize, chat)
├── ui_components.py        # Reusable Streamlit UI components
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment file
├── README.md               # This file
├── output/                 # Sandboxed output folder (all files go here)
└── .streamlit/
    └── config.toml         # Streamlit dark theme configuration
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | Streamlit |
| Speech-to-Text | Groq Whisper API (`whisper-large-v3-turbo`) |
| LLM (Intent + Code Gen) | Ollama (local, e.g., `llama3.2`) |
| Language | Python 3.10+ |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
