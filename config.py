
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

WHISPER_MODEL = "openai/whisper-small"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "ogg", "webm", "m4a", "flac"]
MAX_AUDIO_SIZE_MB = 50
AUDIO_SAMPLE_RATE = 16000

LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 2

SUPPORTED_INTENTS = [
    "create_file",
    "write_code",
    "summarize",
    "general_chat",
    "compound",
]
