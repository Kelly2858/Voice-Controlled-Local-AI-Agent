
import io
import tempfile
import os
import logging
import subprocess
import numpy as np
from pathlib import Path

from config import WHISPER_MODEL, AUDIO_SAMPLE_RATE, MAX_AUDIO_SIZE_MB

import torch
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

_pipeline = None
_loaded_model_name = None


def _get_whisper_pipeline(model_name: str):
    global _pipeline, _loaded_model_name

    if _pipeline is not None and _loaded_model_name == model_name:
        return _pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(
        "Loading Whisper model '%s' on %s (dtype=%s)...",
        model_name, device, torch_dtype,
    )

    _pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=30,
    )
    _loaded_model_name = model_name

    logger.info("Whisper model loaded successfully on %s", device)
    return _pipeline


_FORMAT_SIGNATURES = {
    b"RIFF": "wav",
    b"ID3":  "mp3",
    b"\xff\xfb": "mp3",
    b"\xff\xf3": "mp3",
    b"\xff\xf2": "mp3",
    b"OggS": "ogg",
    b"fLaC": "flac",
    b"\x1aE\xdf\xa3": "webm",
}


def _detect_audio_format(audio_bytes: bytes) -> str:
    header = audio_bytes[:12]
    for magic, ext in _FORMAT_SIGNATURES.items():
        if header[:len(magic)] == magic:
            return ext
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "m4a"
    return "webm"


def _convert_to_wav(input_path: str, output_path: str) -> bool:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "1",
                "-sample_fmt", "s16",
                output_path,
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except FileNotFoundError:
        logger.warning("ffmpeg not found on system PATH")
        return False
    except Exception as e:
        logger.warning("ffmpeg conversion error: %s", e)
        return False


def _load_audio(audio_bytes: bytes, detected_format: str) -> tuple[np.ndarray, int] | None:
    tmp_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=f".{detected_format}",
            delete=False,
            prefix="memo_",
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            import librosa
            audio, sr = librosa.load(tmp_path, sr=AUDIO_SAMPLE_RATE, mono=True)
            logger.info("Audio loaded with librosa (%d samples, %dHz)", len(audio), sr)
            return audio, sr
        except Exception as e:
            logger.info("librosa load failed: %s — trying soundfile", e)

        try:
            import soundfile as sf
            audio, sr = sf.read(tmp_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != AUDIO_SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
                sr = AUDIO_SAMPLE_RATE
            logger.info("Audio loaded with soundfile (%d samples, %dHz)", len(audio), sr)
            return audio.astype(np.float32), sr
        except Exception as e:
            logger.info("soundfile load failed: %s — trying ffmpeg conversion", e)

        wav_path = tmp_path.rsplit(".", 1)[0] + "_converted.wav"
        if _convert_to_wav(tmp_path, wav_path):
            import soundfile as sf
            audio, sr = sf.read(wav_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            logger.info("Audio loaded via ffmpeg conversion (%d samples, %dHz)", len(audio), sr)
            return audio.astype(np.float32), sr

        logger.error("All audio loading methods failed")
        return None

    finally:
        for path in [tmp_path, wav_path]:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def get_whisper_model_name() -> str:
    return WHISPER_MODEL


def transcribe_audio(audio_bytes: bytes, model_name: str | None = None,
                     filename: str = "audio.wav") -> dict:
    model_name = model_name or WHISPER_MODEL

    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_AUDIO_SIZE_MB:
        return {
            "success": False,
            "text": "",
            "error": f"Audio file too large ({size_mb:.1f} MB). Maximum is {MAX_AUDIO_SIZE_MB} MB.",
        }

    if len(audio_bytes) < 1000:
        return {
            "success": False,
            "text": "",
            "error": "Audio recording is too short. Please record at least 1 second of speech.",
        }

    detected_format = _detect_audio_format(audio_bytes)
    user_ext = Path(filename).suffix.lower().lstrip(".")

    if user_ext == "wav" and detected_format != "wav":
        logger.info("Audio labeled .wav but detected as %s", detected_format)

    logger.info(
        "Transcribing audio: detected=%s, size=%.1f KB, model=%s",
        detected_format, len(audio_bytes) / 1024, model_name,
    )

    try:
        result = _load_audio(audio_bytes, detected_format)
        if result is None:
            return {
                "success": False,
                "text": "",
                "error": (
                    f"Could not load audio (format: {detected_format}). "
                    "Please install ffmpeg: https://ffmpeg.org/download.html\n"
                    "On Windows: winget install ffmpeg\n"
                    "On Mac: brew install ffmpeg\n"
                    "On Linux: sudo apt install ffmpeg"
                ),
            }

        audio_array, sample_rate = result

    except Exception as e:
        logger.exception("Audio loading failed")
        return {
            "success": False,
            "text": "",
            "error": f"Failed to load audio: {e}",
        }

    try:
        pipe = _get_whisper_pipeline(model_name)

        output = pipe(
            {"raw": audio_array, "sampling_rate": sample_rate},
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
            },
            return_timestamps=False,
        )

        text = output.get("text", "").strip() if isinstance(output, dict) else str(output).strip()

        if not text:
            return {
                "success": False,
                "text": "",
                "error": "No speech detected. Please speak clearly and try again.",
            }

        logger.info("Transcription successful (%d chars): '%s'", len(text), text[:80])
        return {"success": True, "text": text, "error": None}

    except ImportError as e:
        logger.exception("Missing dependency for Whisper")
        return {
            "success": False,
            "text": "",
            "error": (
                f"Missing dependency: {e}\n"
                "Please install: pip install transformers torch librosa soundfile"
            ),
        }
    except Exception as e:
        logger.exception("Whisper transcription failed")
        error_msg = str(e)

        if "out of memory" in error_msg.lower():
            error_msg = (
                "Out of GPU memory. Try a smaller model (e.g., openai/whisper-tiny) "
                "or set WHISPER_MODEL=openai/whisper-base in your .env file."
            )
        elif "no module" in error_msg.lower() or "import" in error_msg.lower():
            error_msg = (
                f"Missing dependency: {error_msg}\n"
                "Run: pip install transformers torch librosa soundfile"
            )

        return {"success": False, "text": "", "error": error_msg}
