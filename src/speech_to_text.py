"""Whisper transcription; the chosen language is supplied explicitly."""
from pathlib import Path
from threading import Lock

from src.text_utils import validate_languages


class SpeechToText:
    def __init__(self, model_size="base"):
        import torch
        import whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = Lock()
        print(f"Loading Whisper '{model_size}' on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path: str, language: str) -> tuple[str, str]:
        validate_languages(language, language)
        path = Path(audio_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Audio file is missing or empty.")
        with self._lock:
            result = self.model.transcribe(
                str(path), task="transcribe", language=language, temperature=0,
                fp16=(self.device == "cuda"), without_timestamps=True,
                condition_on_previous_text=False, verbose=False,
            )
        return (result.get("text") or "").strip(), language
