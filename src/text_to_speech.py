"""Cached Piper synthesis with validated downloads and atomic WAV output."""
from pathlib import Path
from threading import Lock
import os
import subprocess
import sys
import tempfile
import wave


class TextToSpeech:
    VOICES = {"en": "en_US-lessac-medium", "fr": "fr_FR-siwis-medium"}

    def __init__(self, voice_directory: str):
        self.voice_directory = Path(voice_directory)
        self.voice_directory.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self._lock = Lock()

    def _get_voice(self, language):
        if language not in self.VOICES:
            raise ValueError(f"Unsupported TTS language: {language}")
        if language in self.models:
            return self.models[language]
        voice_name = self.VOICES[language]
        model_path = self.voice_directory / f"{voice_name}.onnx"
        config_path = self.voice_directory / f"{voice_name}.onnx.json"
        valid = lambda: all(p.is_file() and p.stat().st_size > 0 for p in [model_path, config_path])
        if not valid():
            if os.getenv("HF_HUB_OFFLINE", "0") == "1":
                raise FileNotFoundError(f"Offline mode: missing Piper voice files for {voice_name}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "piper.download_voices", "--data-dir",
                     str(self.voice_directory), voice_name],
                    check=True, timeout=300, capture_output=True, text=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Voice download timed out: {voice_name}") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Voice download failed: {voice_name}. {(exc.stderr or '')[-600:]}") from exc
        if not valid():
            raise FileNotFoundError(f"Piper needs both {model_path.name} and {config_path.name}")
        from piper import PiperVoice
        voice = PiperVoice.load(str(model_path))
        self.models[language] = voice
        return voice

    def generate_audio(self, text: str, language: str, output_path: str) -> str:
        if language not in self.VOICES:
            raise ValueError(f"Unsupported TTS language: {language}")
        if not text.strip():
            raise ValueError("Cannot generate speech from empty text.")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Serialize shared model use and first-time downloads across sessions.
        with self._lock:
            voice = self._get_voice(language)
            fd, temporary = tempfile.mkstemp(suffix=".wav", dir=output.parent)
            os.close(fd)
            try:
                with wave.open(temporary, "wb") as wav_file:
                    voice.synthesize_wav(text, wav_file)
                with wave.open(temporary, "rb") as check:
                    if check.getnframes() == 0:
                        raise RuntimeError("Piper produced empty audio.")
                os.replace(temporary, output)
            except Exception as exc:
                raise RuntimeError(f"Speech synthesis failed: {exc}") from exc
            finally:
                Path(temporary).unlink(missing_ok=True)
        return str(output)
