from pathlib import Path
import wave

from piper import PiperVoice


class TextToSpeech:
    """
    Local neural text-to-speech engine powered by Piper.
    """

    VOICES = {
        "en": "en_US-lessac-medium",
        "fr": "fr_FR-siwis-medium",
    }

    def __init__(self, voice_directory: str):
        self.voice_directory = Path(
            voice_directory
        )

        self.voice_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.models = {}

    def _get_voice(
        self,
        language: str,
    ):

        if language not in self.VOICES:
            raise ValueError(
                f"Unsupported TTS language: {language}"
            )

        if language in self.models:
            return self.models[language]

        voice_name = self.VOICES[language]

        model_path = (
            self.voice_directory
            / f"{voice_name}.onnx"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                "\n\n"
                f"Piper voice model not found:\n"
                f"{model_path}\n\n"
                "Download the required voices with:\n"
                "python -m piper.download_voices "
                f"--data-dir {self.voice_directory} "
                f"{voice_name}\n"
            )

        print(
            f"Loading Piper voice: {voice_name}"
        )

        voice = PiperVoice.load(
            str(model_path)
        )

        self.models[language] = voice

        return voice

    def generate_audio(
        self,
        text: str,
        language: str,
        output_path: str,
    ) -> str:

        if not text.strip():
            raise ValueError(
                "Cannot generate speech from empty text."
            )

        voice = self._get_voice(language)

        output_file = Path(output_path)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with wave.open(
            str(output_file),
            "wb",
        ) as wav_file:

            voice.synthesize_wav(
                text,
                wav_file,
            )

        return str(output_file)