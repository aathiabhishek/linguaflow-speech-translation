import whisper
import torch


class SpeechToText:
    """
    Local speech-to-text engine powered by OpenAI Whisper.
    """

    def __init__(self, model_size: str = "base"):
        self.device = self._get_device()

        print(
            f"Loading Whisper '{model_size}' on {self.device}"
        )

        self.model = whisper.load_model(
            model_size,
            device=self.device,
        )

    @staticmethod
    def _get_device():
        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def transcribe(
        self,
        audio_path: str,
        language: str,
    ):
        """
        Transcribe audio using the explicitly selected
        source language.

        Args:
            audio_path: Path to the audio file.
            language: Whisper language code.
                Examples: "en", "fr"

        Returns:
            tuple[str, str]:
                Transcript and language.
        """

        result = self.model.transcribe(
            audio_path,
            task="transcribe",

            # IMPORTANT:
            # Do not let Whisper guess the language.
            language=language,

            temperature=0,
            fp16=False,
            without_timestamps=True,
            condition_on_previous_text=False,
            verbose=False,
        )

        text = result["text"].strip()

        return text, language