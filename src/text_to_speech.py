from pathlib import Path
import subprocess
import sys
import wave

from piper import PiperVoice


class TextToSpeech:
    """
    Local neural text-to-speech engine powered by Piper.

    Automatically downloads the required Piper voice model
    when it is not already available.
    """

    VOICES = {
        "en": "en_US-lessac-medium",
        "fr": "fr_FR-siwis-medium",
    }

    def __init__(self, voice_directory: str):
        self.voice_directory = Path(voice_directory)

        self.voice_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Cache loaded Piper voices so we don't load
        # the same model repeatedly.
        self.models = {}

    def _get_voice(self, language: str):
        """
        Load a Piper voice.

        If the voice model does not exist locally,
        download it automatically.
        """

        # Validate language
        if language not in self.VOICES:
            raise ValueError(
                f"Unsupported TTS language: {language}"
            )

        # Return cached model if already loaded
        if language in self.models:
            return self.models[language]

        voice_name = self.VOICES[language]

        model_path = (
            self.voice_directory
            / f"{voice_name}.onnx"
        )

        # --------------------------------------------------
        # Download voice if it does not exist
        # --------------------------------------------------

        if not model_path.exists():

            print(
                f"Downloading Piper voice: {voice_name}"
            )

            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "piper.download_voices",
                        "--data-dir",
                        str(self.voice_directory),
                        voice_name,
                    ],
                    check=True,
                )

            except subprocess.CalledProcessError as error:

                raise RuntimeError(
                    f"Failed to download Piper voice "
                    f"'{voice_name}'."
                ) from error

        # --------------------------------------------------
        # Verify that the model was downloaded
        # --------------------------------------------------

        if not model_path.exists():

            raise FileNotFoundError(
                "\n\n"
                "Piper voice model could not be downloaded:\n"
                f"{model_path}\n\n"
                "Expected voice:\n"
                f"{voice_name}\n"
            )

        # --------------------------------------------------
        # Load Piper model
        # --------------------------------------------------

        print(
            f"Loading Piper voice: {voice_name}"
        )

        voice = PiperVoice.load(
            str(model_path)
        )

        # Cache model
        self.models[language] = voice

        return voice

    def generate_audio(
        self,
        text: str,
        language: str,
        output_path: str,
    ) -> str:
        """
        Convert text into a WAV audio file.

        Parameters
        ----------
        text:
            Text that should be spoken.

        language:
            TTS language code:
            'en' or 'fr'.

        output_path:
            Destination WAV file.

        Returns
        -------
        str
            Path to generated audio file.
        """

        # Validate text
        if not text.strip():
            raise ValueError(
                "Cannot generate speech from empty text."
            )

        # Get Piper voice
        voice = self._get_voice(language)

        # Prepare output path
        output_file = Path(output_path)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Generate WAV
        with wave.open(
            str(output_file),
            "wb",
        ) as wav_file:

            voice.synthesize_wav(
                text,
                wav_file,
            )

        return str(output_file)
