import torch
from transformers import MarianMTModel, MarianTokenizer


SUPPORTED_LANGUAGES = {"en", "fr"}


class Translator:
    """
    Local neural machine translation using Helsinki-NLP MarianMT.
    """

    def __init__(self):
        self.device = self._get_device()
        self.models = {}

        print(
            f"Translation device: {self.device}"
        )

    @staticmethod
    def _get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def _load_model(
        self,
        source_language: str,
        target_language: str,
    ):
        pair = (
            f"{source_language}-"
            f"{target_language}"
        )

        if pair in self.models:
            return self.models[pair]

        model_name = (
            "Helsinki-NLP/"
            f"opus-mt-{source_language}-"
            f"{target_language}"
        )

        print(
            f"Loading translation model: "
            f"{model_name}"
        )

        tokenizer = MarianTokenizer.from_pretrained(
            model_name
        )

        model = MarianMTModel.from_pretrained(
            model_name
        )

        model.to(self.device)
        model.eval()

        self.models[pair] = {
            "tokenizer": tokenizer,
            "model": model,
        }

        return self.models[pair]

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:

        if not text.strip():
            return ""

        if source_language == target_language:
            return text

        if source_language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                "Unsupported source language."
            )

        if target_language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                "Unsupported target language."
            )

        translation_model = self._load_model(
            source_language,
            target_language,
        )

        tokenizer = translation_model["tokenizer"]
        model = translation_model["model"]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            translated_tokens = model.generate(
                **inputs,
                max_length=512,
                num_beams=1,
            )

        translated_text = tokenizer.decode(
            translated_tokens[0],
            skip_special_tokens=True,
        )

        return translated_text.strip()