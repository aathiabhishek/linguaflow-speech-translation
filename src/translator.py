"""Interchangeable local M2M100 and MarianMT translation backends."""
import os
from pathlib import Path
from threading import Lock

from src.text_utils import split_text, validate_languages

ROOT = Path(__file__).resolve().parent.parent


class Translator:
    def __init__(self, model_path=None, backend=None, num_beams=None, device=None,
                 local_files_only=None):
        self.backend = backend or os.getenv("TRANSLATION_BACKEND", "m2m100")
        if self.backend not in {"m2m100", "marian"}:
            raise ValueError("TRANSLATION_BACKEND must be m2m100 or marian.")
        self.num_beams = int(num_beams if num_beams is not None else os.getenv("TRANSLATION_BEAMS", "1"))
        if self.num_beams < 1:
            raise ValueError("num_beams must be positive.")
        self.local_files_only = (os.getenv("HF_HUB_OFFLINE", "0") == "1"
                                 if local_files_only is None else local_files_only)
        configured = model_path or os.getenv("TRANSLATION_MODEL_PATH")
        if configured:
            path = Path(configured).expanduser()
            self.model_path = path if path.is_absolute() else ROOT / path
            if not self.model_path.is_dir():
                raise FileNotFoundError(f"Configured model directory not found: {self.model_path}")
        else:
            canonical = ROOT / "models/m2m100_418M"
            legacy = ROOT / "models/voices/m2m100_418M"
            self.model_path = canonical if canonical.is_dir() else legacy if legacy.is_dir() else None
        self.requested_device = device or os.getenv("TRANSLATION_DEVICE", "auto")
        self.device = None
        self.models = {}
        self._lock = Lock()

    @property
    def label(self):
        return "M2M100 418M" if self.backend == "m2m100" else "MarianMT"

    def _runtime(self):
        import torch
        if self.device is None:
            selected = self.requested_device
            if selected == "auto":
                selected = ("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
            if selected not in {"cpu", "mps", "cuda"}:
                raise ValueError("Device must be auto, cpu, mps, or cuda.")
            self.device = torch.device(selected)
            print(f"Translation device: {self.device}; backend: {self.backend}")
        return torch

    def _load_model(self, source, target):
        torch = self._runtime()
        key = "m2m100" if self.backend == "m2m100" else f"{source}-{target}"
        if key not in self.models:
            from transformers import (M2M100ForConditionalGeneration, M2M100Tokenizer,
                                      MarianMTModel, MarianTokenizer)
            if self.backend == "m2m100":
                name = str(self.model_path) if self.model_path else "facebook/m2m100_418M"
                tokenizer_cls, model_cls = M2M100Tokenizer, M2M100ForConditionalGeneration
            else:
                name = f"Helsinki-NLP/opus-mt-{source}-{target}"
                tokenizer_cls, model_cls = MarianTokenizer, MarianMTModel
            print(f"Loading translation model: {name}")
            tokenizer = tokenizer_cls.from_pretrained(name, local_files_only=self.local_files_only)
            model = model_cls.from_pretrained(name, local_files_only=self.local_files_only)
            model.to(self.device)
            model.eval()
            self.models[key] = (tokenizer, model)
        return torch, *self.models[key]

    def warmup(self, source="fr", target="en"):
        """Load weights before timing an evaluation; no generation is performed."""
        validate_languages(source, target)
        with self._lock:
            self._load_model(source, target)

    def synchronize(self):
        """Wait for device work before reading inference timings."""
        if self.device is None:
            return
        torch = self._runtime()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        validate_languages(source_language, target_language)
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        if not text.strip():
            return ""
        if source_language == target_language:
            return text
        # Streamlit shares cached resources; src_lang is mutable.
        with self._lock:
            torch, tokenizer, model = self._load_model(source_language, target_language)
            if self.backend == "m2m100":
                tokenizer.src_lang = source_language
            limit = min(512, int(tokenizer.model_max_length),
                        int(getattr(model.config, "max_position_embeddings", 512)))
            count = lambda s: len(tokenizer.encode(s, add_special_tokens=True, truncation=False))
            chunks = split_text(text, count, limit)
            outputs = []
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", truncation=False).to(self.device)
                kwargs = dict(max_length=limit, num_beams=self.num_beams,
                              do_sample=False, forced_eos_token_id=None)
                if self.backend == "m2m100":
                    kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(target_language)
                with torch.inference_mode():
                    ids = model.generate(**inputs, **kwargs)[0]
                eos = model.generation_config.eos_token_id
                eos = tokenizer.eos_token_id if eos is None else eos
                eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}
                if len(ids) >= limit and ids[-1].item() not in eos_ids:
                    raise RuntimeError("Translation reached its output limit. Try shorter segments.")
                answer = tokenizer.decode(ids, skip_special_tokens=True).strip()
                if not answer:
                    raise RuntimeError("Model returned an empty translation.")
                outputs.append(answer)
            return " ".join(outputs)
