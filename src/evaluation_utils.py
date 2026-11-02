"""Validation shared by the evaluation CLI and lightweight tests."""
from pathlib import Path


def load_pairs(english_path, french_path, n):
    if n < 1:
        raise ValueError("Number of pairs must be positive.")
    english = Path(english_path).read_text(encoding="utf-8-sig").splitlines()
    french = Path(french_path).read_text(encoding="utf-8-sig").splitlines()
    if len(english) != len(french):
        raise ValueError("English and French files have different line counts.")
    if len(english) < n:
        raise ValueError(f"Need {n} aligned pairs; found {len(english)}.")
    en, fr = english[:n], french[:n]
    if any(not a.strip() or not b.strip() for a, b in zip(en, fr)):
        raise ValueError("Selected pairs contain empty sentences.")
    return en, fr
