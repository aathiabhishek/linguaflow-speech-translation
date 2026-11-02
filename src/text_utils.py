"""Token-aware splitting without silently discarding long input."""
import re

SUPPORTED_LANGUAGES = {"en", "fr"}


def validate_languages(source, target):
    if source not in SUPPORTED_LANGUAGES or target not in SUPPORTED_LANGUAGES:
        raise ValueError("Supported languages are 'en' and 'fr'.")


def split_text(text, count_tokens, limit):
    if limit < 3:
        raise ValueError("Token limit is too small.")
    if not text.strip():
        return []
    if count_tokens(text) <= limit:
        return [text]
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if count_tokens(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if count_tokens(sentence) <= limit:
            current = sentence
            continue
        for word in sentence.split():
            if count_tokens(word) > limit:
                raise ValueError("One uninterrupted item exceeds the token limit; shorten it or add spaces.")
            candidate = f"{current} {word}".strip()
            if count_tokens(candidate) <= limit:
                current = candidate
            else:
                chunks.append(current)
                current = word
    if current:
        chunks.append(current)
    return chunks
