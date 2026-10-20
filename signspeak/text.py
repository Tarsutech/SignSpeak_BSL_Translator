"""Text helpers shared by the speech and fingerspelling parts."""

from __future__ import annotations

import re

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NON_LETTERS = re.compile(r"[^A-Z]+")


def text_to_tokens(text: str) -> list[str]:
    """Turn free text into fingerspellable tokens: A-Z letters and single spaces.

    Digits and punctuation are dropped, runs of anything else become one space.
    """
    if not text:
        return []
    words = [w for w in _NON_LETTERS.split(text.upper()) if w]
    tokens: list[str] = []
    for i, word in enumerate(words):
        if i:
            tokens.append(" ")
        tokens.extend(word)
    return tokens


def wrap_text(text: str, max_chars: int) -> list[str]:
    """Greedy word wrap that also splits words longer than a line."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        while len(word) > max_chars:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current or not lines:
        lines.append(current)
    return lines
