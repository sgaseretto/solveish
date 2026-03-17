# Auto-extracted from helpers.ipynb — do not edit directly

from pathlib import Path
from collections import Counter

def word_count(text: str) -> dict:
    """Count word frequencies in text, returning a dict of word -> count."""
    words = text.lower().split()
    return dict(Counter(words))

def read_text(path: str) -> str:
    """Read a text file and return its contents."""
    return Path(path).read_text(encoding='utf-8')

def summarize_counts(counts: dict, top_n: int = 10) -> list:
    """Return the top N most frequent items from a counts dict."""
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]