"""
Deduplication utilities for ASR results (e.g. repeated segments).
"""


def clean_repetition(text):
    """Collapse repeated words: 'ha ha ha ha' -> 'ha ha', 'f f f f' -> 'f'. Max 2 same in a row."""
    words = text.split()
    cleaned = []
    for w in words:
        if len(cleaned) >= 2 and w == cleaned[-1] == cleaned[-2]:
            continue
        cleaned.append(w)
    return " ".join(cleaned)


def remove_overlap(prev_text, new_text):
    prev_words = prev_text.split()
    new_words = new_text.split()

    max_overlap = min(len(prev_words), len(new_words))

    for i in range(max_overlap, 0, -1):
        if prev_words[-i:] == new_words[:i]:
            return " ".join(new_words[i:])

    return new_text