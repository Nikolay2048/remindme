import re

_EDGE_PUNCT = " \t\n\r-.,!?;:\"'()[]{}<>"
_RE_SPACES = re.compile(r"\s+")

def normalize_vocab_key(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.strip(_EDGE_PUNCT)
    t = _RE_SPACES.sub(" ", t)
    return t
