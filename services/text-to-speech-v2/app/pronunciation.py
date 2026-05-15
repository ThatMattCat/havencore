"""Whole-word text substitution applied to TTS input before synthesis.

Chatterbox has no lexicon-injection hook the way Kokoro/misaki does — see
resemble-ai/chatterbox#115. The standard workaround is to rewrite the
input spelling so the model's text encoder sees a pronunciation it'll
render correctly. Example: "Selene" → "Suh-leen" forces 2-syllable
/səˈlin/ instead of the 3-syllable reading the model would otherwise pick.

The substitution is whole-word and case-insensitive on lookup, but
preserves the original word's capitalization pattern in the replacement
where straightforward (initial-cap, all-caps).
"""
import re
from typing import Mapping


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply(text: str, pronunciations: Mapping[str, str]) -> str:
    if not pronunciations or not text:
        return text

    # Build one compiled alternation per call. The map is small (a handful
    # of proper nouns); compile cost is negligible.
    keys = sorted(pronunciations.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b",
        flags=re.IGNORECASE,
    )

    # Build a case-insensitive lookup once.
    lower_map = {k.lower(): v for k, v in pronunciations.items()}

    def _sub(m: re.Match) -> str:
        word = m.group(1)
        replacement = lower_map[word.lower()]
        return _preserve_case(word, replacement)

    return pattern.sub(_sub, text)
