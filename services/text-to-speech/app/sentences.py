"""Sentence splitter for the streaming TTS path.

The streaming endpoint synthesizes one chunk at a time so the first audio
event reaches the client as soon as sentence #1 is rendered, instead of
waiting for the whole utterance. We split on sentence boundaries because:

  * Sentence-shaped chunks feel natural in playback (cadence preserved).
  * Chatterbox-Turbo synthesizes each in ~0.5–2 s on a 3090, giving a
    real time-to-first-audio improvement on long replies.
  * The model's prosody is best when it sees a full sentence; splitting
    mid-clause produces clipped intonation.

Known limitation: a naive regex misclassifies abbreviations like
"Dr. Smith said hello." as two sentences. We accept that — current
assistant output doesn't lean on medical/honorific titles, and a heavier
splitter (NLTK Punkt, spaCy) is overkill for a 200-LOC module.
"""
from __future__ import annotations

import re

# Sentence-ending punctuation followed by whitespace. Captures the split
# point AFTER the punctuation so the punctuation stays attached to the
# preceding sentence — the model needs the period to produce natural
# falling intonation.
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Fragments shorter than this get merged into the *next* sentence rather
# than synthesized alone. "Yes." -> ["Yes. Whatever comes next."] reads
# more naturally than two near-instant chunks. Empirically tuned: 30
# chars covers short interjections without swallowing real sentences.
_MIN_SENTENCE_CHARS = 30

# Hard cap on a single chunk so model latency stays bounded per yield.
# Chatterbox-Turbo on a 3090 synthesizes ~25 chars in ~0.5 s; 280 chars
# is ~5–6 s of audio, a reasonable upper bound on time-to-next-chunk.
# Anything longer gets split on the nearest whitespace inside the cap.
_MAX_SENTENCE_CHARS = 280


def _hard_split(s: str) -> list[str]:
    """Split a single oversize chunk into <= _MAX_SENTENCE_CHARS pieces.

    Prefers whitespace boundaries near the cap to avoid breaking mid-word.
    Falls back to a hard slice only if no whitespace exists in the window
    (rare — usually means a URL or token blob, which won't synthesize
    sensibly either way).
    """
    out: list[str] = []
    remaining = s
    while len(remaining) > _MAX_SENTENCE_CHARS:
        window = remaining[:_MAX_SENTENCE_CHARS]
        # Search for the last whitespace in the window; if none, hard-cut.
        cut = window.rfind(" ")
        if cut <= 0:
            cut = _MAX_SENTENCE_CHARS
        out.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        out.append(remaining)
    return out


def split_sentences(text: str) -> list[str]:
    """Return a list of sentence-shaped chunks suitable for streaming TTS.

    Empty / whitespace-only input returns ``[]`` so callers can short-circuit.
    """
    if not text or not text.strip():
        return []

    raw = [s.strip() for s in _SPLIT_RE.split(text.strip()) if s.strip()]

    # Merge short fragments into the next chunk. "Yes. Let me look that up."
    # is a single chunk; "Yes." alone would synth in ~150 ms and just add
    # an audible click between events.
    merged: list[str] = []
    carry = ""
    for chunk in raw:
        candidate = (carry + " " + chunk).strip() if carry else chunk
        if len(candidate) < _MIN_SENTENCE_CHARS:
            carry = candidate
            continue
        merged.append(candidate)
        carry = ""
    if carry:
        # Trailing short fragment with no "next chunk" to merge into.
        # Emit it as its own chunk rather than gluing it to the previous —
        # a 250–500 ms tail chunk is fine, and gluing here would defeat
        # streaming on common shapes like "...long sentence. Anything else?"
        merged.append(carry)

    # Enforce the hard cap. A run-on sentence (no punctuation in a 400-char
    # blob) won't be split by the regex above, so we cut it here.
    out: list[str] = []
    for chunk in merged:
        if len(chunk) <= _MAX_SENTENCE_CHARS:
            out.append(chunk)
        else:
            out.extend(_hard_split(chunk))
    return out
