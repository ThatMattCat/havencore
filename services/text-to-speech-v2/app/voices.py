"""Voice registry — resolves a ``voice`` request field to a reference WAV path.

Chatterbox-Turbo is zero-shot; to get a stable named voice the model needs a
short reference clip per generation. We mirror v1's "named voices + OpenAI
aliases" surface so callers see the same `voice` field semantics.

The registry is built by scanning two dirs on each query (cheap — these
directories hold at most a few dozen tiny files) so newly uploaded clips
appear without a service restart:

  1. Custom volume-mounted clip at /app/voices/<name>.wav  (uploaded)
  2. Bundled clip at /opt/chatterbox-voices/<name>.wav      (image)

Custom clips override bundled ones with the same name, so an operator can
drop "Selene.wav" into the mounted dir to override a bundled "Selene".

Resolution order for an incoming ``voice`` request field:
  1. Exact match in either dir → use that path
  2. OpenAI alias (alloy/echo/etc.) → resolve to the configured default
  3. Unknown → resolve to the configured default (with a warning)
"""
import logging
import os
from pathlib import Path

import config

logger = logging.getLogger("text-to-speech-v2.voices")

# Same aliases as v1 so OpenAI-SDK clients pointed at hardcoded voice names
# keep working when the agent flips TTS_PROVIDER=v2.
OPENAI_VOICE_ALIASES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}


def _scan_dir(d: str) -> dict[str, str]:
    """Return ``{voice_name: absolute_path}`` for each .wav in ``d``."""
    if not os.path.isdir(d):
        return {}
    found: dict[str, str] = {}
    for entry in os.listdir(d):
        if entry.lower().endswith(".wav"):
            name = Path(entry).stem
            found[name] = os.path.join(d, entry)
    return found


def registry() -> dict[str, str]:
    """Live registry (rescan each call)."""
    bundled = _scan_dir(config.VOICES_BUNDLED_DIR)
    user = _scan_dir(config.VOICES_USER_DIR)
    return {**bundled, **user}


def user_voices() -> list[str]:
    """Return only voices uploaded by the operator (deletable)."""
    return sorted(_scan_dir(config.VOICES_USER_DIR).keys())


def list_names() -> list[str]:
    return sorted(registry().keys())


def default_voice(reg: dict[str, str] | None = None) -> str:
    """Resolve the configured default name against the live registry."""
    reg = registry() if reg is None else reg
    if config.VOICE in reg:
        return config.VOICE
    if reg:
        fallback = sorted(reg.keys())[0]
        logger.warning(
            "Configured CHATTERBOX_VOICE=%r not found in voices dirs; "
            "falling back to %r", config.VOICE, fallback,
        )
        return fallback
    logger.error(
        "No reference voice clips found in %s or %s. Chatterbox-Turbo will "
        "fall back to its default zero-shot voice (inconsistent across calls).",
        config.VOICES_BUNDLED_DIR, config.VOICES_USER_DIR,
    )
    return ""


def resolve(name: str | None) -> tuple[str, str | None]:
    """Return ``(canonical_voice_name, audio_prompt_path_or_None)``.

    ``audio_prompt_path_or_None`` is ``None`` only if no voice clips at all
    are registered — callers should pass it to ``model.generate`` only when
    not ``None`` to let Chatterbox fall back to its zero-shot default.
    """
    reg = registry()
    default = default_voice(reg)
    if name and name in reg:
        return name, reg[name]
    if name and name in OPENAI_VOICE_ALIASES:
        return default, reg.get(default)
    if name:
        logger.warning(
            "Unknown voice %r; falling back to default %r", name, default,
        )
    return default, (reg.get(default) if default else None)
