import re

import mlx_whisper
import numpy as np

import config

_MODEL_NAME = config.WHISPER_MODEL

# Whisper hallucinates these stock phrases on silence/near-silence (artifacts of
# its YouTube-subtitle training data) — never real commands, so drop them rather
# than waking the brain on garbage.
_HALLUCINATIONS = re.compile(
    r"^(thanks?\s+for\s+watching!?|thank\s+you\s+for\s+watching!?|"
    r"subscribe(\s+to\s+my\s+channel)?!?|"
    r"please\s+subscribe!?|like\s+and\s+subscribe!?|"
    r"продолжение\s+следует\.?|субтитры\s+делал\s+.*|"
    r"редактор\s+субтитров\s+.*|корректор\s+.*)\.?$",
    re.IGNORECASE,
)

# Whisper often tacks a hallucinated outro onto the end (or start) of a real
# utterance rather than hallucinating the whole thing, e.g. "Рэс включи
# музыку. Продолжение следует." — split into sentences and drop only the
# hallucinated ones instead of keeping or discarding the whole string.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _strip_hallucinations(text: str) -> str:
    sentences = _SENTENCE_SPLIT.split(text)
    kept = [s for s in sentences if s.strip() and not _HALLUCINATIONS.match(s.strip())]
    return " ".join(kept).strip()


def transcribe(audio: np.ndarray) -> str:
    """audio: float32 mono array at config.SAMPLE_RATE."""
    if audio.size == 0:
        return ""
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=_MODEL_NAME,
        language=config.WHISPER_LANGUAGE,
    )
    text = result["text"].strip()
    return _strip_hallucinations(text)
