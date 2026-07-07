import mlx_whisper
import numpy as np

import config

_MODEL_NAME = config.WHISPER_MODEL


def transcribe(audio: np.ndarray) -> str:
    """audio: float32 mono array at config.SAMPLE_RATE."""
    if audio.size == 0:
        return ""
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=_MODEL_NAME,
        language=config.WHISPER_LANGUAGE,
    )
    return result["text"].strip()
