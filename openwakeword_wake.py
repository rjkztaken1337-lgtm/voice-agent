"""Thin wrapper around openWakeWord for real-time acoustic wake-word detection.
Used instead of matching "влад" via regex over Whisper output, so
music_control.pause() can fire the instant the wake word is spoken, before any
transcription happens.

Replaces an earlier Porcupine (Picovoice) attempt: Picovoice's free console
requires a business/company email to sign up, which blocked account creation.
openWakeWord is fully open source and runs entirely locally with no account or
API key — the trade-off is training a custom model via the community Colab
notebook (dscripka/openWakeWord) instead of Picovoice's few-minute console.

The assistant's spoken persona name stays "Влад" everywhere else — only the
acoustic trigger is the English word "Vlad", since this training pipeline's
synthetic data generation and base embedding model are English-biased.
"""

import openwakeword
from openwakeword.model import Model

import config

FRAME_LENGTH = 1280  # 80ms at 16kHz - openWakeWord's native frame size
SAMPLE_RATE = 16000


class WakeWordDetector:
    """Wraps a loaded openwakeword.Model with a single custom wake-word model.
    Feed it exactly `frame_length`-sample int16 mono frames at `sample_rate` Hz."""

    def __init__(self, model_path: str, wakeword_name: str, threshold: float):
        openwakeword.utils.download_models()
        self._model = Model(wakeword_models=[model_path], inference_framework="onnx")
        self._wakeword_name = wakeword_name
        self._threshold = threshold

    @property
    def frame_length(self) -> int:
        return FRAME_LENGTH

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def process(self, frame) -> float:
        """frame: a sequence of `frame_length` int16 samples. Returns the
        wake word's detection score for this frame (0..1)."""
        return self._model.predict(frame)[self._wakeword_name]

    def detected(self, frame) -> bool:
        return self.process(frame) > self._threshold

    def delete(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.delete()


def create_from_config() -> WakeWordDetector:
    if not config.WAKEWORD_MODEL_PATH.exists():
        raise RuntimeError(
            f"Файл модели wake-word не найден: {config.WAKEWORD_MODEL_PATH}"
        )
    return WakeWordDetector(
        model_path=str(config.WAKEWORD_MODEL_PATH),
        wakeword_name=config.WAKEWORD_MODEL_PATH.stem,
        threshold=config.WAKEWORD_THRESHOLD,
    )
