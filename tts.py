from TTS.api import TTS

import config

_tts = None


def _get_tts():
    global _tts
    if _tts is None:
        _tts = TTS(config.TTS_MODEL_NAME)
    return _tts


def synthesize(text: str, out_path, language: str = "ru"):
    if not config.VOICE_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Нет сэмпла голоса: {config.VOICE_SAMPLE_PATH}. "
            "Положи туда чистую запись друга (10-30 сек, wav)."
        )
    tts = _get_tts()
    tts.tts_to_file(
        text=text,
        speaker_wav=str(config.VOICE_SAMPLE_PATH),
        language=language,
        file_path=str(out_path),
    )
    return out_path
