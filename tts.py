import hashlib
import threading

import numpy as np
import soundfile as sf
from TTS.api import TTS

import config

_tts = None
_cond_latents = None  # (gpt_cond_latent, speaker_embedding) for the cloned voice
_synth_lock = threading.Lock()
_CACHE_DIR = config.STATE_DIR / "tts_cache"

OUTPUT_SAMPLE_RATE = 24000


def _get_tts():
    global _tts
    if _tts is None:
        _tts = TTS(config.TTS_MODEL_NAME).to("mps")
    return _tts


def _get_conditioning_latents():
    # Recomputing latents from speaker_wav on every call (what TTS.tts_to_file
    # does internally) is most of the per-call cost for short phrases, so we
    # derive them once and reuse — relies on coqui-tts's internal Xtts model API,
    # pinned to coqui-tts==0.27.5.
    global _cond_latents
    if _cond_latents is None:
        if not config.VOICE_SAMPLE_PATH.exists():
            raise FileNotFoundError(
                f"Нет сэмпла голоса: {config.VOICE_SAMPLE_PATH}. "
                "Положи туда чистую запись друга (10-30 сек, wav)."
            )
        model = _get_tts().synthesizer.tts_model
        _cond_latents = model.get_conditioning_latents(
            audio_path=[str(config.VOICE_SAMPLE_PATH)]
        )
    return _cond_latents


def warm_up():
    """Loads the TTS model onto MPS and precomputes the cloned voice's
    conditioning latents, so the first real reply doesn't pay for any of it."""
    _get_conditioning_latents()


def synthesize_array(text: str, language: str = "ru") -> np.ndarray:
    gpt_cond_latent, speaker_embedding = _get_conditioning_latents()
    model = _get_tts().synthesizer.tts_model
    with _synth_lock:
        out = model.inference(text, language, gpt_cond_latent, speaker_embedding)
    return np.asarray(out["wav"], dtype=np.float32)


def synthesize(text: str, out_path, language: str = "ru"):
    wav = synthesize_array(text, language=language)
    sf.write(str(out_path), wav, OUTPUT_SAMPLE_RATE)
    return out_path


def get_cached(text: str, language: str = "ru"):
    """Synthesizes once per (text, language) pair and reuses the cached wav on
    later calls — used for short filler phrases so playback is instant after
    the first warm-up."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{language}:{text}".encode()).hexdigest()[:16]
    path = _CACHE_DIR / f"{key}.wav"
    if not path.exists():
        synthesize(text, path, language=language)
    return path
