import hashlib
import threading

import numpy as np
import soundfile as sf
from TTS.api import TTS

import config
import text_normalize

_tts = None
_cond_latents = None  # (gpt_cond_latent, speaker_embedding) for the chosen built-in speaker
_synth_lock = threading.Lock()
_CACHE_DIR = config.STATE_DIR / "tts_cache"

OUTPUT_SAMPLE_RATE = 24000


def _get_tts():
    global _tts
    if _tts is None:
        _tts = TTS(config.TTS_MODEL_NAME).to("mps")
    return _tts


def _get_conditioning_latents():
    # Built-in XTTS speaker presets ship their conditioning latents precomputed
    # inside the model checkpoint (speaker_manager.speakers), so there's no wav
    # to clone from and no per-call/per-warm-up latent computation at all —
    # just a dict lookup by name.
    global _cond_latents
    if _cond_latents is None:
        model = _get_tts().synthesizer.tts_model
        speaker = model.speaker_manager.speakers[config.TTS_SPEAKER_NAME]
        _cond_latents = (speaker["gpt_cond_latent"], speaker["speaker_embedding"])
    return _cond_latents


def warm_up():
    """Loads the TTS model onto MPS and fetches the chosen built-in speaker's
    conditioning latents, so the first real reply doesn't pay for any of it."""
    _get_conditioning_latents()


def synthesize_array(text: str, language: str = "ru") -> np.ndarray:
    if language == "ru":
        text = text_normalize.prepare_for_tts(text)
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
