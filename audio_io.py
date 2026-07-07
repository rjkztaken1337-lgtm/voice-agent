import collections
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

import config

_FADE_MS = 6  # short edge fade to keep segment joins click-free


def _condition(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Prepare a segment for glitch-free playback: force float32 in [-1, 1] (a
    stray sample >1 pops as a click) and fade the first/last few ms to zero.

    XTTS segments don't start at zero (~0.03), so butting them together — or
    starting a fresh stream on each — produces an audible click. Fading both
    edges to zero removes that step, and also makes a buffer underrun between
    segments fall on silence instead of a discontinuity."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    np.clip(audio, -1.0, 1.0, out=audio)
    n = min(int(sample_rate * _FADE_MS / 1000), audio.size // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def load_array(path) -> np.ndarray:
    data, _ = sf.read(str(path), dtype="float32")
    return data


def save_wav(path, audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE):
    sf.write(str(path), audio, sample_rate)


def play_wav(path):
    data, sample_rate = sf.read(str(path), dtype="float32")
    play_array(data, sample_rate)


def play_array(audio: np.ndarray, sample_rate: int):
    sd.play(_condition(audio, sample_rate), sample_rate)
    sd.wait()


class StreamPlayer:
    """Gapless, underrun-proof playback of segments that arrive over time.

    A streamed reply is synthesized sentence by sentence, so the next segment
    isn't always ready when the current one finishes. A blocking write() stream
    then STARVES between segments and CoreAudio clicks on the underrun. Here a
    callback-driven stream pulls from an internal buffer and, when the buffer runs
    dry, outputs zeros — a clean silence instead of a click. Combined with the
    edge-fades in _condition (every segment starts and ends at zero), the
    звук→тишина→звук transition is seamless whether or not synthesis kept up."""

    def __init__(self, sample_rate: int, blocksize: int = 1024):
        self._sr = sample_rate
        self._chunks = collections.deque()
        self._cur = None
        self._pos = 0
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def _fill(self, out: np.ndarray, frames: int):
        """Copy up to `frames` samples from the buffer into `out`; zero-fill the
        remainder if the buffer runs dry. Kept separate from the callback so it
        can be unit-tested without a real audio device."""
        i = 0
        with self._lock:
            while i < frames:
                if self._cur is None or self._pos >= len(self._cur):
                    if not self._chunks:
                        break
                    self._cur = self._chunks.popleft()
                    self._pos = 0
                take = min(frames - i, len(self._cur) - self._pos)
                out[i:i + take] = self._cur[self._pos:self._pos + take]
                self._pos += take
                i += take
        if i < frames:
            out[i:] = 0.0  # buffer dry -> silence, never a click

    def _callback(self, outdata, frames, time_info, status):
        self._fill(outdata[:, 0], frames)

    def play(self, audio: np.ndarray):
        conditioned = _condition(audio, self._sr)
        with self._lock:
            self._chunks.append(conditioned)

    def _pending(self) -> int:
        with self._lock:
            rem = sum(len(c) for c in self._chunks)
            if self._cur is not None:
                rem += len(self._cur) - self._pos
            return rem

    def close(self):
        # Let the callback play out everything queued, plus a short tail for the
        # device's own buffer, before tearing the stream down.
        while self._pending() > 0:
            sd.sleep(20)
        sd.sleep(80)
        try:
            self._stream.stop()
        finally:
            self._stream.close()
