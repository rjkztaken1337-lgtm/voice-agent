import queue
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

import config


class PushToTalkRecorder:
    """Records mic audio while `start`/`stop` bracket a key-hold."""

    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._chunks = []
        self._q = queue.Queue()
        self._stream = None
        self._recording = False

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata.copy())

    def start(self):
        if self._recording:
            return
        self._chunks = []
        self._q = queue.Queue()
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if not self._recording:
            return np.zeros((0,), dtype="float32")
        self._recording = False
        self._stream.stop()
        self._stream.close()
        while not self._q.empty():
            self._chunks.append(self._q.get())
        if not self._chunks:
            return np.zeros((0,), dtype="float32")
        return np.concatenate(self._chunks, axis=0).flatten()

    @property
    def is_recording(self) -> bool:
        return self._recording


def save_wav(path, audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE):
    sf.write(str(path), audio, sample_rate)


def play_wav(path):
    data, sample_rate = sf.read(str(path), dtype="float32")
    sd.play(data, sample_rate)
    sd.wait()


def play_array(audio: np.ndarray, sample_rate: int):
    sd.play(audio, sample_rate)
    sd.wait()
