import collections
import queue
import threading

import numpy as np
import sounddevice as sd
import webrtcvad

import config

FRAME_MS = 30
FRAME_SAMPLES = int(config.SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 = 2 bytes/sample
SILENCE_FRAMES_TO_END = int(600 / FRAME_MS)  # ~0.6s of trailing silence ends an utterance


class AudioCapture:
    """Owns one persistent mic input stream and feeds two independently-sized
    frame consumers from it: webrtcvad (30ms/480-sample frames) for speech
    segmentation, and Porcupine (whatever frame size its loaded model wants)
    for wake-word detection.

    A ring buffer holds the last `prebuffer_ms` of raw audio so a command
    spoken in the same breath as the wake word ("Vlad, что там с погодой")
    isn't clipped by record_after_wake().
    """

    def __init__(self, aggressiveness: int = 2, prebuffer_ms: int = 400):
        self._vad = webrtcvad.Vad(aggressiveness)
        self._stream = None
        self._queue = queue.Queue()
        self._leftover = b""
        self._ring_lock = threading.Lock()
        self._ring = collections.deque(maxlen=int(config.SAMPLE_RATE * prebuffer_ms / 1000) * 2)

    def _callback(self, indata, frames, time_info, status):
        data = bytes(indata)
        with self._ring_lock:
            self._ring.extend(data)
        self._queue.put(data)

    def start(self) -> None:
        self._stream = sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=0,
            dtype="int16",
            channels=1,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _read_frame(self, frame_bytes: int) -> bytes:
        """Blocks until `frame_bytes` of fresh audio are available and returns
        exactly that many, carrying any excess over to the next call."""
        buf = bytearray(self._leftover)
        while len(buf) < frame_bytes:
            buf.extend(self._queue.get())
        frame, self._leftover = bytes(buf[:frame_bytes]), bytes(buf[frame_bytes:])
        return frame

    def _ring_snapshot(self) -> bytes:
        with self._ring_lock:
            return bytes(self._ring)

    def wait_for_wake(self, detector, on_detected=None) -> None:
        """Blocks until the wake word is detected. Calls `on_detected()`
        (e.g. music_control.pause) the instant it fires, before returning."""
        frame_bytes = detector.frame_length * 2
        while True:
            frame = self._read_frame(frame_bytes)
            samples = np.frombuffer(frame, dtype=np.int16)
            if detector.detected(samples):
                if on_detected is not None:
                    on_detected()
                return

    def _collect_until_silence(self, speech_frames, num_silence):
        while True:
            frame = self._read_frame(FRAME_BYTES)
            speech_frames.append(frame)
            if self._vad.is_speech(frame, config.SAMPLE_RATE):
                num_silence = 0
            else:
                num_silence += 1
                if num_silence > SILENCE_FRAMES_TO_END:
                    break
        pcm = b"".join(speech_frames)
        audio_int16 = np.frombuffer(pcm, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    def record_after_wake(self):
        """Records the utterance right after a wake-word hit. Seeds from the
        ring buffer already in "triggered" state instead of re-running VAD's
        pretrigger gate, so a command spoken in the same breath isn't clipped."""
        prebuffer = self._ring_snapshot()
        usable_len = len(prebuffer) - (len(prebuffer) % FRAME_BYTES)
        speech_frames = [
            prebuffer[i:i + FRAME_BYTES] for i in range(0, usable_len, FRAME_BYTES)
        ]
        return self._collect_until_silence(speech_frames, num_silence=0)

    def listen_for_utterance(self, timeout: float | None = None):
        """Blocks until speech is detected and returns the segment once trailing
        silence ends. If `timeout` is given and no speech starts within that many
        seconds, returns None instead of blocking forever."""
        pretrigger_frames = 0
        max_pretrigger_frames = int(timeout * 1000 / FRAME_MS) if timeout else None

        while True:
            frame = self._read_frame(FRAME_BYTES)
            if self._vad.is_speech(frame, config.SAMPLE_RATE):
                return self._collect_until_silence([frame], num_silence=0)
            pretrigger_frames += 1
            if max_pretrigger_frames is not None and pretrigger_frames >= max_pretrigger_frames:
                return None
