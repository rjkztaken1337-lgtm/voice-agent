import queue

import numpy as np
import sounddevice as sd
import webrtcvad

import config

FRAME_MS = 30
FRAME_SAMPLES = int(config.SAMPLE_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 = 2 bytes/sample
SILENCE_FRAMES_TO_END = int(600 / FRAME_MS)  # ~0.6s of trailing silence ends an utterance
# Hard ceiling on a single utterance's length. Without this, continuous nearby
# background speech (no clean 0.6s gap ever occurs) makes VAD wait forever for
# trailing silence — the assistant looks completely unresponsive for however
# long people keep talking nearby, then dumps one huge garbled multi-minute
# blob into STT. Forcing a cutoff bounds the worst case and keeps the assistant
# checking for the wake word at least this often.
MAX_UTTERANCE_FRAMES = int(12000 / FRAME_MS)  # 12s


class AudioCapture:
    """Owns one persistent mic input stream and hands out speech segments via
    webrtcvad (30ms/480-sample frames)."""

    def __init__(self, aggressiveness: int = 2):
        self._vad = webrtcvad.Vad(aggressiveness)
        self._stream = None
        self._queue = queue.Queue()
        self._leftover = b""

    def _callback(self, indata, frames, time_info, status):
        self._queue.put(bytes(indata))

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

    def flush(self) -> None:
        """Discards whatever the mic captured while we were speaking — without
        this, our own TTS (picked up off the speakers, not headphones) sits in
        the queue and the next listen_for_utterance() treats it as user speech,
        causing the laggy "keeps listening" pileup after every reply."""
        self._leftover = b""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _read_frame(self, frame_bytes: int) -> bytes:
        """Blocks until `frame_bytes` of fresh audio are available and returns
        exactly that many, carrying any excess over to the next call."""
        buf = bytearray(self._leftover)
        while len(buf) < frame_bytes:
            buf.extend(self._queue.get())
        frame, self._leftover = bytes(buf[:frame_bytes]), bytes(buf[frame_bytes:])
        return frame

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
            if len(speech_frames) >= MAX_UTTERANCE_FRAMES:
                break
        pcm = b"".join(speech_frames)
        audio_int16 = np.frombuffer(pcm, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

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
