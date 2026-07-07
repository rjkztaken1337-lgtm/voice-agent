import numpy as np
import sounddevice as sd
import webrtcvad

import config

FRAME_MS = 30
FRAME_SAMPLES = int(config.SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_FRAMES_TO_END = int(600 / FRAME_MS)  # ~0.6s of trailing silence ends an utterance


class VadListener:
    """Continuously listens on the mic and returns one speech segment at a time.

    Blocks until webrtcvad detects speech, keeps recording until trailing
    silence, then returns the segment as float32 audio in [-1, 1].
    """

    def __init__(self, aggressiveness: int = 2):
        self._vad = webrtcvad.Vad(aggressiveness)

    def listen_for_utterance(self, timeout: float | None = None):
        """Blocks until speech is detected and returns the segment once trailing
        silence ends. If `timeout` is given and no speech starts within that many
        seconds, returns None instead of blocking forever."""
        speech_frames = []
        triggered = False
        num_silence = 0
        max_pretrigger_frames = int(timeout * 1000 / FRAME_MS) if timeout else None
        pretrigger_frames = 0

        with sd.RawInputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
        ) as stream:
            while True:
                block, _ = stream.read(FRAME_SAMPLES)
                frame = bytes(block)
                is_speech = self._vad.is_speech(frame, config.SAMPLE_RATE)

                if not triggered:
                    if is_speech:
                        triggered = True
                        speech_frames.append(frame)
                        num_silence = 0
                    else:
                        pretrigger_frames += 1
                        if max_pretrigger_frames is not None and pretrigger_frames >= max_pretrigger_frames:
                            return None
                    continue

                speech_frames.append(frame)
                if is_speech:
                    num_silence = 0
                else:
                    num_silence += 1
                    if num_silence > SILENCE_FRAMES_TO_END:
                        break

        pcm = b"".join(speech_frames)
        audio_int16 = np.frombuffer(pcm, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0
