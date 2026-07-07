"""Throwaway smoke test: confirms the trained openWakeWord ONNX model loads and
actually fires on "Vlad", in isolation from the rest of the pipeline.
Run, say "Vlad" a few times, Ctrl+C to stop. Delete once the feature is confirmed
working end-to-end."""

import numpy as np
import sounddevice as sd

import openwakeword_wake

detector = openwakeword_wake.create_from_config()
print(f"Модель загружена (frame_length={detector.frame_length}, sample_rate={detector.sample_rate}).")
print("Скажи 'Vlad'. Ctrl+C для выхода.", flush=True)

frame_bytes = detector.frame_length * 2
leftover = b""

try:
    with sd.RawInputStream(
        samplerate=detector.sample_rate,
        blocksize=0,
        dtype="int16",
        channels=1,
    ) as stream:
        while True:
            block, _ = stream.read(detector.frame_length)
            buf = leftover + bytes(block)
            while len(buf) >= frame_bytes:
                frame = buf[:frame_bytes]
                buf = buf[frame_bytes:]
                samples = np.frombuffer(frame, dtype=np.int16)
                if detector.detected(samples):
                    print("Обнаружено!", flush=True)
            leftover = buf
except KeyboardInterrupt:
    pass
finally:
    detector.delete()
