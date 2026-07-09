"""Tests for AudioCapture.listen_for_barge_in()'s debounce logic. No real
mic, no real audio device: webrtcvad.Vad.is_speech is mocked and frames are
fed directly into the capture's internal queue, following the same
mock-the-boundary pattern as test_music_control.py.

Run: .venv/bin/python3 -m unittest test_wake_listener -v
"""

import queue as queue_mod
import threading
import unittest
from unittest.mock import MagicMock

import wake_listener
from wake_listener import AudioCapture, FRAME_BYTES


def _frame(n=FRAME_BYTES):
    return b"\x00" * n


def _make_capture(is_speech_side_effect):
    capture = AudioCapture()
    capture._vad = MagicMock()
    capture._vad.is_speech.side_effect = is_speech_side_effect
    return capture


class ListenForBargeInTests(unittest.TestCase):
    """listen_for_barge_in() must debounce against brief single-frame blips
    (this is exactly what the assistant's own TTS bleeding into the mic off
    the speakers looks like — see wake_listener.py's module notes) and only
    fire on_triggered once min_consecutive back-to-back frames read as
    speech."""

    def test_short_blip_does_not_trigger(self):
        # 3 "speech" frames (one short of min_consecutive=4) then a
        # "silence" frame resets the streak — must not fire on_triggered.
        capture = _make_capture([True, True, True, False])
        for _ in range(4):
            capture._queue.put(_frame())

        cancel_event = threading.Event()
        on_triggered = MagicMock()
        result_holder = []

        def run():
            result_holder.append(
                capture.listen_for_barge_in(cancel_event, on_triggered, min_consecutive=4, poll_s=0.01)
            )

        t = threading.Thread(target=run, daemon=True)
        t.start()
        # No 5th frame is ever queued, so the loop is left polling the empty
        # queue exactly like it would once a reply finishes — cancel it.
        t.join(timeout=0.3)
        cancel_event.set()
        t.join(timeout=1)

        on_triggered.assert_not_called()
        self.assertIsNone(result_holder[0])

    def test_sustained_speech_triggers_once(self):
        def speech_pattern(*_a, **_k):
            speech_pattern.calls += 1
            return speech_pattern.calls <= 4

        speech_pattern.calls = 0
        capture = _make_capture(speech_pattern)
        # 4 consecutive speech frames to satisfy the debounce, then enough
        # trailing silence frames for _collect_until_silence to end it.
        for _ in range(4 + wake_listener.SILENCE_FRAMES_TO_END + 1):
            capture._queue.put(_frame())

        cancel_event = threading.Event()
        on_triggered = MagicMock()

        audio = capture.listen_for_barge_in(cancel_event, on_triggered, min_consecutive=4, poll_s=0.01)

        on_triggered.assert_called_once()
        self.assertIsNotNone(audio)
        self.assertGreater(len(audio), 0)

    def test_cancel_set_first_returns_none_without_triggering(self):
        capture = _make_capture([False, False])
        cancel_event = threading.Event()
        cancel_event.set()
        on_triggered = MagicMock()

        result = capture.listen_for_barge_in(cancel_event, on_triggered, min_consecutive=4, poll_s=0.01)

        on_triggered.assert_not_called()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
