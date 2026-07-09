"""Tests for main.py's converse() follow-up window. No real audio, no real STT/TTS,
no real mpv — every boundary (AudioCapture, stt.transcribe, handle_command,
music_control) is mocked.

Run: .venv/bin/python3 -m unittest test_main -v
"""

import queue
import unittest
from unittest.mock import patch

import main


class FakeAudio:
    def __init__(self, size=1):
        self.size = size


class FakeCapture:
    """Stands in for AudioCapture: hands back one queued "utterance" per
    listen_for_utterance() call, then None once the queue is empty (as if
    the follow-up timeout silence was reached)."""

    def __init__(self, utterances):
        self._utterances = list(utterances)
        self.flush_calls = 0
        self.listen_calls = 0

    def flush(self):
        self.flush_calls += 1

    def listen_for_utterance(self, timeout=None):
        self.listen_calls += 1
        if self._utterances:
            return self._utterances.pop(0)
        return None


class ConverseFollowupWindowTests(unittest.TestCase):
    """converse() must not keep re-arming the wake-word-free follow-up window
    forever. If music never starts playing (e.g. the command opened an app,
    not mpv playback), the loop must not keep re-listening indefinitely —
    otherwise any unrelated speech in the room within earshot gets treated as
    a command to the assistant, one follow-up turn after another, with no
    natural end. This reproduces the exact chain seen live: "включи YouTube"
    -> "включи музыку" -> a mis-transcribed phrase -> a Whisper hallucination
    -> unrelated background conversation, all fed to handle_command() back to
    back because is_playing() stayed False the whole time."""

    def test_does_not_chain_multiple_unrelated_utterances(self):
        capture = FakeCapture([FakeAudio(), FakeAudio(), FakeAudio()])
        with patch.object(main, "handle_command") as fake_handle, \
             patch.object(main.stt, "transcribe", return_value="какая-то фраза"), \
             patch.object(main.music_control, "pause"), \
             patch.object(main.music_control, "is_playing", return_value=False):
            main.converse(capture)
        fake_handle.assert_called_once()

    def test_single_followup_still_handled(self):
        """The fix must not regress the basic case: one genuine follow-up in
        the window should still be transcribed and handled."""
        capture = FakeCapture([FakeAudio()])
        with patch.object(main, "handle_command") as fake_handle, \
             patch.object(main.stt, "transcribe", return_value="сколько времени"), \
             patch.object(main.music_control, "pause"), \
             patch.object(main.music_control, "is_playing", return_value=False):
            main.converse(capture)
        fake_handle.assert_called_once_with("сколько времени", capture)

    def test_no_utterance_handles_nothing(self):
        capture = FakeCapture([])
        with patch.object(main, "handle_command") as fake_handle, \
             patch.object(main.stt, "transcribe"), \
             patch.object(main.music_control, "pause"), \
             patch.object(main.music_control, "is_playing", return_value=False):
            main.converse(capture)
        fake_handle.assert_not_called()


class FakePlayer:
    def __init__(self):
        self.played = []
        self.stop_now_calls = 0
        self.close_calls = 0

    def play(self, chunk):
        self.played.append(chunk)

    def stop_now(self):
        self.stop_now_calls += 1

    def close(self):
        self.close_calls += 1


class FakeCaptureNoTrigger:
    """Stands in for a barge-in watcher that never fires — the reply plays
    out fully and _play_with_barge_in cancels it once audio_q is drained."""

    def listen_for_barge_in(self, cancel_event, on_triggered, min_consecutive=None):
        cancel_event.wait()
        return None


class FakeCaptureTriggers:
    """Stands in for a watcher that fires on_triggered immediately, as if
    the user started talking the instant playback began."""

    def __init__(self, audio):
        self._audio = audio

    def listen_for_barge_in(self, cancel_event, on_triggered, min_consecutive=None):
        on_triggered()
        return self._audio


class PlayWithBargeInTests(unittest.TestCase):
    """_play_with_barge_in() must behave like the old _drain_audio+close()
    pattern when nobody interrupts, but cut playback instantly and return the
    captured follow-up audio when the watcher fires."""

    def test_normal_completion_closes_player_and_returns_none(self):
        audio_q = queue.Queue()
        audio_q.put("chunk1")
        audio_q.put("chunk2")
        audio_q.put(main._STREAM_DONE)
        player = FakePlayer()
        capture = FakeCaptureNoTrigger()

        result = main._play_with_barge_in(audio_q, player, capture)

        self.assertIsNone(result)
        self.assertEqual(player.played, ["chunk1", "chunk2"])
        self.assertEqual(player.close_calls, 1)
        self.assertEqual(player.stop_now_calls, 0)

    def test_triggered_stops_playback_and_returns_captured_audio(self):
        audio_q = queue.Queue()
        player = FakePlayer()
        capture = FakeCaptureTriggers(audio="captured-audio")

        result = main._play_with_barge_in(audio_q, player, capture)

        self.assertEqual(result, "captured-audio")
        self.assertEqual(player.stop_now_calls, 1)
        self.assertEqual(player.close_calls, 0)


if __name__ == "__main__":
    unittest.main()
