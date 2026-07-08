"""Mock-only tests for the mpv-based music_control.py engine and the bare
volume-trigger regexes in main.py. No real playback, no real mpv process, no
real IPC socket — every test patches the boundary (socket / subprocess /
shutil.which) so this suite is safe to run anytime.

Run: .venv/bin/python3 -m unittest test_music_control -v
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import music_control
import local_actions
from main import _BARE_VOL_UP_RE, _BARE_VOL_DOWN_RE


class FakeSocket:
    """Stands in for the mpv IPC unix socket. Feeds back one JSON-lines reply
    per sendall(), matching the request_id it was just sent — enough for
    _request()'s read loop without a real socket or mpv process."""

    def __init__(self, replies=None, extra_lines=None):
        self.sent = []
        self._replies = list(replies or [])
        self._extra_lines = list(extra_lines or [])

    def sendall(self, payload):
        self.sent.append(json.loads(payload.decode()))

    def makefile(self, mode):
        req = self.sent[-1]
        data = json.loads(reply_for(req, self._replies))
        lines = self._extra_lines + [json.dumps(data) + "\n"]
        return FakeFile(lines)

    def close(self):
        pass

    def settimeout(self, t):
        pass

    def connect(self, addr):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def reply_for(req, replies):
    if replies:
        return replies.pop(0)
    return json.dumps({"request_id": req["request_id"], "error": "success", "data": None})


class FakeFile:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        if not self._lines:
            return ""
        return self._lines.pop(0)


class IpcRequestTests(unittest.TestCase):
    """_request() must match replies by request_id and skip unsolicited event
    lines rather than treating them as the answer."""

    def test_parses_successful_reply(self):
        sock = FakeSocket(replies=[json.dumps({"request_id": 1, "error": "success", "data": 42})])
        result = music_control._request(sock, ["get_property", "volume"], request_id=1)
        self.assertEqual(result["data"], 42)

    def test_raises_on_mpv_error(self):
        sock = FakeSocket(replies=[json.dumps({"request_id": 1, "error": "property not found"})])
        with self.assertRaises(RuntimeError):
            music_control._request(sock, ["get_property", "nope"], request_id=1)

    def test_skips_unsolicited_event_lines(self):
        sock = FakeSocket(
            extra_lines=[json.dumps({"event": "pause"}) + "\n"],
            replies=[json.dumps({"request_id": 1, "error": "success", "data": "ok"})],
        )
        result = music_control._request(sock, ["get_property", "path"], request_id=1)
        self.assertEqual(result["data"], "ok")


class FadeMathTests(unittest.TestCase):
    """_fade() must reach exactly the target volume and never overshoot past it,
    regardless of direction."""

    def _run_fade(self, start, end):
        sent_volumes = []

        def fake_cmd(command, sock=None):
            if command[0] == "set_property" and command[1] == "volume":
                sent_volumes.append(command[2])
            return None

        with patch.object(music_control, "_cmd", side_effect=fake_cmd):
            with patch.object(music_control, "time") as fake_time:
                music_control._fade(MagicMock(), start, end, duration=0.08)
        return sent_volumes

    def test_fade_up_ends_exactly_at_target(self):
        volumes = self._run_fade(0, 35)
        self.assertEqual(volumes[-1], 35)
        self.assertTrue(all(v <= 35 for v in volumes))
        self.assertEqual(volumes, sorted(volumes))

    def test_fade_down_ends_exactly_at_target(self):
        volumes = self._run_fade(35, 0)
        self.assertEqual(volumes[-1], 0)
        self.assertTrue(all(v >= 0 for v in volumes))
        self.assertEqual(volumes, sorted(volumes, reverse=True))


class EnsureMpvTests(unittest.TestCase):
    """_ensure_mpv() branching: already-alive short-circuits, missing binary
    raises, and a successful spawn is detected via the polled _mpv_alive()."""

    def test_noop_when_already_alive(self):
        with patch.object(music_control, "_mpv_alive", return_value=True):
            with patch.object(music_control, "subprocess") as fake_subprocess:
                music_control._ensure_mpv()
                fake_subprocess.Popen.assert_not_called()

    def test_raises_when_mpv_not_installed(self):
        with patch.object(music_control, "_mpv_alive", return_value=False):
            with patch.object(music_control, "shutil") as fake_shutil:
                fake_shutil.which.return_value = None
                with self.assertRaises(RuntimeError):
                    music_control._ensure_mpv()

    def test_spawns_and_waits_for_socket(self):
        alive_calls = {"n": 0}

        def fake_alive():
            # False on the pre-lock check AND the post-lock re-check (that
            # second check is what guards against a concurrent spawn race),
            # then True once polled after Popen() has been called.
            alive_calls["n"] += 1
            return alive_calls["n"] > 2

        with patch.object(music_control, "_mpv_alive", side_effect=fake_alive):
            with patch.object(music_control, "shutil") as fake_shutil:
                fake_shutil.which.return_value = "/opt/homebrew/bin/mpv"
                with patch.object(music_control, "subprocess") as fake_subprocess:
                    # Real lock/log files under db/ (gitignored) — only the
                    # mpv spawn itself is mocked, so flock() gets a real fd.
                    music_control._ensure_mpv()
                fake_subprocess.Popen.assert_called_once()


class MusicPlayContractTests(unittest.TestCase):
    """local_actions._music_play() must keep returning "" on success and the
    fixed error string on RuntimeError, regardless of the mpv rewrite underneath."""

    def test_liked_success_is_silent(self):
        with patch.object(music_control, "play_liked", return_value="Включаю твои любимые треки."):
            self.assertEqual(local_actions._music_play("включи любимые треки"), "")

    def test_liked_failure_reports_spoken_error(self):
        with patch.object(music_control, "play_liked", side_effect=RuntimeError("no net")):
            self.assertEqual(
                local_actions._music_play("включи любимые треки"),
                "Не получилось, проблема с интернетом.",
            )

    def test_non_music_text_returns_none(self):
        self.assertIsNone(local_actions._music_play("какая погода"))


class BareTriggerRegexTests(unittest.TestCase):
    """Bare volume triggers must match only a standalone utterance, not the
    word embedded in an unrelated sentence, and must NOT match when preceded
    by the wake word (that case is handled by split_wake_word instead)."""

    def test_matches_bare_up(self):
        self.assertTrue(_BARE_VOL_UP_RE.match("громче"))
        self.assertTrue(_BARE_VOL_UP_RE.match("Погромче!"))

    def test_matches_bare_down(self):
        self.assertTrue(_BARE_VOL_DOWN_RE.match("тише"))
        self.assertTrue(_BARE_VOL_DOWN_RE.match("Потише."))

    def test_does_not_match_with_wake_word_prefix(self):
        self.assertIsNone(_BARE_VOL_UP_RE.match("рэс, погромче"))

    def test_does_not_match_word_inside_sentence(self):
        self.assertIsNone(_BARE_VOL_DOWN_RE.match("тише, дети, не шумите"))
        self.assertIsNone(_BARE_VOL_UP_RE.match("сделай мне погромче музыку"))


if __name__ == "__main__":
    unittest.main()
