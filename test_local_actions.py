"""Tests for local_actions.py's volume matching. No real osascript calls —
_osa and _get_volume are mocked.

Run: .venv/bin/python3 -m unittest test_local_actions -v
"""

import unittest
from unittest.mock import patch

import local_actions


class VolumeCaseTests(unittest.TestCase):
    """STT capitalizes the first word of a standalone utterance ("Громче.",
    "Тише."), so the volume regexes must match regardless of case — otherwise
    the bare "Громче"/"Тише" triggers in main.py match on entry (their regexes
    do use re.IGNORECASE) but _volume() then silently fails to match and
    handle() returns None, so nothing actually happens and nothing is spoken."""

    def test_capitalized_up_matches(self):
        with patch.object(local_actions, "_get_volume", return_value=50), \
             patch.object(local_actions, "_osa") as fake_osa:
            reply = local_actions._volume("Громче.")
        self.assertEqual(reply, "Сделал погромче.")
        fake_osa.assert_called_once()

    def test_capitalized_down_matches(self):
        with patch.object(local_actions, "_get_volume", return_value=50), \
             patch.object(local_actions, "_osa") as fake_osa:
            reply = local_actions._volume("Тише.")
        self.assertEqual(reply, "Сделал потише.")
        fake_osa.assert_called_once()


if __name__ == "__main__":
    unittest.main()
