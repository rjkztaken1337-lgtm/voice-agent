import glob
import os
from pathlib import Path

# Auto-accept Coqui XTTS v2's CPML license (non-commercial, personal use here) so
# TTS() doesn't block on an interactive prompt when run headless.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "db"


def _find_claude_bin() -> str:
    """Locate the Claude Code CLI native binary bundled with the VS Code extension.

    Globbed rather than hardcoded because the path embeds the extension version,
    which changes on auto-update. Falls back to `claude` on PATH if not found.
    """
    matches = sorted(
        glob.glob(
            str(
                Path.home()
                / ".vscode/extensions/anthropic.claude-code-*-darwin-arm64"
                / "resources/native-binary/claude"
            )
        )
    )
    return matches[-1] if matches else "claude"


# We run the Claude Code CLI itself as the agent's "brain", authenticated via the
# user's existing Claude Code subscription (OAuth) instead of a billed Anthropic
# API key, which the user cannot fund from their country.
CLAUDE_BIN = _find_claude_bin()
# Haiku + low effort trades some reply quality/grammar for much lower per-turn
# latency and cost than Sonnet — acceptable here since replies are short and
# conversational, not code-heavy reasoning.
CLAUDE_MODEL = "haiku"
CLAUDE_EFFORT = "low"

WAKE_WORD = "Рэс"
FOLLOWUP_TIMEOUT_SEC = 7.0

WHISPER_MODEL = "mlx-community/whisper-medium-mlx"
# Pinned to "ru" rather than auto-detect: short/clipped utterances (a bare wake
# word, a 2-3 word command) gave Whisper too little signal to guess the
# language reliably, and a wrong guess produced garbled or English output even
# though the assistant only ever hears Russian.
WHISPER_LANGUAGE = "ru"

# Default location for the instant weather command (used when no city is spoken).
# QUERY is what we send to the weather API (geocodes reliably in English);
# SPOKEN is how the assistant says it aloud (prepositional case).
DEFAULT_CITY_QUERY = "Saint Petersburg"
DEFAULT_CITY_SPOKEN = "в Петербурге"

VOICE_SAMPLE_PATH = BASE_DIR / "voice_sample" / "friend.wav"
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

SAMPLE_RATE = 16000
