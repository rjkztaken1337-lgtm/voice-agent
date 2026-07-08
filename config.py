import glob
import os
from pathlib import Path

# Auto-accept Coqui XTTS v2's CPML license (non-commercial, personal use here) so
# TTS() doesn't block on an interactive prompt when run headless.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "db"


def _load_env_file(path: Path) -> dict:
    """Minimal KEY=VALUE parser for a local .env — avoids adding python-dotenv
    as a dependency for just two variables."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


# agent.py's headless `claude -p` subprocess uses ITS OWN dedicated proxy API
# key/URL (below), separate from whatever key this interactive Claude Code
# session itself is authenticated with. They used to share the same key via
# ambient environment inheritance, which caused intermittent "Not logged in"
# errors when both hit the proxy at once (proxy enforces per-key session
# limits). Kept in .env (gitignored) rather than hardcoded here.
_agent_env = _load_env_file(BASE_DIR / ".env")
AGENT_ANTHROPIC_API_KEY = _agent_env.get("ANTHROPIC_API_KEY")
AGENT_ANTHROPIC_BASE_URL = _agent_env.get("ANTHROPIC_BASE_URL")


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
# Haiku keeps per-turn latency and cost low (vs. Sonnet) for short,
# conversational replies. Effort raised from "low" to "medium" to spend more
# reasoning budget on Russian grammar/phrasing — watch the [timing] prints in
# agent.py to sanity-check this hasn't made replies feel sluggish.
CLAUDE_MODEL = "haiku"
CLAUDE_EFFORT = "medium"

WAKE_WORD = "Рэс"
FOLLOWUP_TIMEOUT_SEC = 7.0

# large-v3-turbo reuses full large-v3's encoder with a distilled 4-layer
# decoder — near large-v3 accuracy, much faster than large-v3 proper. If live
# testing shows its Russian accuracy isn't good enough, swap to
# "mlx-community/whisper-large-v3-mlx" (full large-v3, no distillation) —
# one-line change, nothing else to touch.
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
# Pinned to "ru" rather than auto-detect: short/clipped utterances (a bare wake
# word, a 2-3 word command) gave Whisper too little signal to guess the
# language reliably, and a wrong guess produced garbled or English output even
# though the assistant only ever hears Russian.
WHISPER_LANGUAGE = "ru"
# Beam search (vs. default greedy) trades latency for accuracy — worth it here
# since we already prioritize accuracy for short commands. 5 is Whisper's own
# standard beam width.
WHISPER_BEAM_SIZE = 5
# Seeds the decoder with domain vocabulary so short, acoustically ambiguous
# utterances (wake word, app/command names, city names) bias toward correct
# spellings instead of the nearest-sounding dictionary word. Only the last
# ~224 tokens influence decoding, so keep this compact.
WHISPER_INITIAL_PROMPT = (
    "Голосовые команды ассистенту по имени Рэс: включи Яндекс Музыку, "
    "поставь плейлист, следующий трек, останови музыку, сделай погромче, "
    "сделай потише, какая погода в Петербурге, в Москве, который час."
)
# False rather than the mlx_whisper default (True): each call transcribes one
# short, self-contained utterance, not a continuous long-form stream, so
# there's no real "previous segment" context worth conditioning on within a
# call — and conditioning risks a bad early guess biasing the rest of it.
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False

# Default location for the instant weather command (used when no city is spoken).
# QUERY is what we send to the weather API (geocodes reliably in English);
# SPOKEN is how the assistant says it aloud (prepositional case).
DEFAULT_CITY_QUERY = "Saint Petersburg"
DEFAULT_CITY_SPOKEN = "в Петербурге"

VOICE_SAMPLE_PATH = BASE_DIR / "voice_sample" / "friend.wav"
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Digit-to-words conversion for TTS input (e.g. "23" -> "двадцать три"). Low
# risk, on by default.
TTS_NORMALIZE_NUMBERS = True
# ruaccent stress-mark injection before TTS. OFF by default: XTTS was not
# trained on ruaccent's "+"-before-stressed-vowel convention (built for Silero
# TTS), so behavior here is unverified and may sound worse, not better — see
# text_normalize.py's module docstring. Flip to True to live-test, flip back
# to revert; no other code changes needed either way.
TTS_USE_RUACCENT = False

SAMPLE_RATE = 16000
