import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-5"

# Push-to-talk key, see pynput.keyboard.Key names (e.g. "alt_r", "cmd_r", "f13")
HOTKEY = "alt_r"

WHISPER_MODEL = "mlx-community/whisper-medium-mlx"
WHISPER_LANGUAGE = None  # None = auto-detect between ru/en

VOICE_SAMPLE_PATH = BASE_DIR / "voice_sample" / "friend.wav"
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

DB_PATH = BASE_DIR / "db" / "agent_memory.sqlite3"
HISTORY_TURNS_LOADED = 20

SAMPLE_RATE = 16000

# Shell commands containing these substrings require spoken confirmation
DANGEROUS_PATTERNS = [
    "rm ", "rm -", "sudo", "git push --force", "git reset --hard",
    "mkfs", "diskutil erase", "> /dev/", "shutdown", "reboot",
    "kill -9", "pkill", "chmod -R", "chown -R",
]
