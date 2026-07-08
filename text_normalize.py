"""Text preprocessing applied to LLM/template output immediately before it
reaches XTTS synthesis: digit-to-words normalization, and (optional,
toggleable) Russian stress-mark injection via ruaccent.

Order matters: numbers must be converted to words BEFORE stress marks are
added — ruaccent operates on Cyrillic words, not digit strings, so running it
before number conversion would leave digits completely unaccented.
"""

import re

from num2words import num2words

import config

# Matches a signed integer or decimal (dot or comma) not adjacent to another
# word character, a colon, or a run of digit-separator punctuation — so clock
# times like "14:30" (both sides border a colon) and mid-word hyphens are
# left untouched.
_NUMBER_RE = re.compile(r"(?<![\w.,:])-?\d+(?:[.,]\d+)?(?![\w.,:])")


def normalize_numbers(text: str) -> str:
    """Cardinal numbers only — covers weather temperatures, track/playlist
    counts, simple quantities. Clock times ("14:30") and calendar dates are
    deliberately left untouched: correct Russian time/date reading has its
    own grammar (e.g. "без пятнадцати три") that num2words doesn't model."""

    def _convert(m: re.Match) -> str:
        raw = m.group(0)
        value = raw.replace(",", ".")
        try:
            number = float(value) if "." in value else int(value)
            return num2words(number, lang="ru")
        except (ValueError, NotImplementedError):
            return raw

    return _NUMBER_RE.sub(_convert, text)


# --- ruaccent stress-mark injection (OFF by default — see config.TTS_USE_RUACCENT) ---
#
# Risk: this was investigated once before on this project and abandoned with
# a note about "degraded intonation" — but nothing was actually implemented
# or committed then, so there's no real before/after data behind that claim.
# This is a genuine retry. Concretely: ruaccent.process_all() inserts a
# literal "+" right before the stressed vowel (e.g. "прив+ет"), a convention
# built for Silero TTS. Coqui XTTS v2's Russian pipeline was not trained on
# it, so behavior here is unverified — hence an isolated, one-flag-toggleable
# step rather than baking it in unconditionally.

_accentizer = None


def _get_accentizer():
    global _accentizer
    if _accentizer is None:
        from ruaccent import RUAccent  # lazy: zero cost when the toggle is off

        _accentizer = RUAccent()
        _accentizer.load(
            omograph_model_size="turbo3.1",
            use_dictionary=True,
            tiny_mode=False,
            device="CPU",
        )
    return _accentizer


def add_stress_marks(text: str) -> str:
    return _get_accentizer().process_all(text)


def prepare_for_tts(text: str) -> str:
    """Single entry point tts.py calls before synthesis."""
    if config.TTS_NORMALIZE_NUMBERS:
        text = normalize_numbers(text)
    if config.TTS_USE_RUACCENT:
        text = add_stress_marks(text)
    return text


def warm_up():
    """Preloads the ruaccent model at startup (if enabled), mirroring
    tts.warm_up() — so the first real reply doesn't pay for it."""
    if config.TTS_USE_RUACCENT:
        _get_accentizer()
