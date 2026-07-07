"""Resolve a spoken app name — even one mangled by speech-to-text ("телеграм"
heard as "телеграф") — to a real installed macOS app, and remember what worked.

resolve() tries, in order: things it has already learned, a curated Russian alias
table, a fuzzy match against those aliases (to survive STT errors), and finally a
fuzzy match against the actual installed apps. Every successful open is written
back to a learned map, so the exact way YOU say it becomes an instant hit next
time — the assistant gets better at your speech over use.
"""

import difflib
import json
import os
import subprocess

import config

_APP_DIRS = [
    "/Applications",
    "/System/Applications",
    "/System/Applications/Utilities",
    os.path.expanduser("~/Applications"),
]

# Curated spoken-Russian (and frequent mis-hearings) -> real macOS app name.
_ALIASES = {
    "сафари": "Safari", "браузер": "Safari",
    "хром": "Google Chrome", "гугл хром": "Google Chrome", "гугл": "Google Chrome",
    "телеграм": "Telegram", "телеграмм": "Telegram", "телеграф": "Telegram",
    "телега": "Telegram", "тг": "Telegram",
    "спотифай": "Spotify", "спотик": "Spotify", "спотифай музыку": "Spotify",
    "музыка": "Яндекс Музыка", "музыку": "Яндекс Музыка",
    "яндекс музыка": "Яндекс Музыка", "яндекс музыку": "Яндекс Музыка",
    "яндекс мьюзик": "Яндекс Музыка", "yandex music": "Яндекс Музыка",
    "эпл мьюзик": "Music",
    "заметки": "Notes", "почта": "Mail", "почту": "Mail",
    "календарь": "Calendar", "терминал": "Terminal",
    "настройки": "System Settings", "системные настройки": "System Settings",
    "карты": "Maps", "фото": "Photos", "фотографии": "Photos",
    "калькулятор": "Calculator", "финдер": "Finder", "проводник": "Finder",
    "код": "Visual Studio Code", "вскод": "Visual Studio Code",
    "вижуал студио": "Visual Studio Code", "редактор кода": "Visual Studio Code",
    "дискорд": "Discord", "слак": "Slack", "зум": "zoom.us",
    "ворд": "Microsoft Word", "эксель": "Microsoft Excel", "поверпоинт": "Microsoft PowerPoint",
    "превью": "Preview", "просмотр": "Preview", "напоминания": "Reminders",
    "часы": "Clock", "погода": "Weather", "акции": "Stocks", "книги": "Books",
    "фигма": "Figma", "обсидиан": "Obsidian", "нотион": "Notion",
}

_LEARNED_PATH = config.STATE_DIR / "learned_apps.json"

_installed = None
_learned = None


def _norm(s: str) -> str:
    return s.strip().lower().replace("ё", "е")


def _scan_installed():
    global _installed
    if _installed is None:
        found = {}
        for directory in _APP_DIRS:
            try:
                for name in os.listdir(directory):
                    if name.endswith(".app"):
                        real = name[:-4]
                        found[real.lower()] = real
            except OSError:
                pass
        _installed = found
    return _installed


def _load_learned():
    global _learned
    if _learned is None:
        try:
            _learned = json.loads(_LEARNED_PATH.read_text())
        except Exception:
            _learned = {}
    return _learned


def _remember(key: str, app: str):
    learned = _load_learned()
    if learned.get(key) != app:
        learned[key] = app
        try:
            config.STATE_DIR.mkdir(parents=True, exist_ok=True)
            _LEARNED_PATH.write_text(json.dumps(learned, ensure_ascii=False, indent=2))
        except OSError:
            pass


def resolve(spoken: str):
    """Best-effort spoken name -> real app name, or None if nothing plausible."""
    spoken = _norm(spoken)
    if not spoken:
        return None
    learned = _load_learned()
    if spoken in learned:
        return learned[spoken]
    if spoken in _ALIASES:
        return _ALIASES[spoken]
    # fuzzy over aliases first (survives STT errors like телеграф -> телеграм)
    match = difflib.get_close_matches(spoken, _ALIASES.keys(), n=1, cutoff=0.75)
    if match:
        return _ALIASES[match[0]]
    # then fuzzy over actually-installed app names
    installed = _scan_installed()
    match = difflib.get_close_matches(spoken, installed.keys(), n=1, cutoff=0.7)
    if match:
        return installed[match[0]]
    return None


def open_app(spoken: str):
    """Resolve + launch. Returns the real app name on success (and learns the
    phrasing), else None so the caller can fall through to the brain."""
    app = resolve(spoken)
    if not app:
        return None
    try:
        subprocess.run(["open", "-a", app], check=True, timeout=5, capture_output=True)
    except Exception:
        return None
    _remember(_norm(spoken), app)
    return app
