"""Instant local actions — things the assistant just DOES on the machine without
waking the slow brain: check the weather via a fast API, open apps, set volume.

handle(text) performs the matching action and returns a spoken confirmation, or
None if the text is none of these (the caller then falls through to the brain).
Any failure also returns None, so the brain can still try — an action command is
only "claimed" here when it actually succeeds.
"""

import json
import re
import subprocess
import urllib.parse
import urllib.request

import apps
import config
import music_control

# --- weather (fast API instead of the brain's slow web search) ---------------
_WEATHER_RE = re.compile(r"погод|градус", re.IGNORECASE)
_CITY_RE = re.compile(r"\bв\s+([а-яё][а-яё\-]+(?:\s+[а-яё\-]+)?)", re.IGNORECASE)
# words that follow "в ..." but aren't a city — don't treat them as a place
_NOT_CITY = {"городе", "сейчас", "сегодня", "завтра", "нём", "нем", "деревне"}


# wttr.in usually returns a Russian description in lang_ru, but not for every
# condition — fall back to this map so the assistant never reads it in English.
_DESC_RU = {
    "sunny": "солнечно", "clear": "ясно",
    "partly cloudy": "переменная облачность", "cloudy": "облачно",
    "overcast": "пасмурно", "mist": "дымка", "fog": "туман", "freezing fog": "туман",
    "patchy rain possible": "местами дождь", "patchy rain nearby": "местами дождь",
    "light drizzle": "морось", "patchy light drizzle": "морось",
    "light rain": "небольшой дождь", "light rain shower": "небольшой дождь",
    "patchy light rain": "небольшой дождь",
    "moderate rain": "дождь", "moderate rain at times": "дождь", "rain shower": "дождь",
    "heavy rain": "сильный дождь", "heavy rain at times": "сильный дождь",
    "torrential rain shower": "ливень", "moderate or heavy rain shower": "сильный дождь",
    "light snow": "небольшой снег", "patchy light snow": "небольшой снег",
    "light snow showers": "небольшой снег",
    "moderate snow": "снег", "snow": "снег", "heavy snow": "сильный снег",
    "light sleet": "мокрый снег", "sleet": "мокрый снег",
    "thundery outbreaks possible": "возможна гроза", "thunderstorm": "гроза",
}


def _has_cyrillic(s):
    return any("а" <= ch <= "я" or ch == "ё" for ch in s.lower())


def _describe_ru(cur):
    # lang_ru is usually Russian, but for some conditions wttr.in leaves English
    # in it — only trust it if it actually contains Cyrillic.
    if cur.get("lang_ru"):
        val = cur["lang_ru"][0]["value"]
        if _has_cyrillic(val):
            return val.lower()
    if cur.get("weatherDesc"):
        eng = cur["weatherDesc"][0]["value"].strip().lower()
        return _DESC_RU.get(eng, "")  # unknown -> skip rather than speak English
    return ""


def _weather(text):
    if not _WEATHER_RE.search(text):
        return None
    m = _CITY_RE.search(text)
    if m and m.group(1).lower() not in _NOT_CITY:
        query = m.group(1).strip()
        spoken = "в " + query
    else:
        query = config.DEFAULT_CITY_QUERY
        spoken = config.DEFAULT_CITY_SPOKEN
    url = "https://wttr.in/" + urllib.parse.quote(query) + "?format=j1&lang=ru"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
        cur = data["current_condition"][0]
    except Exception:
        return None  # let the brain try if the API is down/slow

    temp = cur.get("temp_C", "?")
    feels = cur.get("FeelsLikeC")
    wind = cur.get("windspeedKmph", "?")
    desc = _describe_ru(cur)

    reply = f"Сейчас {spoken} около {temp} градусов"
    if desc:
        reply += f", {desc}"
    reply += f", ветер {wind} километров в час."
    if feels and feels != temp:
        reply += f" Ощущается как {feels}."
    return reply


# --- open an app -------------------------------------------------------------
# Several ways to phrase "open" plus an optional "приложение"/"программу" filler,
# then the app name itself — everything after that is handed to apps.resolve()
# which fuzzy-matches it (surviving STT errors) and remembers what worked.
_OPEN_RE = re.compile(
    r"\b(?:откр[а-яё]*|запуст[а-яё]*|запуск[а-яё]*|запущ[а-яё]*|"
    r"включ[а-яё]*|завед[а-яё]*|завод[а-яё]*|вруб[а-яё]*|разверн[а-яё]*|разворач[а-яё]*)\s+"
    r"(?:мне\s+|пожалуйста\s+)?(?:приложение\s+|программу\s+)?([а-яёa-z][а-яёa-z\s]*)",
    re.IGNORECASE,
)


_TRAILING_FILLER = re.compile(
    r"\s+((?:у\s+меня\s+)?на\s+(?:компьютере|ноутбуке|маке|компе)|пожалуйста|сейчас|мне)\s*$",
    re.IGNORECASE,
)


def _open_app(text):
    m = _OPEN_RE.search(text)
    if not m:
        return None
    name = _TRAILING_FILLER.sub("", m.group(1).strip()).strip()
    app = apps.open_app(name)
    if not app:
        return None  # unknown app -> let the brain figure it out
    return f"Открываю {app}."


# --- open a website (sites with no dedicated native app) ---------------------
# Reuses the same open-verb list; only kicks in when _open_app already found no
# matching installed app, so this never steals app names.
_SITES = {
    "ютуб": "https://youtube.com", "youtube": "https://youtube.com",
    "вк": "https://vk.com", "вконтакте": "https://vk.com",
    "гитхаб": "https://github.com", "github": "https://github.com",
    "яндекс": "https://ya.ru", "почту гугл": "https://mail.google.com",
    "гмайл": "https://mail.google.com",
}


def _open_site(text):
    m = _OPEN_RE.search(text)
    if not m:
        return None
    name = _TRAILING_FILLER.sub("", m.group(1).strip()).strip().lower().replace("ё", "е")
    url = _SITES.get(name)
    if not url:
        return None
    try:
        subprocess.run(["open", url], check=True, timeout=5, capture_output=True)
    except Exception:
        return None
    return f"Открываю {name}."


# --- YouTube search ------------------------------------------------------
# "найди/включи/поставь видео <query> на/в ютубе" -> opens a search-results
# page (not the first video, that needs the YouTube Data API - out of scope).
_YOUTUBE_RE = re.compile(
    r"\b(?:найди|найти|поищи|включи|поставь)\s+(?:видео\s+)?(.+?)\s+"
    r"(?:на|в)\s+ютуб[а-яё]*\b",
    re.IGNORECASE,
)


def _youtube_search(text):
    m = _YOUTUBE_RE.search(text)
    if not m:
        return None
    query = m.group(1).strip()
    if not query:
        return None
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)
    try:
        subprocess.run(["open", url], check=True, timeout=5, capture_output=True)
    except Exception:
        return None
    return f"Ищу на ютубе {query}."


# --- Yandex Music fast-path -----------------------------------------------
# Only the handful of fixed common phrasings are handled here (instant, no
# brain roundtrip); anything else falls through to the brain, which can call
# music_control.py directly via Bash for arbitrary track/playlist names.
_MUSIC_LIKED_RE = re.compile(
    r"\b(?:включи|поставь|сыграй)\s+(?:мои\s+)?(?:любимые\s+треки|избранное)\b",
    re.IGNORECASE,
)
_MUSIC_TRACK_RE = re.compile(
    r"\b(?:включи|поставь)\s+(?:трек|песню)\s+(.+)", re.IGNORECASE,
)
_MUSIC_PLAYLIST_RE = re.compile(
    r"\b(?:включи|поставь)\s+(?:плейлист|подборку)\s+(.+)", re.IGNORECASE,
)
_MUSIC_STOP_RE = re.compile(r"\b(?:стоп|хватит|останови музыку)\b", re.IGNORECASE)
_MUSIC_NEXT_RE = re.compile(r"\b(?:следующий трек|дальше)\b", re.IGNORECASE)


def _music_play(text):
    # Once a music phrasing is recognized below, the command is "claimed" — unlike
    # the generic handle()-level catch (which means "not this handler, try the
    # next"), a failure here is a real error (e.g. a transient network blip
    # calling the Yandex Music API), not a non-match. Falling through to the
    # brain in that case just repeats the same network call ~40x slower with no
    # feedback in between, so report the failure immediately instead.
    #
    # On success we deliberately return "" rather than the confirmation text —
    # music switches feel abrupt/ugly with a spoken "Включаю..." cutting across
    # the track's own audio, so success is silent (logged only) and only a
    # genuine failure gets spoken.
    try:
        if _MUSIC_LIKED_RE.search(text):
            print(f"[local_actions] {music_control.play_liked()}", flush=True)
            return ""
        m = _MUSIC_TRACK_RE.search(text)
        if m:
            print(f"[local_actions] {music_control.play_track(m.group(1).strip())}", flush=True)
            return ""
        m = _MUSIC_PLAYLIST_RE.search(text)
        if m:
            print(f"[local_actions] {music_control.play_playlist(m.group(1).strip())}", flush=True)
            return ""
        if _MUSIC_STOP_RE.search(text):
            print(f"[local_actions] {music_control.stop()}", flush=True)
            return ""
        if _MUSIC_NEXT_RE.search(text):
            print(f"[local_actions] {music_control.next_track()}", flush=True)
            return ""
    except Exception as exc:
        print(f"[local_actions] music command failed: {type(exc).__name__}: {exc}", flush=True)
        return "Не получилось, проблема с интернетом."
    return None


# --- volume ------------------------------------------------------------------
_VOL_SET_RE = re.compile(r"громкость\s+(?:на\s+)?(\d{1,3})")
_VOL_UP_RE = re.compile(r"\b(громче|погромче|прибавь|сделай громче)", re.IGNORECASE)
_VOL_DOWN_RE = re.compile(r"\b(тише|потише|убавь|сделай тише)", re.IGNORECASE)


def _osa(script):
    return subprocess.run(["osascript", "-e", script], timeout=3,
                          capture_output=True, text=True)


def _get_volume():
    out = _osa("output volume of (get volume settings)")
    return int(out.stdout.strip())


def _volume(text):
    try:
        m = _VOL_SET_RE.search(text)
        if m:
            v = max(0, min(100, int(m.group(1))))
            _osa(f"set volume output volume {v}")
            return f"Готово, громкость {v}."
        if _VOL_UP_RE.search(text):
            v = min(100, _get_volume() + 15)
            _osa(f"set volume output volume {v}")
            return "Сделал погромче."
        if _VOL_DOWN_RE.search(text):
            v = max(0, _get_volume() - 15)
            _osa(f"set volume output volume {v}")
            return "Сделал потише."
    except Exception:
        return None
    return None


# Explicit "открой ..." / "громче" win over the weather matcher (so "открой
# погоду" opens an app rather than reading out a forecast). _open_site only
# fires when _open_app already found no installed app for the name. _music_play
# and _youtube_search go before _open_site/_open_app so their triggers aren't
# swallowed by the "включи/поставь X" open-app pattern.
_HANDLERS = (_music_play, _youtube_search, _open_app, _open_site, _volume, _weather)


def handle(text):
    for fn in _HANDLERS:
        try:
            reply = fn(text)
        except Exception:
            reply = None
        if reply is not None:
            return reply
    return None
