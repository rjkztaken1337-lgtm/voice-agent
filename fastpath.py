"""Instant local answers for small talk and clock/date questions.

The heavy path (Claude Code CLI as the brain) costs several seconds per turn, so
trivial conversational inputs like "привет" or "как дела?" shouldn't go through
it at all — they get an immediate, natural reply here with no filler.

respond(text) returns (reply_text, static) or None:
  - None            -> not small talk; caller falls through to the real brain.
  - (text, True)    -> a fixed reply, safe to cache and play instantly.
  - (text, False)   -> a dynamic reply (time/date); synthesize fresh, don't cache.

SAFETY: confirmation words ("да", "подтверждаю", "делай", "нет") are deliberately
NOT matched here. They may be the user approving a pending dangerous action, so
they must always reach the brain, which owns the confirm-before-acting flow.
"""

import datetime
import random
import re

_PUNCT_RE = re.compile(r"[.,!?;:…\"'«»()\-]+")
_SPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


# Optional politeness/filler words that may wrap an utterance without changing its
# intent ("ну слушай, как дела" -> still just "как дела").
_WRAP = r"(?:ну |слушай |скажи |а |эй |так )*"
_TAIL = r"(?: пожалуйста| друг| братан| дружище)?"


def _intent(pattern: str) -> re.Pattern:
    return re.compile(rf"^{_WRAP}(?:{pattern}){_TAIL}$")


# Each intent: a matcher and a list of interchangeable replies (rotated for variety).
_STATIC_INTENTS = [
    (
        _intent(r"привет|здравствуй(?:те)?|здорово|хай|салют|доброе утро|добрый день|добрый вечер|доброго времени(?: суток)?"),
        ["Привет!", "Привет, я тут."],
    ),
    (
        _intent(r"(?:привет[ ]?)?как (?:дела|ты|сам|жизнь|настроение)(?: у тебя)?|как ты там|как поживаешь|че как|что нового|как оно"),
        ["Всё отлично, готов помогать. А у тебя как?", "Отлично! Чем займёмся?", "Всё супер, жду задачу."],
    ),
    (
        _intent(r"спасибо(?: большое)?|спасиб|благодарю|благодарствую|спс|мерси"),
        ["Пожалуйста!", "Обращайся!", "Всегда рад помочь."],
    ),
    (
        _intent(r"пока|до свидания|до встречи|прощай|бывай|спокойной ночи|доброй ночи"),
        ["Пока! Если что — зови.", "До связи!", "Давай, обращайся."],
    ),
    (
        _intent(r"кто ты|ты кто|как тебя зовут|как тебя звать|представься|назови себя"),
        ["Я Рэс, твой голосовой помощник."],
    ),
    (
        _intent(r"что (?:ты )?умеешь|что (?:ты )?можешь(?: делать)?|твои возможности|чем (?:ты )?можешь помочь"),
        ["Могу искать в интернете, запускать программы, работать с файлами и просто поболтать. Скажи, что нужно."],
    ),
    (
        _intent(r"ты тут|ты здесь|ты меня слышишь|слышишь меня|ты живой|ты работаешь"),
        ["Тут, слушаю.", "Здесь, весь во внимании."],
    ),
]

_TIME_RE = _intent(r"который час|сколько (?:сейчас )?времени|сколько на часах|текущее время|время сейчас|сколько время")
_DATE_RE = _intent(r"какое (?:сегодня )?число|какой (?:сегодня )?день(?: недели)?|какая сегодня дата|сегодня какое число|число сегодня|какой сейчас день")

_ONES = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
          "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят"]
_MONTHS = ["", "января", "февраля", "марта", "апреля", "мая", "июня",
           "июля", "августа", "сентября", "октября", "ноября", "декабря"]
_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


def _spell(n: int, feminine: bool = False) -> str:
    """Spell 0-59 in Russian words so XTTS reads the clock naturally instead of
    voicing raw digits. `feminine` swaps один/два -> одна/две (for 'минута')."""
    def unit(u: int) -> str:
        if feminine and u == 1:
            return "одна"
        if feminine and u == 2:
            return "две"
        return _ONES[u]

    if n < 10:
        return unit(n)
    if n < 20:
        return _TEENS[n - 10]
    tens, u = divmod(n, 10)
    return _TENS[tens] + (f" {unit(u)}" if u else "")


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    d = n % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many


def _time_reply() -> str:
    now = datetime.datetime.now()
    h, m = now.hour, now.minute
    hours = f"{_spell(h)} {_plural(h, 'час', 'часа', 'часов')}"
    if m == 0:
        return f"Сейчас {hours} ровно."
    minutes = f"{_spell(m, feminine=True)} {_plural(m, 'минута', 'минуты', 'минут')}"
    return f"Сейчас {hours} {minutes}."


def _date_reply() -> str:
    now = datetime.datetime.now()
    return f"Сегодня {now.day} {_MONTHS[now.month]}, {_WEEKDAYS[now.weekday()]}."


def respond(text: str):
    norm = _normalize(text)
    if not norm:
        return None
    if _TIME_RE.match(norm):
        return _time_reply(), False
    if _DATE_RE.match(norm):
        return _date_reply(), False
    for matcher, replies in _STATIC_INTENTS:
        if matcher.match(norm):
            return random.choice(replies), True
    return None
