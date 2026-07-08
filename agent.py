import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import config

_SYSTEM_PROMPT = (
    "Ты — голосовой ассистент пользователя по имени Рэс, работающий на его ноутбуке. "
    "Отвечай кратко и разговорно, как для озвучки вслух — без markdown, звёздочек, списков и таблиц. "
    "Каждое слово твоего ответа немедленно озвучивается TTS, поэтому пиши ТОЛЬКО то, что должно "
    "прозвучать вслух пользователю — никогда не думай вслух и не описывай свои внутренние шаги "
    "('сначала я поищу', 'эта команда ищет в коде', 'нужно проверить файл' и т.п.), не используй "
    "нумерацию (1., 2. ...) и не пиши план действий текстом. Всё это оставляй в вызовах инструментов, "
    "а не в тексте ответа. "
    "Отвечай ВСЕГДА только по-русски — ни отдельных английских слов, ни фраз вроде 'Opening it now' "
    "в ответе быть не должно, даже если промежуточный результат инструмента на английском. "
    "Говори естественным, грамматически правильным разговорным русским языком — как живой носитель "
    "языка, а не дословный перевод с английского: избегай калек, неестественного порядка слов и "
    "режущих слух конструкций. "
    "Можешь выполнять shell-команды, читать и писать файлы, искать в интернете через инструменты. "
    "Если для простого действия (открыть сайт, приложение, посчитать) очевиден прямой способ — "
    "делай его сразу одним вызовом инструмента, не исследуй код проекта и не ищи, как это 'уже "
    "реализовано' — это не нужно и долго. "
    "Если тебя просят выполнить потенциально опасную или необратимую команду (удаление файлов, "
    "форматирование дисков, force push, sudo, kill процессов и т.п.), НЕ вызывай инструмент сразу. "
    "Сначала одним коротким предложением опиши, что именно собираешься сделать, и спроси подтверждение. "
    "Выполняй инструмент только когда пользователь явно согласится в следующем сообщении "
    "('да', 'подтверждаю', 'делай'). "
    "Никогда не проговаривай фразы вроде «секунду», «сейчас сделаю», «одну минуту» и подобные — "
    "сразу вызывай нужный инструмент молча и озвучивай только финальный ответ. "
    "Для фактов и погоды делай ОДИН короткий веб-поиск и сразу отвечай по его результатам — "
    "не открывай сайты по одному и не делай несколько поисков подряд, это долго. "
    "Яндекс Музыка установлена на компьютере, но у неё нет AppleScript-управления — никогда не "
    "говори, что приложение не установлено, и не открывай вместо него браузер. Чтобы включить "
    "трек или плейлист по запросу пользователя, выполни shell-команду (рабочая директория уже "
    "правильная, используй именно venv-интерпретатор, не системный python3): "
    ".venv/bin/python3 music_control.py play \"<запрос>\" — включить трек по названию; "
    ".venv/bin/python3 music_control.py playlist \"<запрос>\" — включить плейлист по названию; "
    ".venv/bin/python3 music_control.py liked — включить любимые треки; "
    ".venv/bin/python3 music_control.py next — следующий трек; "
    ".venv/bin/python3 music_control.py stop — остановить музыку. "
    "Команда сама печатает готовую фразу для ответа пользователю — используй её как основу ответа."
)

_SESSION_ID_PATH = config.STATE_DIR / "cli_session_id.txt"
_TURN_COUNT_PATH = config.STATE_DIR / "cli_turn_count.txt"
_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# --resume re-processes the ENTIRE session transcript as context on every turn,
# so a long-lived session steadily inflates time-to-first-token (measured: an
# unbounded session grew to ~4.6 MB / 260+ events and pushed ttft to ~11s). We
# rotate to a fresh session once it gets large or old, keeping short-term
# follow-up memory while capping the per-turn context — and thus the latency.
_MAX_TRANSCRIPT_BYTES = 400_000
_MAX_TURNS_PER_SESSION = 10


def _agent_subprocess_env() -> dict:
    """Env for the headless `claude -p` subprocess, with its own dedicated proxy
    API key/URL overridden (see config.AGENT_ANTHROPIC_*) instead of inheriting
    whatever key this interactive Claude Code session itself is using. Sharing
    one key between both caused intermittent "Not logged in" errors — the
    proxy appears to enforce a per-key concurrent-session limit."""
    env = os.environ.copy()
    if config.AGENT_ANTHROPIC_API_KEY:
        env["ANTHROPIC_API_KEY"] = config.AGENT_ANTHROPIC_API_KEY
    if config.AGENT_ANTHROPIC_BASE_URL:
        env["ANTHROPIC_BASE_URL"] = config.AGENT_ANTHROPIC_BASE_URL
    return env


def _transcript_size(session_id: str) -> int:
    """Best-effort size of the CLI's transcript for this session. Returns 0 if the
    file can't be located (the projects-dir path scheme is CC-internal), so
    rotation falls back to the turn counter."""
    try:
        project = _PROJECTS_DIR / str(config.BASE_DIR).replace("/", "-")
        transcript = project / f"{session_id}.jsonl"
        return transcript.stat().st_size if transcript.exists() else 0
    except OSError:
        return 0


def _get_session_id():
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    if _TURN_COUNT_PATH.exists():
        try:
            count = int(_TURN_COUNT_PATH.read_text().strip() or "0")
        except ValueError:
            count = 0

    if _SESSION_ID_PATH.exists():
        session_id = _SESSION_ID_PATH.read_text().strip()
        too_big = _transcript_size(session_id) > _MAX_TRANSCRIPT_BYTES
        if session_id and count < _MAX_TURNS_PER_SESSION and not too_big:
            _TURN_COUNT_PATH.write_text(str(count + 1))
            return session_id, True

    session_id = str(uuid.uuid4())
    _SESSION_ID_PATH.write_text(session_id)
    _TURN_COUNT_PATH.write_text("1")
    return session_id, False


def run_agent_turn(user_text: str) -> str:
    """Runs one turn through the Claude Code CLI in headless print mode.

    Session continuity (cross-restart memory) comes from --resume/--session-id.
    Uses its own dedicated proxy API key (see _agent_subprocess_env), separate
    from the interactive Claude Code session's key.
    """
    session_id, resuming = _get_session_id()

    cmd = [
        config.CLAUDE_BIN,
        "-p", user_text,
        "--output-format", "json",
        "--permission-mode", "bypassPermissions",
        "--tools", "Bash,Read,Write,WebSearch",
        "--setting-sources", "",
        "--append-system-prompt", _SYSTEM_PROMPT,
        "--model", config.CLAUDE_MODEL,
        "--effort", config.CLAUDE_EFFORT,
        "--resume" if resuming else "--session-id", session_id,
    ]

    for attempt in range(2):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(config.BASE_DIR),
                env=_agent_subprocess_env(),
            )
        except subprocess.TimeoutExpired:
            return "Что-то зависло, не смог выполнить за отведённое время. Попробуй ещё раз или переформулируй."

        if result.returncode != 0:
            err = result.stderr.strip()
            # "Not logged in" here is a transient OAuth-token race, not a real
            # logout: it can happen when another `claude` CLI invocation reads/
            # refreshes the same credentials file at the same instant. Retrying
            # once after a beat lets the token settle instead of surfacing a
            # spurious login error to the user.
            if "Not logged in" in err and attempt == 0:
                time.sleep(1.5)
                continue
            return f"Ошибка агента: {err[:300] or 'неизвестная ошибка'}"

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return result.stdout.strip() or "Пустой ответ от агента."

        if data.get("is_error"):
            msg = data.get("result", "неизвестная ошибка")
            if "Not logged in" in str(msg) and attempt == 0:
                time.sleep(1.5)
                continue
            return f"Ошибка агента: {msg}"

        return (data.get("result") or "").strip() or "Готово."


def run_agent_turn_streaming(user_text: str):
    """Same turn as run_agent_turn, but yields the reply text incrementally as the
    model generates it, so the caller can start speaking the first sentence while
    the rest is still being written (and while tool calls run).

    Uses --output-format stream-json (which REQUIRES --verbose) plus
    --include-partial-messages to get token-level text deltas. Uses the same
    dedicated proxy API key as run_agent_turn (see _agent_subprocess_env);
    only the output format and stdout parsing differ.
    """
    session_id, resuming = _get_session_id()

    cmd = [
        config.CLAUDE_BIN,
        "-p", user_text,
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--permission-mode", "bypassPermissions",
        "--tools", "Bash,Read,Write,WebSearch",
        "--setting-sources", "",
        "--append-system-prompt", _SYSTEM_PROMPT,
        "--model", config.CLAUDE_MODEL,
        "--effort", config.CLAUDE_EFFORT,
        "--resume" if resuming else "--session-id", session_id,
    ]

    for attempt in range(2):
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(config.BASE_DIR),
            env=_agent_subprocess_env(),
        )
        killer = threading.Timer(180, proc.kill)  # overall wall-clock guard
        killer.start()

        # Temporary diagnostic timing (see systematic-debugging investigation into
        # inconsistent turn latency) — prints wall/API time, tool-call count and
        # names so slow turns can be correlated with tool use vs. plain generation.
        t0 = time.monotonic()
        first_token_t = None
        tool_calls = []

        got_text = False
        result_text = ""
        is_error = False
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = ev.get("type")
                if etype == "stream_event":
                    sub = ev.get("event", {})
                    if sub.get("type") == "content_block_delta":
                        delta = sub.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            if first_token_t is None:
                                first_token_t = time.monotonic()
                                print(f"[timing] first_token={round((first_token_t - t0) * 1000)}ms", flush=True)
                            got_text = True
                            yield delta["text"]
                elif etype == "assistant":
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "tool_use":
                            tool_calls.append(block.get("name", "?"))
                elif etype == "result":
                    result_text = (ev.get("result") or "").strip()
                    is_error = bool(ev.get("is_error"))
                    elapsed = round((time.monotonic() - t0) * 1000)
                    print(
                        f"[timing] wall={elapsed}ms cli_duration={ev.get('duration_ms')}ms "
                        f"api_duration={ev.get('duration_api_ms')}ms turns={ev.get('num_turns')} "
                        f"tools={tool_calls}",
                        flush=True,
                    )
        finally:
            killer.cancel()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

        if got_text:
            return

        # "Not logged in" here is a transient OAuth-token race, not a real
        # logout: it can happen when another `claude` CLI invocation reads/
        # refreshes the same credentials file at the same instant. Retrying
        # once after a beat lets the token settle instead of surfacing a
        # spurious login error to the user.
        if is_error and "Not logged in" in result_text and attempt == 0:
            time.sleep(1.5)
            continue

        # Streamed no text (killed by timeout, crash, or empty reply) — fall back
        # to whatever the final result event carried, else a spoken error.
        if is_error:
            yield f"Ошибка агента: {result_text or 'неизвестная ошибка'}"
        else:
            yield result_text or "Что-то зависло, не смог ответить. Попробуй ещё раз."
        return
