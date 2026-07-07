"""Yandex Music control — search and play tracks/playlists through the unofficial
`yandex-music` API (the desktop app itself has no AppleScript playback support, so
we fetch and play the audio ourselves via `afplay` instead of driving the app).

Invoked two ways: as fast-path commands from local_actions.py (liked tracks — the
one truly common phrasing) and via the brain's Bash tool for anything free-form
(arbitrary track/playlist names), through the CLI entry point below.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile

from yandex_music import Client

import config

_SECRETS_PATH = config.STATE_DIR / "secrets.json"
_NOW_PLAYING_PATH = config.STATE_DIR / "now_playing.json"

_client = None

# Playing at full system volume means the very first "Влад, ..." said over loud
# music can't even be transcribed (VAD/Whisper can't hear the user), and pause()
# can't kick in until AFTER that command is already recognized — chicken-and-egg.
# Capping playback volume keeps the wake word intelligible from the start; the
# per-turn SIGSTOP ducking still handles full silence during conversation.
_PLAYBACK_VOLUME = 35


def _set_system_volume(level: int) -> None:
    try:
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"],
            timeout=3, capture_output=True,
        )
    except Exception:
        pass


def _load_token() -> str:
    try:
        token = json.loads(_SECRETS_PATH.read_text()).get("yandex_music_token")
    except Exception:
        token = None
    if not token:
        raise RuntimeError(
            "Токен Яндекс Музыки не настроен — добавь его в db/secrets.json."
        )
    return token


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(_load_token()).init()
    return _client


def _load_state() -> dict:
    try:
        return json.loads(_NOW_PLAYING_PATH.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    _NOW_PLAYING_PATH.write_text(json.dumps(state))


def _kill_current() -> None:
    state = _load_state()
    pid = state.get("pid")
    if pid:
        # SIGCONT first: a paused (SIGSTOPped) process holds SIGTERM pending
        # until resumed, so killing a paused track would otherwise hang around.
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    path = state.get("path")
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def pause() -> None:
    """Pauses the current track in place (SIGSTOP) so the mic can hear the user
    over it — the wake-word window ducks to silence rather than just quieter."""
    state = _load_state()
    pid = state.get("pid")
    if pid and not state.get("paused"):
        try:
            os.kill(pid, signal.SIGSTOP)
            state["paused"] = True
            _save_state(state)
        except ProcessLookupError:
            pass


def resume() -> None:
    """Resumes a track paused by pause(). No-op if the state changed underneath
    (e.g. the user's command already stopped or switched tracks)."""
    state = _load_state()
    pid = state.get("pid")
    if pid and state.get("paused"):
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        state["paused"] = False
        _save_state(state)


def _play_track(track, queue=None) -> str:
    _kill_current()
    _set_system_volume(_PLAYBACK_VOLUME)
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    track.download(path)
    proc = subprocess.Popen(["afplay", path])
    _save_state({"pid": proc.pid, "path": path, "queue": queue or []})
    return track.title or "трек"


def play_track(query: str) -> str:
    client = _get_client()
    results = client.search(query, type_="track")
    if not results or not results.tracks or not results.tracks.results:
        return "Не нашёл такой трек в Яндекс Музыке."
    title = _play_track(results.tracks.results[0])
    return f"Включаю {title}."


def play_liked() -> str:
    client = _get_client()
    likes = client.users_likes_tracks()
    track_ids = [t.track_id for t in likes.tracks] if likes and likes.tracks else []
    if not track_ids:
        return "В избранном пока нет треков."
    track = client.tracks([track_ids[0]])[0]
    _play_track(track, queue=track_ids[1:])
    return "Включаю твои любимые треки."


def play_playlist(query: str) -> str:
    client = _get_client()
    results = client.search(query, type_="playlist")
    if not results or not results.playlists or not results.playlists.results:
        return "Не нашёл такой плейлист."
    playlist = results.playlists.results[0]
    full = client.users_playlists(playlist.kind, playlist.owner.uid)
    track_ids = [t.track_id for t in full.tracks] if full and full.tracks else []
    if not track_ids:
        return "В этом плейлисте нет треков."
    track = client.tracks([track_ids[0]])[0]
    _play_track(track, queue=track_ids[1:])
    return f"Включаю плейлист {playlist.title}."


def stop() -> str:
    _kill_current()
    _save_state({})
    return "Останавливаю музыку."


def next_track() -> str:
    state = _load_state()
    queue = state.get("queue") or []
    if not queue:
        _kill_current()
        _save_state({})
        return "Плейлист закончился."
    next_id, *rest = queue
    track = _get_client().tracks([next_id])[0]
    _play_track(track, queue=rest)
    return "Следующий трек."


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: music_control.py <play|playlist|liked|stop|next> [query]")
        sys.exit(1)
    cmd, rest = sys.argv[1], " ".join(sys.argv[2:])
    try:
        if cmd == "play":
            print(play_track(rest))
        elif cmd == "playlist":
            print(play_playlist(rest))
        elif cmd == "liked":
            print(play_liked())
        elif cmd == "stop":
            print(stop())
        elif cmd == "next":
            print(next_track())
        else:
            print(f"Неизвестная команда: {cmd}")
            sys.exit(1)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
