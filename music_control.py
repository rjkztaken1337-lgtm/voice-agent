"""Yandex Music control — search and play tracks/playlists through the unofficial
`yandex-music` API (the desktop app itself has no AppleScript playback support, so
we fetch and play the audio ourselves instead of driving the app).

Invoked two ways: as fast-path commands from local_actions.py (liked tracks — the
one truly common phrasing) and via the brain's Bash tool for anything free-form
(arbitrary track/playlist names), through the CLI entry point below.

Playback runs through a persistent `mpv --idle` process controlled over its
IPC unix socket, rather than one `afplay` subprocess per track. This is what
lets both call sites (this long-running process AND each fresh
`music_control.py play ...` invocation the brain spawns) coordinate control of
"what's playing right now" through one shared player instead of pid/signal
bookkeeping in now_playing.json. It also means:
- playback volume is mpv's own internal 0-130 "volume" property, entirely
  decoupled from the real macOS system output volume (no more touching real
  system volume at all, ever);
- track switches are `loadfile ... replace` inside the same running process,
  never a mid-waveform SIGTERM — combined with a short fade-out/fade-in this
  removes the audible click that used to happen on "next track".
"""

import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from yandex_music import Client

import config

_SECRETS_PATH = config.STATE_DIR / "secrets.json"
_NOW_PLAYING_PATH = config.STATE_DIR / "now_playing.json"
_MPV_SOCKET_PATH = config.STATE_DIR / "mpv.sock"
_MPV_LOCK_PATH = config.STATE_DIR / "mpv.lock"
_MPV_LOG_PATH = config.STATE_DIR / "mpv.log"

_client = None

# Playing at full volume means the very first "Рэс, ..." said over loud music
# can't even be transcribed (VAD/Whisper can't hear the user), and pause()
# can't kick in until AFTER that command is already recognized — chicken-and-
# egg. Capping playback volume keeps the wake word intelligible from the
# start. This is mpv's own internal volume property (0-130 scale), NOT the
# real macOS system output volume — ear-test after changes here, this number
# doesn't necessarily map to the same perceived loudness as the old
# macOS-volume-based cap did.
_PLAYBACK_VOLUME = 35

# Real fade (not an instant jump) between silence and _PLAYBACK_VOLUME on every
# track switch / pause / resume — this is what makes switches click-free.
_FADE_SEC = 0.18
_FADE_STEP_SEC = 0.02

_PLAYBACK_START_TIMEOUT = 0.8  # max wait for a newly loaded file to start producing audio
_IPC_TIMEOUT = 3               # socket timeout per command, seconds

# A brand-new brew install's mpv binary can take several seconds on its very
# first-ever launch (macOS Gatekeeper scans a freshly installed binary before
# letting it run) — confirmed by hand: first launch ~2.2s, every launch after
# that ~0.1s. 2s was too tight and made that one-time cold start look like a
# real failure, so this is generous specifically to absorb that one-off cost.
_SPAWN_WAIT_TIMEOUT = 8


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


# --- mpv IPC client ----------------------------------------------------------

def _connect() -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(_IPC_TIMEOUT)
    try:
        sock.connect(str(_MPV_SOCKET_PATH))
    except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
        sock.close()
        raise RuntimeError("Не удалось соединиться с mpv.") from exc
    return sock


def _request(sock: socket.socket, command: list, request_id: int = 1) -> dict:
    payload = json.dumps({"command": command, "request_id": request_id}) + "\n"
    sock.sendall(payload.encode())
    f = sock.makefile("r")
    while True:
        line = f.readline()
        if not line:
            raise RuntimeError("mpv закрыл соединение без ответа.")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("request_id") == request_id and "event" not in msg:
            if msg.get("error") != "success":
                raise RuntimeError(f"mpv: {msg.get('error')}")
            return msg
        # else: an unsolicited event line, or a reply to some other request — skip it.


def _cmd(command: list, sock: socket.socket = None):
    """One-shot helper: opens its own connection unless one is passed in (fades
    reuse a single connection across many calls to avoid per-step reconnect cost)."""
    owns_sock = sock is None
    sock = sock or _connect()
    try:
        return _request(sock, command).get("data")
    finally:
        if owns_sock:
            sock.close()


def _mpv_alive() -> bool:
    try:
        with _connect() as sock:
            _request(sock, ["get_property", "mpv-version"])
        return True
    except Exception:
        return False


def _ensure_mpv() -> None:
    if _mpv_alive():
        return
    config.STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_fh = open(_MPV_LOCK_PATH, "a+")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        if _mpv_alive():  # someone else may have spawned it while we waited for the lock
            return
        if _MPV_SOCKET_PATH.exists():
            _MPV_SOCKET_PATH.unlink()  # stale socket left by a dead mpv
        if shutil.which("mpv") is None:
            raise RuntimeError("mpv не установлен — выполни: brew install mpv")
        log_fh = open(_MPV_LOG_PATH, "ab")
        subprocess.Popen(
            ["mpv", "--idle=yes", "--no-video", "--no-terminal", "--really-quiet",
             f"--input-ipc-server={_MPV_SOCKET_PATH}", "--volume=0"],
            stdin=subprocess.DEVNULL, stdout=log_fh, stderr=log_fh,
            start_new_session=True,  # survives whichever process spawned it exiting
        )
        deadline = time.monotonic() + _SPAWN_WAIT_TIMEOUT
        while time.monotonic() < deadline:
            if _mpv_alive():
                return
            time.sleep(0.05)
        raise RuntimeError("mpv не запустился вовремя.")
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


# --- fade / playback helpers --------------------------------------------------

def _get_volume_safe(sock: socket.socket) -> float:
    try:
        return float(_cmd(["get_property", "volume"], sock=sock))
    except Exception:
        return 0.0


def _fade(sock: socket.socket, start: float, end: float, duration: float = _FADE_SEC) -> None:
    steps = max(4, round(duration / _FADE_STEP_SEC))
    for i in range(1, steps + 1):
        v = start + (end - start) * i / steps
        _cmd(["set_property", "volume", round(v, 1)], sock=sock)
        if i < steps:
            time.sleep(_FADE_STEP_SEC)


def _wait_for_playback_start(sock: socket.socket, timeout: float = _PLAYBACK_START_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _cmd(["get_property", "playback-time"], sock=sock)
            return
        except Exception:
            time.sleep(0.02)
    # Give up silently — volume is still 0 at this point, so the fade-in below
    # just starts slightly before/while the file is still buffering, which is
    # harmless (no click either way).


def _load_and_play(path: str, queue: list = None) -> None:
    _ensure_mpv()
    with _connect() as sock:
        old_path = None
        try:
            old_path = _cmd(["get_property", "path"], sock=sock)
        except Exception:
            pass  # nothing was loaded (idle mpv) — nothing to clean up
        cur_vol = _get_volume_safe(sock)
        _fade(sock, cur_vol, 0)
        _cmd(["loadfile", path, "replace"], sock=sock)
        _wait_for_playback_start(sock)
        _fade(sock, 0, _PLAYBACK_VOLUME)
    _save_state({"queue": queue or []})
    if old_path and old_path != path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except OSError:
            pass


def _download_and_play(track, queue: list = None) -> None:
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    track.download(path)
    _load_and_play(path, queue=queue)


def pause() -> None:
    """Fades and pauses the current track in place so the mic can hear the user
    over it — the wake-word window ducks to silence rather than just quieter."""
    try:
        _ensure_mpv()
        with _connect() as sock:
            if _cmd(["get_property", "idle-active"], sock=sock):
                return  # nothing loaded
            if _cmd(["get_property", "pause"], sock=sock):
                return  # already paused
            cur_vol = _get_volume_safe(sock)
            _fade(sock, cur_vol, 0)
            _cmd(["set_property", "pause", True], sock=sock)
    except Exception:
        pass


def resume() -> None:
    """Resumes a track paused by pause(). No-op if nothing is paused (e.g. the
    user's command already stopped or switched tracks)."""
    try:
        _ensure_mpv()
        with _connect() as sock:
            if not _cmd(["get_property", "pause"], sock=sock):
                return
            _cmd(["set_property", "pause", False], sock=sock)
            _fade(sock, 0, _PLAYBACK_VOLUME)
    except Exception:
        pass


def is_playing() -> bool:
    """True if a track is currently loaded and not paused — used by main.py to
    skip the post-command follow-up window (which would otherwise duck this
    same track for the whole timeout right after it started)."""
    try:
        with _connect() as sock:
            idle = _cmd(["get_property", "idle-active"], sock=sock)
            paused = _cmd(["get_property", "pause"], sock=sock)
        return (not idle) and (not paused)
    except Exception:
        return False


def play_track(query: str) -> str:
    client = _get_client()
    results = client.search(query, type_="track")
    if not results or not results.tracks or not results.tracks.results:
        return "Не нашёл такой трек в Яндекс Музыке."
    track = results.tracks.results[0]
    _download_and_play(track, queue=[])
    return f"Включаю {track.title or 'трек'}."


def play_liked() -> str:
    client = _get_client()
    likes = client.users_likes_tracks()
    track_ids = [t.track_id for t in likes.tracks] if likes and likes.tracks else []
    if not track_ids:
        return "В избранном пока нет треков."
    track = client.tracks([track_ids[0]])[0]
    _download_and_play(track, queue=track_ids[1:])
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
    _download_and_play(track, queue=track_ids[1:])
    return f"Включаю плейлист {playlist.title}."


def stop() -> str:
    try:
        _ensure_mpv()
        with _connect() as sock:
            path = None
            try:
                path = _cmd(["get_property", "path"], sock=sock)
            except Exception:
                pass
            cur_vol = _get_volume_safe(sock)
            _fade(sock, cur_vol, 0)
            _cmd(["stop"], sock=sock)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    finally:
        _save_state({})
    return "Останавливаю музыку."


def next_track() -> str:
    state = _load_state()
    queue = state.get("queue") or []
    if not queue:
        stop()
        return "Плейлист закончился."
    next_id, *rest = queue
    track = _get_client().tracks([next_id])[0]
    _download_and_play(track, queue=rest)
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
