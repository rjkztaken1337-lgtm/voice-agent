import queue
import re
import threading

import agent
import audio_io
import config
import fastpath
import local_actions
import music_control
import openwakeword_wake
import stt
import tts
from wake_listener import AudioCapture

_WAKE_PREFIX_RE = re.compile(r"^\s*(?:vlad|влад[а-яё]*)[,!.]?\s*", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
# A sentence boundary in a live token stream: terminal punctuation (optionally
# closing a quote/paren) immediately followed by whitespace. "3.5" / "5,5" won't
# match (no space after), so numbers stay intact.
_SENTENCE_END = re.compile(r"[.!?…]+[)\"»']?\s")
_STREAM_DONE = object()


def _split_sentences(text: str):
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def speak(text: str, language: str = "ru"):
    """Synthesizes and plays sentence-by-sentence: the next sentence synthesizes
    in the background while the current one plays, so long replies start
    speaking almost immediately instead of waiting on the full synthesis."""
    print(f"[Влад] {text}", flush=True)
    sentences = _split_sentences(text)
    audio_q = queue.Queue(maxsize=1)

    def produce():
        try:
            for sentence in sentences:
                audio_q.put(tts.synthesize_array(sentence, language=language))
        except Exception as exc:
            audio_q.put(exc)
        finally:
            audio_q.put(_STREAM_DONE)

    threading.Thread(target=produce, daemon=True).start()
    player = audio_io.StreamPlayer(tts.OUTPUT_SAMPLE_RATE)
    try:
        _drain_audio(audio_q, player)
    finally:
        player.close()


def speak_cached(text: str, language: str = "ru"):
    print(f"[Влад] {text}", flush=True)
    audio_io.play_wav(tts.get_cached(text, language=language))


def _strip_wake_prefix(text: str) -> str:
    """Best-effort strip of a leading "Vlad"/"Влад" from text transcribed out of
    record_after_wake()'s audio, which starts at the wake word itself since
    detection is now acoustic (Porcupine), not over already-transcribed text."""
    return _WAKE_PREFIX_RE.sub("", text, count=1).strip()


def _start_stream_synth(chunk_iter, language: str = "ru"):
    """Consumes a stream of text chunks from the brain, assembles them into whole
    sentences, and synthesizes each sentence in a background thread as soon as it
    completes — pushing ready audio onto a queue. Returns the queue immediately so
    synthesis (and the brain subprocess feeding it) start running right away,
    before the caller begins playback."""
    audio_q = queue.Queue(maxsize=2)

    def produce():
        buf = ""
        try:
            for chunk in chunk_iter:
                buf += chunk
                while True:
                    match = _SENTENCE_END.search(buf)
                    if not match:
                        break
                    sentence = buf[: match.end()].strip()
                    buf = buf[match.end():]
                    if sentence:
                        print(f"[Влад] {sentence}", flush=True)
                        audio_q.put(tts.synthesize_array(sentence, language=language))
            tail = buf.strip()
            if tail:
                print(f"[Влад] {tail}", flush=True)
                audio_q.put(tts.synthesize_array(tail, language=language))
        except Exception as exc:
            audio_q.put(exc)
        finally:
            audio_q.put(_STREAM_DONE)

    threading.Thread(target=produce, daemon=True).start()
    return audio_q


def _drain_audio(audio_q, player):
    while True:
        item = audio_q.get()
        if item is _STREAM_DONE:
            return
        if isinstance(item, Exception):
            raise item
        player.play(item)


def speak_stream(chunk_iter, language: str = "ru"):
    audio_q = _start_stream_synth(chunk_iter, language=language)
    player = audio_io.StreamPlayer(tts.OUTPUT_SAMPLE_RATE)
    try:
        _drain_audio(audio_q, player)
    finally:
        player.close()


def handle_command(command_text: str):
    print(f"[вы] {command_text}", flush=True)

    # Fast path: small talk and clock/date questions answer instantly and locally,
    # with no filler and without waking the heavy brain. Isolated defensively — a
    # bug here must never take down the assistant, so fall through to the brain.
    try:
        hit = fastpath.respond(command_text)
    except Exception:
        hit = None
    if hit is not None:
        reply, static = hit
        (speak_cached if static else speak)(reply)
        return

    # Instant local actions: weather (fast API, not the slow web search), opening
    # apps, volume. These DO something and confirm it, all without the brain.
    try:
        action_reply = local_actions.handle(command_text)
    except Exception:
        action_reply = None
    if action_reply is not None:
        # These are all short, low-cardinality confirmations (volume/open-app/
        # weather-template text) — cache-by-text so repeats are instant instead
        # of paying live XTTS synthesis every time.
        speak_cached(action_reply)
        return

    # Real task -> the brain. Start the subprocess + sentence synthesis right away,
    # then only speak a filler if it's actually slow to the first sentence, so
    # quick answers don't get a needless "Секунду".
    audio_q = _start_stream_synth(agent.run_agent_turn_streaming(command_text))
    player = audio_io.StreamPlayer(tts.OUTPUT_SAMPLE_RATE)
    try:
        _drain_audio(audio_q, player)
    finally:
        player.close()


def converse(capture: AudioCapture):
    """After a command is handled, keep listening for follow-ups for a short
    window without requiring the wake word again. Falls back to wake-word mode
    once the user goes quiet.

    Re-ducks before every listen, not just once at entry: the command that got
    us here may have just started a fresh (unpaused) track — e.g. "Влад, включи
    любимые треки" — or a previous follow-up may have (next track), so each
    turn needs its own pause() to catch whatever is currently playing."""
    while True:
        try:
            music_control.pause()
        except Exception:
            pass
        audio = capture.listen_for_utterance(timeout=config.FOLLOWUP_TIMEOUT_SEC)
        if audio is None or audio.size == 0:
            return
        text = stt.transcribe(audio)
        if not text:
            return
        handle_command(text)


def main():
    print(f"Слушаю. Скажи '{config.WAKE_WORD}' чтобы начать. Ctrl+C для выхода.", flush=True)
    tts.warm_up()
    for phrase in ["Да?", "Сделал погромче.", "Сделал потише."]:
        tts.get_cached(phrase)

    detector = openwakeword_wake.create_from_config()
    capture = AudioCapture()
    capture.start()

    def _duck_on_wake():
        # Fires the instant Porcupine hits, before any transcription happens —
        # this is the whole point of the acoustic wake word over the old
        # regex-over-transcript approach, which couldn't duck until the full
        # wake utterance had already been recorded and transcribed.
        try:
            music_control.pause()
        except Exception:
            pass

    try:
        while True:
            capture.wait_for_wake(detector, on_detected=_duck_on_wake)

            try:
                audio = capture.record_after_wake()
                text = _strip_wake_prefix(stt.transcribe(audio)) if audio is not None and audio.size else ""
                if not text:
                    speak_cached("Да?")
                    audio2 = capture.listen_for_utterance(timeout=config.FOLLOWUP_TIMEOUT_SEC)
                    command_text = stt.transcribe(audio2) if audio2 is not None else ""
                    if not command_text:
                        continue
                    handle_command(command_text)
                else:
                    handle_command(text)

                converse(capture)
            finally:
                try:
                    music_control.resume()
                except Exception:
                    pass
    finally:
        detector.delete()
        capture.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
