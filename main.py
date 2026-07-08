import queue
import re
import threading
import time

import agent
import audio_io
import config
import fastpath
import local_actions
import music_control
import stt
import text_normalize
import tts
from wake_listener import AudioCapture

_WAKE_RE = re.compile(r"\b(?:привет[,!.]?\s*)?р[еэ]с[а-яё]*\b[,!.]?\s*", re.IGNORECASE)
# Bare volume triggers (no wake word) — anchored to the WHOLE utterance so
# "Рэс, погромче" still goes through the normal split_wake_word() path (there's
# text before the match) rather than being double-handled here.
_BARE_VOL_UP_RE = re.compile(r"^(?:громче|погромче)[.!?]*$", re.IGNORECASE)
_BARE_VOL_DOWN_RE = re.compile(r"^(?:тише|потише)[.!?]*$", re.IGNORECASE)
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
    print(f"[Рэс] {text}", flush=True)
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
    print(f"[Рэс] {text}", flush=True)
    audio_io.play_wav(tts.get_cached(text, language=language))


def split_wake_word(text: str):
    """Looks for the wake word anywhere in a transcribed utterance. Returns the
    text after it (may be empty, e.g. a bare "Рэс") if found, or None if the
    wake word wasn't said at all."""
    match = _WAKE_RE.search(text)
    if not match:
        return None
    return text[match.end():].strip()


def _start_stream_synth(chunk_iter, language: str = "ru"):
    """Consumes a stream of text chunks from the brain, assembles them into whole
    sentences, and synthesizes each sentence as soon as it completes — pushing
    ready audio onto a queue. Returns the queue immediately so synthesis (and the
    brain subprocess feeding it) start running right away, before the caller
    begins playback.

    Splitting into brain and synth threads (rather than one thread doing both)
    matters because tts.synthesize_array() is a multi-second blocking call: if
    the same thread that reads chunk_iter also synthesizes, draining the brain's
    subprocess stdout stalls for the full synth duration between sentences,
    which measurably added the sum of all per-sentence synth times on top of the
    brain's own generation time (confirmed via [timing] instrumentation: wall
    matched first_token + sum(tts_synth) almost exactly). With separate threads,
    the brain can keep generating/streaming the next sentence's text while the
    current one is still synthesizing."""
    audio_q = queue.Queue(maxsize=2)
    sentence_q = queue.Queue()

    def collect():
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
                        sentence_q.put(sentence)
            tail = buf.strip()
            if tail:
                sentence_q.put(tail)
        except Exception as exc:
            sentence_q.put(exc)
        finally:
            sentence_q.put(_STREAM_DONE)

    def synthesize():
        try:
            while True:
                item = sentence_q.get()
                if item is _STREAM_DONE:
                    return
                if isinstance(item, Exception):
                    audio_q.put(item)
                    return
                print(f"[Рэс] {item}", flush=True)
                t_synth = time.monotonic()
                audio_q.put(tts.synthesize_array(item, language=language))
                print(f"[timing] tts_synth={round((time.monotonic() - t_synth) * 1000)}ms len={len(item)}", flush=True)
        except Exception as exc:
            audio_q.put(exc)
        finally:
            audio_q.put(_STREAM_DONE)

    threading.Thread(target=collect, daemon=True).start()
    threading.Thread(target=synthesize, daemon=True).start()
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
        # Empty string means the action ran silently (e.g. music commands, which
        # sound ugly cutting across the track's own audio) — nothing to speak.
        # Non-empty replies are short, low-cardinality confirmations (volume/
        # open-app/weather-template text) — cache-by-text so repeats are instant
        # instead of paying live XTTS synthesis every time.
        if action_reply:
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
    us here may have just started a fresh (unpaused) track — e.g. "Рэс, включи
    любимые треки" — or a previous follow-up may have (next track), so each
    turn needs its own pause() to catch whatever is currently playing."""
    while True:
        try:
            music_control.pause()
        except Exception:
            pass
        capture.flush()
        audio = capture.listen_for_utterance(timeout=config.FOLLOWUP_TIMEOUT_SEC)
        if audio is None or audio.size == 0:
            return
        text = stt.transcribe(audio)
        if not text:
            return
        handle_command(text)
        try:
            if music_control.is_playing():
                return
        except Exception:
            pass


def main():
    print(f"Слушаю. Скажи '{config.WAKE_WORD}' чтобы начать. Ctrl+C для выхода.", flush=True)
    tts.warm_up()
    text_normalize.warm_up()
    for phrase in ["Да?", "Сделал погромче.", "Сделал потише."]:
        tts.get_cached(phrase)

    capture = AudioCapture()
    capture.start()

    try:
        while True:
            audio = capture.listen_for_utterance()
            if audio is None or audio.size == 0:
                continue
            text = stt.transcribe(audio)
            if not text:
                continue

            # Bare volume triggers ("Громче"/"Тише" alone, no wake word) — checked
            # on the raw transcript, before split_wake_word, and anchored to the
            # WHOLE utterance so they can't fire on a stray word inside an
            # unrelated sentence. Fire-and-forget: no duck/resume, no follow-up
            # window, since there's no command boundary to bracket.
            stripped = text.strip()
            if _BARE_VOL_UP_RE.match(stripped) or _BARE_VOL_DOWN_RE.match(stripped):
                reply = local_actions.handle(stripped)
                if reply:
                    speak_cached(reply)
                capture.flush()
                continue

            # Wake word is matched over the transcript, not acoustically — ducking
            # can only start once transcription confirms the wake word was said.
            command_text = split_wake_word(text)
            if command_text is None:
                continue

            try:
                music_control.pause()
            except Exception:
                pass

            try:
                if not command_text:
                    speak_cached("Да?")
                    capture.flush()
                    audio2 = capture.listen_for_utterance(timeout=config.FOLLOWUP_TIMEOUT_SEC)
                    command_text2 = stt.transcribe(audio2) if audio2 is not None else ""
                    if not command_text2:
                        continue
                    handle_command(command_text2)
                else:
                    handle_command(command_text)

                # A command that just started/left music playing (e.g. "включи
                # любимые треки") shouldn't immediately duck it again for a
                # follow-up window nobody asked for — that's what froze the
                # track right after it started. Only chase a follow-up when
                # there's no active playback to interrupt.
                if not music_control.is_playing():
                    converse(capture)
                else:
                    capture.flush()
            finally:
                try:
                    music_control.resume()
                except Exception:
                    pass
    finally:
        capture.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
