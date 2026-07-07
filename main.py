import tempfile
import time
from pathlib import Path

from pynput import keyboard

import agent
import audio_io
import config
import memory_store
import stt
import tts

recorder = audio_io.PushToTalkRecorder()
_hotkey = getattr(keyboard.Key, config.HOTKEY, config.HOTKEY)


def speak(text: str, language: str = "ru"):
    print(f"[агент] {text}")
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        tts.synthesize(text, f.name, language=language)
        audio_io.play_wav(f.name)


def make_confirm_fn():
    def confirm(command: str) -> bool:
        speak(f"Собираюсь выполнить: {command}. Подтверждаешь? Скажи да или нет.")
        print("Удерживай хоткей и скажи да/нет...")
        wait_for_press()
        recorder.start()
        wait_for_release()
        audio = recorder.stop()
        answer = stt.transcribe(audio).lower()
        print(f"[подтверждение] {answer}")
        return any(word in answer for word in ("да", "yes", "подтвержда", "выполняй"))

    return confirm


def wait_for_press():
    pressed = {"flag": False}

    def on_press(key):
        if key == _hotkey:
            pressed["flag"] = True
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def wait_for_release():
    def on_release(key):
        if key == _hotkey:
            return False

    with keyboard.Listener(on_release=on_release) as listener:
        listener.join()


def handle_turn():
    audio = recorder.stop()
    user_text = stt.transcribe(audio)
    if not user_text:
        print("(ничего не распознано)")
        return
    print(f"[вы] {user_text}")

    history = memory_store.load_recent_turns()
    reply = agent.run_agent_turn(user_text, history, confirm_fn=make_confirm_fn())

    memory_store.append_turn("user", user_text)
    memory_store.append_turn("assistant", reply)

    speak(reply)


def main():
    print(f"Готов. Удерживай '{config.HOTKEY}' и говори, отпусти чтобы отправить.")
    print("Ctrl+C для выхода.")

    def on_press(key):
        if key == _hotkey and not recorder.is_recording:
            print("[запись...]")
            recorder.start()

    def on_release(key):
        if key == _hotkey and recorder.is_recording:
            print("[обработка...]")
            handle_turn()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
