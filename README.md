# voice-agent

Голосовой ассистент на push-to-talk с клонированным голосом (Coqui XTTS v2) и мозгом на Claude (Anthropic API). Локальные STT/TTS, память между сессиями в SQLite, может выполнять shell-команды, читать/писать файлы и искать в интернете.

## Установка

1. Python 3.11 через Homebrew (`brew install python@3.11`), venv в `.venv/`.
2. `pip install -r requirements.txt`.
3. Скопировать `.env.example` в `.env` и вписать `ANTHROPIC_API_KEY`.
4. Положить чистую запись голоса друга (10–30 сек, без шума/музыки, WAV) в `voice_sample/friend.wav`.
5. При первом запуске macOS спросит разрешения на микрофон и Accessibility/Input Monitoring — разрешить.

## Запуск

```
source .venv/bin/activate
python main.py
```

Удерживай горячую клавишу (по умолчанию правый Option, см. `config.py:HOTKEY`), говори, отпусти — агент ответит голосом друга.

## Безопасность

Команды из `config.DANGEROUS_PATTERNS` (rm, sudo, force-push и т.п.) требуют голосового подтверждения перед выполнением.
