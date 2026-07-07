import subprocess

import config

TOOL_DEFINITIONS = [
    {
        "name": "run_shell_command",
        "description": (
            "Выполнить shell-команду на ноутбуке пользователя и вернуть stdout/stderr. "
            "Используй для открытия приложений, чтения списков файлов, запуска скриптов и т.п."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Команда для bash"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Прочитать содержимое текстового файла по абсолютному пути.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Записать текст в файл по абсолютному пути (перезаписывает).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
]


def is_dangerous(command: str) -> bool:
    return any(pattern in command for pattern in config.DANGEROUS_PATTERNS)


def run_shell_command(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        return output.strip() or "(команда выполнена, пустой вывод)"
    except subprocess.TimeoutExpired:
        return "Команда превысила таймаут 60 секунд"


def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except OSError as e:
        return f"Ошибка чтения файла: {e}"


def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w") as f:
            f.write(content)
        return f"Файл {path} записан ({len(content)} символов)"
    except OSError as e:
        return f"Ошибка записи файла: {e}"


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "run_shell_command":
        return run_shell_command(tool_input["command"])
    if name == "read_file":
        return read_file(tool_input["path"])
    if name == "write_file":
        return write_file(tool_input["path"], tool_input["content"])
    return f"Неизвестный инструмент: {name}"
