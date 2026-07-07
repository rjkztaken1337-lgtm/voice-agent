from anthropic import Anthropic

import config
import tools

_SYSTEM_PROMPT = (
    "Ты — голосовой ассистент пользователя, работающий на его ноутбуке. "
    "Отвечай кратко и разговорно, как для озвучки вслух — без markdown, списков и таблиц. "
    "Можешь выполнять shell-команды, читать/писать файлы и искать в интернете через инструменты. "
    "Если тебя просят выполнить потенциально опасную или необратимую команду, сначала опиши, "
    "что именно собираешься сделать, одним коротким предложением."
)

_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def run_agent_turn(user_text: str, history: list[dict], confirm_fn=None) -> str:
    """history: list of {"role": "user"/"assistant", "content": str}.
    confirm_fn(command: str) -> bool, asked before dangerous shell commands.
    Returns the assistant's final text reply.
    """
    client = _get_client()
    messages = list(history) + [{"role": "user", "content": user_text}]

    all_tools = tools.TOOL_DEFINITIONS + [_WEB_SEARCH_TOOL]

    while True:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=all_tools,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "run_shell_command":
                command = block.input["command"]
                if tools.is_dangerous(command) and confirm_fn is not None:
                    if not confirm_fn(command):
                        output = "Пользователь отменил выполнение команды."
                    else:
                        output = tools.execute_tool(block.name, block.input)
                else:
                    output = tools.execute_tool(block.name, block.input)
            elif block.name in ("read_file", "write_file"):
                output = tools.execute_tool(block.name, block.input)
            else:
                continue  # server-side tools (web_search) are handled by the API itself
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                }
            )
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
