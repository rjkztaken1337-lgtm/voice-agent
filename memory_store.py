import sqlite3

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _connect():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def append_turn(role: str, content: str):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO turns (role, content) VALUES (?, ?)", (role, content)
        )
    conn.close()


def load_recent_turns(limit: int = config.HISTORY_TURNS_LOADED):
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]
