# agents/memory/episodic.py

import os
import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional

# Database lives outside agents/ so it persists across runs
DB_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../../memory/episodes.db")
)


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id          TEXT PRIMARY KEY,
                signature   TEXT NOT NULL,
                context     TEXT,
                reflection  TEXT,
                fix_applied TEXT,
                outcome     TEXT,           -- 'success' or 'failure'
                created_at  TEXT
            )
        """)
        # Index on signature for fast lookup
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signature ON episodes(signature)"
        )
        conn.commit()


def normalize_signature(
    error_trace:   str,
    function_name: str = "",
    file_name:     str = ""
) -> str:
    """
    Produce a stable lookup key from an error.
    Strips line numbers and specific values — keeps structure.

    "KeyError: 'timeout' at config.py:34 in parse_config"
    → "KeyError::parse_config::config.py"

    Two identical bugs in different repos will match.
    Two different bugs in the same function won't collide if error type differs.
    """
    error_type = error_trace.split(":")[0].strip()
    func = function_name.strip().lower() if function_name else "unknown"
    file = os.path.basename(file_name).strip().lower() if file_name else "unknown"
    return f"{error_type}::{func}::{file}"


def store_episode(
    signature:   str,
    context:     str,    # what the agent was doing
    reflection:  str,    # what assumption was wrong
    fix_applied: str,    # what code change fixed it
    outcome:     str     # "success" or "failure"
) -> str:
    """Store one episode. Returns the episode id."""
    episode_id = str(uuid.uuid4())[:8]
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO episodes
                (id, signature, context, reflection, fix_applied, outcome, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id, signature, context,
                reflection, fix_applied, outcome,
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
    return episode_id


def recall(signature: str, limit: int = 3) -> List[dict]:
    """
    Retrieve past episodes for this signature.
    Successful fixes come first, then most recent.
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM episodes
            WHERE  signature = ?
            ORDER  BY
                CASE outcome WHEN 'success' THEN 0 ELSE 1 END,
                created_at DESC
            LIMIT ?
            """,
            (signature, limit)
        ).fetchall()
    return [dict(row) for row in rows]


def recall_as_context(signature: str) -> str:
    """
    Format past episodes as a hint for the agent.
    Deliberately excludes actual fix code — if we include it, the model
    copy-pastes it as text instead of calling write_file as a tool.
    """
    episodes = recall(signature)
    if not episodes:
        return ""

    lines = [f"[Past experience with similar error '{signature}']"]
    for ep in episodes:
        if ep["outcome"] == "success":
            lines.append(f"  Previously fixed by: {ep['reflection']}")
        else:
            lines.append(f"  Previously attempted: {ep['reflection']} — did not work")
    lines.append(
        "[Use this as a HINT only. You MUST still call tools (read_file, write_file, etc.) "
        "using the proper JSON tool-call format. "
        "Do NOT write <function=...> XML tags. Do NOT output code as plain text.]"
    )
    return "\n".join(lines)