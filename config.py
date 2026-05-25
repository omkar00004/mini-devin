# config.py

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL               = "llama-3.3-70b-versatile"  # Primary LLM used for agents and summarization

# ── Agent Performance & Limits ───────────────────────────────────────────────
MAX_STEPS           = 20    # Maximum steps CoderAgent can run in a ReAct loop
MAX_TOOL_RETRIES    = 2     # Number of retries to fix malformed or failed tool calls

# ── Context Memory Summarization ────────────────────────────────────────────
SUMMARIZE_THRESHOLD = 15    # Trigger summarization when non-system messages exceed this count
KEEP_RECENT         = 4     # Number of most recent messages to keep intact after summarization
MAX_TOKENS_SUMMARY  = 600   # Maximum token limit for the generated context summary paragraph

# ── Planner Agent Polling ───────────────────────────────────────────────────
POLL_INTERVAL_SEC   = 2     # Time in seconds between polling CoderAgent for task status
POLL_TIMEOUT_SEC    = 300   # Maximum time in seconds to wait before a task poll times out

# ── Temperature (Creativity/Deterministic Settings) ──────────────────────────
TEMPERATURE_AGENT   = 0.2   # Temperature for CoderAgent (low for logical precision)
TEMPERATURE_PLANNER = 0.1   # Temperature for PlannerAgent (extremely low for deterministic planning)
TEMPERATURE_SUMMARY = 0.1   # Temperature for summarizer (extremely low for pure factual accuracy)