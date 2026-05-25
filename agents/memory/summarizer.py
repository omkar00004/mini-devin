# agents/memory/summarizer.py

import sys, os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import (
    MODEL,
    TEMPERATURE_SUMMARY,
    MAX_TOKENS_SUMMARY,
    SUMMARIZE_THRESHOLD,
    KEEP_RECENT,
)

client = Groq(api_key=os.environ["GROQ_API_KEY"])



def should_summarize(messages: list) -> bool:
    non_system = [m for m in messages if m["role"] != "system"]
    return len(non_system) > SUMMARIZE_THRESHOLD


def summarize_messages(messages: list) -> list:
    """
    Compress the middle of the messages list into one summary message.

    BEFORE (18 messages):
    [0]  system
    [1]  user: original task
    [2]  assistant: tool call
    [3]  tool: result
    ...  (middle — gets compressed)
    [14] assistant: tool call   ← recent, kept
    [15] tool: result           ← recent, kept
    [16] assistant: tool call   ← recent, kept
    [17] tool: result           ← recent, kept

    AFTER (7 messages):
    [0]  system
    [1]  user: original task
    [2]  user: SUMMARY of messages 2..13    ← compressed
    [3]  assistant: tool call               ← recent 4, intact
    [4]  tool: result
    [5]  assistant: tool call
    [6]  tool: result
    """
    if not should_summarize(messages):
        return messages

    system_msg  = messages[0]
    user_task   = messages[1]
    middle      = messages[2 : -KEEP_RECENT]
    recent      = messages[-KEEP_RECENT:]

    if not middle:
        return messages

    # Extra LLM call — worth the tradeoff because it preserves all facts
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": (
                "Compress this AI agent work history into one dense paragraph.\n"
                "You MUST preserve:\n"
                "- Every file read (name + key observations about content)\n"
                "- Every fix attempted (succeeded or failed, and why)\n"
                "- Every error seen (exact type, location, message)\n"
                "- Current state of each modified file\n\n"
                "Be specific. Include file names, function names, line numbers, "
                "error types. Do NOT omit failed attempts.\n\n"
                "Messages to compress:\n" +
                json.dumps(middle, indent=2)
            )
        }],
        temperature=0.1,     # very low — we want facts, not creativity
        max_tokens=600
    )

    summary_text = response.choices[0].message.content
    compressed_count = len(middle)

    summary_msg = {
        "role":    "user",
        "content": (
            f"[CONTEXT SUMMARY — {compressed_count} earlier messages compressed]\n"
            f"{summary_text}"
        )
    }

    return [system_msg, user_task, summary_msg] + recent