# agents/debugger_agent.py

import os
import sys
import json
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from rich.console import Console
from rich.markup import escape

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_client import call_mcp, list_tools
from models.task_card import TaskCard, TaskArtifact
from models.failure import FailureType, FailureReport, ESCALATION_POLICY
from memory.episodic import init_db, normalize_signature, store_episode
from memory.summarizer import summarize_messages, should_summarize
from test_agent import run_tests
from config import (
    MODEL, MAX_STEPS, TEMPERATURE_AGENT,
    SUMMARIZE_THRESHOLD, KEEP_RECENT
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
FILESYSTEM_MCP_URL = os.environ.get("FILESYSTEM_MCP_URL", "http://localhost:8000")
SHELL_MCP_URL      = os.environ.get("SHELL_MCP_URL",      "http://localhost:8001")
DEBUGGER_PORT      = int(os.environ.get("DEBUGGER_AGENT_PORT", 9002))

client  = Groq(api_key=GROQ_API_KEY)
console = Console(highlight=False)
app     = FastAPI()

task_store: dict[str, dict] = {}
TOOL_REGISTRY: dict[str, str] = {}


# ── Tool setup (same pattern as CoderAgent) ────────────────────────────────

def discover_tools() -> list:
    all_tools = []
    for url in [FILESYSTEM_MCP_URL, SHELL_MCP_URL]:
        tools = list_tools(url)
        for t in tools:
            TOOL_REGISTRY[t["name"]] = url
            all_tools.append(t)
    return all_tools


def convert_to_groq_tools(mcp_tools: list) -> list:
    return [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["inputSchema"]
        }}
        for t in mcp_tools
    ]


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    return json.dumps(
        call_mcp(TOOL_REGISTRY[tool_name], tool_name, arguments),
        indent=2
    )


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{ts}[/dim] {msg}")


# ── Failure Classification ─────────────────────────────────────────────────

def classify_failure(error_text: str, attempts: int) -> FailureType:
    """
    Classify what type of failure occurred.
    This drives the escalation policy — not a threshold count.

    Rules applied in order — first match wins.
    """
    err = error_text.lower()

    # Format errors — model glitch, not a real problem
    if any(x in err for x in ["tool_use_failed", "function=", "xml", "invalid_request"]):
        return FailureType.FORMAT_ERROR

    # Environment errors — wrong Python, missing package
    if any(x in err for x in ["modulenotfounderror", "importerror",
                                "command not found", "no such file or directory",
                                "permission denied"]):
        return FailureType.ENVIRONMENT

    # Scope errors — fix worked but broke something in another file
    if any(x in err for x in ["nameerror", "attributeerror", "cannot import"]):
        return FailureType.SCOPE_TOO_NARROW

    # Wrong approach — tried the fix, tests still fail
    if attempts >= 2:
        return FailureType.WRONG_APPROACH

    return FailureType.UNKNOWN


# ── Structured Reflection ──────────────────────────────────────────────────

def generate_reflection(
    task_description: str,
    previous_fix:     str,
    test_failure:     str,
    file_content:     str
) -> str:
    """
    The core of DebuggerAgent — forces the model to articulate
    what assumption was wrong BEFORE producing any fix.

    This is not prompt engineering — it changes the reasoning mode.
    The model must explain causality, not just pattern-match to a fix.
    """
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{
            "role": "user",
            "content": (
                f"A previous fix attempt failed. Analyze why.\n\n"
                f"Task: {task_description}\n\n"
                f"Previous fix applied:\n{previous_fix}\n\n"
                f"Test failure output:\n{test_failure}\n\n"
                f"Current file content:\n{file_content[:2000]}\n\n"
                f"Answer these THREE questions specifically:\n"
                f"1. What assumption did the previous fix make?\n"
                f"2. What does the test failure prove about that assumption?\n"
                f"3. What is the corrected approach?\n\n"
                f"Be specific. Name exact functions, variables, line numbers."
            )
        }],
        temperature = 0.1,    # want precise analysis, not creative
        max_tokens  = 500,
    )
    return response.choices[0].message.content.strip()


# ── Debug ReAct Loop ───────────────────────────────────────────────────────

def run_debug_loop(
    task_card:       TaskCard,
    test_result:     "TestResult",
    previous_fix:    str,
) -> tuple[str, list, FailureReport | None]:
    """
    DebuggerAgent's ReAct loop.
    Difference from CoderAgent: reflection step happens FIRST,
    before any tool calls. The reflection becomes part of the prompt.

    Returns: (final_answer, artifacts, failure_report_or_None)
    failure_report is None if task succeeded, populated if escalation needed.
    """
    task_id   = task_card.task_id
    mcp_tools  = discover_tools()
    groq_tools = convert_to_groq_tools(mcp_tools)

    # Track attempts for escalation policy
    attempts      = 0
    files_modified = []
    last_error     = test_result.failure_details

    w = console.width
    console.print(f"\n[bold bright_white]╭─ DebuggerAgent ({MODEL})[/bold bright_white]")
    console.print(f"[bold bright_white]│[/bold bright_white]")
    console.print(f"[bold bright_white]│[/bold bright_white]  "
                  f"{escape(task_card.description[:80])}")
    console.print(f"[bold bright_white]╰{'─' * (w-2)}[/bold bright_white]\n")

    while attempts < 3:
        attempts += 1

        # ── Classify before attempting ─────────────────────────────────────
        failure_type = classify_failure(last_error, attempts)
        policy       = ESCALATION_POLICY[failure_type]

        if policy["escalate"] and attempts > policy["max_local_retries"]:
            log(f"[yellow]Escalating: {failure_type.value} after {attempts} attempts[/yellow]")

            # Read any files we've touched to build the failure report
            suggested_scope = list(set(
                task_card.context.relevant_files + files_modified
            ))

            report = FailureReport(
                task_id         = task_id,
                failure_type    = failure_type,
                attempts        = attempts,
                last_error      = last_error[:500],
                reflection      = f"After {attempts} attempts: {failure_type.value}. "
                                  f"Files examined: {suggested_scope}",
                files_seen      = files_modified,
                suggested_scope = suggested_scope,
                should_escalate = True,
            )
            return "", [], report

        # ── Read current file state ────────────────────────────────────────
        file_content = ""
        if task_card.context.relevant_files:
            read = call_mcp(
                FILESYSTEM_MCP_URL, "read_file",
                {"path": task_card.context.relevant_files[0]}
            )
            file_content = read.get("content", "")

        # ── Structured reflection BEFORE fixing ────────────────────────────
        console.print(f"  [dim]Reflecting on failure (attempt {attempts})...[/dim]",
                      end="\r")
        reflection = generate_reflection(
            task_description = task_card.description,
            previous_fix     = previous_fix,
            test_failure     = last_error[:1000],
            file_content     = file_content,
        )

        console.print(f"  [bold yellow]● Reflection[/bold yellow]")
        console.print(f"  [dim]{escape(reflection[:300])}[/dim]\n")

        # Store reflection in episodic memory regardless of outcome
        if task_card.context.error_trace:
            sig = normalize_signature(
                task_card.context.error_trace,
                file_name=task_card.context.relevant_files[0]
                          if task_card.context.relevant_files else ""
            )
            store_episode(
                signature   = sig,
                context     = task_card.description[:100],
                reflection  = reflection[:300],
                fix_applied = "in_progress",
                outcome     = "attempt"
            )

        # ── ReAct loop with reflection injected into prompt ────────────────
        messages = [
            {
                "role": "system",
                "content": (
                    "You are DebuggerAgent — you fix bugs that CoderAgent could not fix.\n"
                    "Tools: " + ", ".join(TOOL_REGISTRY.keys()) + "\n\n"
                    "Rules:\n"
                    "1. Read the file first — do not assume its current state\n"
                    "2. Apply the fix described in the reflection\n"
                    "3. Run the file to verify before finishing\n"
                    "4. Be surgical — change only what the reflection identifies"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task_card.description}\n\n"
                    f"Files to fix: {task_card.context.relevant_files}\n\n"
                    f"Test failure:\n{last_error[:800]}\n\n"
                    f"Reflection (what was wrong and how to fix it):\n{reflection}\n\n"
                    f"Apply the fix described in the reflection. "
                    f"Then run the code to verify."
                )
            }
        ]

        # Inner ReAct loop — same pattern as CoderAgent
        fix_summary = ""
        for step in range(1, MAX_STEPS + 1):

            if should_summarize(messages):
                messages = summarize_messages(messages)

            snapshot = messages.copy()
            MAX_TOOL_RETRIES = 2

            for attempt_inner in range(MAX_TOOL_RETRIES):
                try:
                    with console.status(
                        f"[dim]Step {step}...[/dim]", spinner="dots"
                    ):
                        response = client.chat.completions.create(
                            model       = MODEL,
                            messages    = messages,
                            tools       = groq_tools,
                            tool_choice = "auto",
                            temperature = TEMPERATURE_AGENT,
                        )
                    break
                except Exception as e:
                    if "tool_use_failed" in str(e) and attempt_inner < 1:
                        messages = snapshot.copy()
                        messages.append({
                            "role":    "user",
                            "content": "Use JSON tool calls only. No XML tags."
                        })
                        continue
                    raise

            message     = response.choices[0].message
            stop_reason = response.choices[0].finish_reason

            if stop_reason == "tool_calls" and message.tool_calls:
                if message.content:
                    console.print(f"  [dim]{escape(message.content[:200])}[/dim]")

                messages.append({
                    "role":       "assistant",
                    "content":    message.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name,
                                      "arguments": tc.function.arguments}}
                        for tc in message.tool_calls
                    ]
                })

                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    # Print compact tool line
                    verb  = {"read_file": "Read", "write_file": "Write",
                             "execute_command": "Run", "run_python": "Run",
                             "list_dir": "List"}.get(tool_name, tool_name)
                    label = args.get("path") or args.get("command", "")[:50]
                    console.print(
                        f"  [bold cyan]❯[/bold cyan] {verb} "
                        f"[dim]{escape(label)}[/dim]"
                    )

                    result_str = execute_tool_call(tool_name, args)

                    if tool_name == "write_file" and "path" in args:
                        if args["path"] not in files_modified:
                            files_modified.append(args["path"])

                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": result_str
                    })

            elif stop_reason == "stop":
                fix_summary = message.content or ""
                if fix_summary:
                    console.print(f"\n  [green]✓[/green] {escape(fix_summary[:200])}")
                break

        # ── Run tests again after this fix attempt ─────────────────────────
        if files_modified:
            test_result = run_tests(
                task_id,
                files_modified[-1],
                task_card.description
            )
            last_error    = test_result.failure_details
            previous_fix  = fix_summary

            if test_result.passed():
                # Update episodic memory with successful fix
                if task_card.context.error_trace:
                    store_episode(
                        signature   = sig,
                        context     = task_card.description[:100],
                        reflection  = reflection[:300],
                        fix_applied = fix_summary[:300],
                        outcome     = "success"
                    )
                artifacts = [
                    TaskArtifact(type="file_patch", file=f,
                                 description=f"Fixed by DebuggerAgent")
                    for f in files_modified
                ]
                return fix_summary, artifacts, None

    # Exhausted all attempts
    report = FailureReport(
        task_id         = task_id,
        failure_type    = FailureType.WRONG_APPROACH,
        attempts        = attempts,
        last_error      = last_error[:500],
        reflection      = "Exhausted all debug attempts",
        files_seen      = files_modified,
        suggested_scope = task_card.context.relevant_files,
        should_escalate = True,
    )
    return "", [], report


# ── Background runner ──────────────────────────────────────────────────────

def run_debug_background(
    task_card:    TaskCard,
    test_failure: str,
    previous_fix: str
):
    """Runs in background thread — same pattern as CoderAgent."""
    from test_agent import TestResult as TR
    dummy_result = TR(
        verdict="fail", tests_written=0, tests_passed=0,
        tests_failed=1, failure_details=test_failure, test_file=""
    )
    try:
        answer, artifacts, failure_report = run_debug_loop(
            task_card, dummy_result, previous_fix
        )
        update = {
            "status":    "completed" if not failure_report else "escalate",
            "artifacts": [a.__dict__ for a in artifacts],
            "result":    answer,
        }
        if failure_report:
            update["failure_report"] = failure_report.to_dict()
        task_store[task_card.task_id].update(update)

    except Exception as e:
        log(f"[red]DebuggerAgent task failed: {e}[/red]")
        task_store[task_card.task_id].update({
            "status": "failed", "error": str(e)
        })


# ── A2A Endpoints ──────────────────────────────────────────────────────────

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name":        "DebuggerAgent",
        "version":     "1.0",
        "description": "Fixes bugs CoderAgent could not fix via structured reflection",
        "endpoint":    f"http://localhost:{DEBUGGER_PORT}",
        "capabilities": ["reflection", "root_cause_analysis", "escalation"],
    }


@app.post("/tasks/send")
async def receive_task(request: Request):
    try:
        data = await request.json()
        task_card    = TaskCard.from_dict(data["task_card"])
        test_failure = data.get("test_failure", "")
        previous_fix = data.get("previous_fix", "")
    except Exception as e:
        return JSONResponse({"error": f"Invalid request: {e}"}, status_code=400)

    task_store[task_card.task_id] = {
        **task_card.to_dict(),
        "status":      "in_progress",
        "received_at": datetime.utcnow().isoformat(),
    }

    thread = threading.Thread(
        target = run_debug_background,
        args   = (task_card, test_failure, previous_fix),
        daemon = True
    )
    thread.start()

    log(f"[cyan]DebuggerAgent accepted task {task_card.task_id}[/cyan]")
    return JSONResponse(
        {"task_id": task_card.task_id, "status": "accepted"},
        status_code=202
    )


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_store:
        return JSONResponse({"error": f"Not found: {task_id}"}, status_code=404)
    return JSONResponse(task_store[task_id])


if __name__ == "__main__":
    import uvicorn
    init_db()
    log(f"[cyan]DebuggerAgent starting on port {DEBUGGER_PORT}[/cyan]")
    uvicorn.run(app, host="0.0.0.0", port=DEBUGGER_PORT, log_level="warning")