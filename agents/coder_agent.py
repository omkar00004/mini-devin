# agents/coder_agent.py

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
import urllib.request as _urllib

# Add parent dir to path so memory/ and models/ imports work
sys.path.insert(0, os.path.dirname(__file__))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL,
    MAX_STEPS,
    TEMPERATURE_AGENT,
    SUMMARIZE_THRESHOLD,
    KEEP_RECENT,
    MAX_TOOL_RETRIES,
)

from mcp_client import call_mcp, list_tools
from models.task_card import TaskCard, TaskArtifact
from memory.episodic import init_db, normalize_signature, recall_as_context, store_episode
from memory.summarizer import summarize_messages, should_summarize

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

# ── Config ─────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
FILESYSTEM_MCP_URL = os.environ.get("FILESYSTEM_MCP_URL", "http://localhost:8000")
SHELL_MCP_URL      = os.environ.get("SHELL_MCP_URL",      "http://localhost:8001")
DEBUGGER_AGENT_URL = os.environ.get("DEBUGGER_AGENT_URL", "http://localhost:9002")
CODER_PORT         = int(os.environ.get("CODER_AGENT_PORT", 9001))

client  = Groq(api_key=GROQ_API_KEY)
console = Console(highlight=False)
app     = FastAPI()

# ── In-memory task store ───────────────────────────────────────────────────
# Maps task_id → task dict with status and artifacts
# Temporary — lives only while the process runs
task_store: dict[str, dict] = {}

# ── Tool Registry ──────────────────────────────────────────────────────────
TOOL_REGISTRY: dict[str, str] = {}


def discover_tools() -> list:
    all_tools = []
    for server_url in [FILESYSTEM_MCP_URL, SHELL_MCP_URL]:
        tools = list_tools(server_url)
        for tool in tools:
            TOOL_REGISTRY[tool["name"]] = server_url
            all_tools.append(tool)
    return all_tools


def convert_to_groq_tools(mcp_tools: list) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["inputSchema"],
            }
        }
        for t in mcp_tools
    ]


def build_system_prompt(mcp_tools: list) -> str:
    """Dynamic — reflects actual available tools at runtime."""
    tool_lines = [f"- {t['name']}: {t['description']}" for t in mcp_tools]
    tools_section = "\n".join(tool_lines)
    return f"""You are CoderAgent — a precise software engineer working inside a sandbox.

Tools available:
{tools_section}

Rules:
1. Always read a file before editing it — never assume its current contents.
2. After writing a fix, always run the code to verify it works.
3. To run code use execute_command with the right command for the language:
   Python → python filename.py  |  Node → node filename.js  |  Go → go run filename.go
4. Fix ALL bugs you find, not just the first one.
5. After write_file, read_file again to confirm content is exactly what you intended.
6. When done, summarize every change you made and why."""


def execute_tool_call(tool_name: str, arguments: dict) -> str:
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: '{tool_name}'"})
    result = call_mcp(TOOL_REGISTRY[tool_name], tool_name, arguments)
    return json.dumps(result, indent=2)


# ── Claude Code Style Terminal Output ──────────────────────────────────────
# Minimal, clean, flowing — no heavy panels. Uses ❯ markers,
# dim metadata, inline tool results, and muted colors.

TOOL_VERBS = {
    "read_file":       ("Read",  "cyan"),
    "write_file":      ("Write", "magenta"),
    "list_dir":        ("List",  "cyan"),
    "execute_command": ("Run",   "yellow"),
    "run_python":      ("Run",   "yellow"),
}


def _tool_label(name: str, args: dict) -> str:
    """Build a human-readable one-liner like 'Read buggy_code.py'."""
    verb, _ = TOOL_VERBS.get(name, (name, "white"))
    if name in ("read_file", "write_file"):
        return f"{verb} {args.get('path', '?')}"
    elif name == "list_dir":
        return f"{verb} {args.get('path', '.')}"
    elif name == "execute_command":
        cmd = args.get("command", "")
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"{verb} `{cmd}`"
    elif name == "run_python":
        return f"{verb} python"
    return f"{verb}"


def log(msg: str):
    """Thread-safe console print with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    console.print(f"[dim]{ts}[/dim] {msg}")


def print_init(task: str, tool_names: list):
    """Print the startup banner — compact and clean."""
    w = console.width
    console.print()
    console.print(f"[bold bright_white]╭─ CoderAgent[/bold bright_white] [dim]({MODEL})[/dim]")
    console.print(f"[dim]│[/dim]")
    console.print(f"[dim]│[/dim]  {escape(task[:80])}")
    console.print(f"[dim]│[/dim]")
    console.print(f"[dim]│[/dim]  [dim italic]{len(tool_names)} tools: {', '.join(tool_names)}[/dim italic]")
    console.print(f"[dim]╰{'─' * (w - 2)}[/dim]")
    console.print()


def print_thinking(text: str):
    """Print the agent's reasoning — dim and indented, like inner monologue."""
    if not text or not text.strip():
        return
    lines = text.strip().splitlines()
    for line in lines:
        console.print(f"  [dim]{escape(line)}[/dim]")
    console.print()


def print_tool_use(name: str, args: dict):
    """Print a tool call — one clean line with a colored marker."""
    _, color = TOOL_VERBS.get(name, (name, "white"))
    label    = _tool_label(name, args)
    console.print(f"  [bold {color}]❯ {escape(label)}[/bold {color}]")


def print_tool_args(args: dict, tool_name: str):
    """Print tool arguments inline — only for write operations."""
    if tool_name == "write_file":
        content = args.get("content", "")
        if content:
            lines = content.splitlines()
            show = lines[:20]
            console.print()
            for line in show:
                console.print(f"    [green]+[/green] [dim]{escape(line)}[/dim]")
            if len(lines) > 20:
                console.print(f"    [dim]... {len(lines) - 20} more lines[/dim]")
            console.print()
    elif tool_name == "run_python":
        code = args.get("code", "")
        if code:
            lines = code.splitlines()
            show = lines[:15]
            console.print()
            for line in show:
                console.print(f"    [dim]{escape(line)}[/dim]")
            if len(lines) > 15:
                console.print(f"    [dim]... {len(lines) - 15} more lines[/dim]")
            console.print()


def print_result(result_str: str):
    """Print tool result — compact, color-coded by success/failure."""
    try:
        parsed = json.loads(result_str)
    except Exception:
        short = result_str[:300] if len(result_str) > 300 else result_str
        console.print(f"    [dim]{escape(short)}[/dim]")
        return

    if "error" in parsed:
        console.print(f"    [red]✗ {escape(str(parsed['error'])[:200])}[/red]")
        return

    # Command results
    if "stdout" in parsed or "stderr" in parsed:
        stdout = parsed.get("stdout", "").strip()
        stderr = parsed.get("stderr", "").strip()
        code   = parsed.get("exit_code", 0)
        marker = "[green]✓[/green]" if code == 0 else "[red]✗[/red]"
        output = stdout or stderr
        if output:
            lines = output.splitlines()
            show  = lines[:12]
            console.print(f"    {marker} [dim]exit {code}[/dim]")
            for line in show:
                console.print(f"    [dim]{escape(line)}[/dim]")
            if len(lines) > 12:
                console.print(f"    [dim]... {len(lines) - 12} more lines[/dim]")
        else:
            console.print(f"    {marker} [dim]exit {code}[/dim]")
        return

    # File read results
    if "content" in parsed:
        content = parsed["content"]
        lines = content.splitlines()
        size  = parsed.get("size_bytes", "?")
        n     = parsed.get("lines", len(lines))
        console.print(f"    [green]✓[/green] [dim]{n} lines · {size} bytes[/dim]")
        show = lines[:15]
        for line in show:
            console.print(f"    [dim]{escape(line)}[/dim]")
        if len(lines) > 15:
            console.print(f"    [dim]... {len(lines) - 15} more lines[/dim]")
        return

    # Write results
    if parsed.get("success"):
        written = parsed.get("bytes_written", "?")
        console.print(f"    [green]✓[/green] [dim]{written} bytes written[/dim]")
        return

    # list_dir results
    if "entries" in parsed:
        entries = parsed["entries"]
        console.print(f"    [green]✓[/green] [dim]{len(entries)} items[/dim]")
        for entry in entries[:10]:
            icon = "📁" if entry["type"] == "directory" else "📄"
            console.print(f"    [dim]{icon} {entry['name']}[/dim]")
        if len(entries) > 10:
            console.print(f"    [dim]... {len(entries) - 10} more[/dim]")
        return

    # Fallback
    short = json.dumps(parsed, indent=2)
    if len(short) > 300:
        short = short[:300] + "\n..."
    console.print(f"    [dim]{escape(short)}[/dim]")


def print_finished(answer: str, steps: int, messages_count: int, elapsed: float):
    """Print final answer — clean block, then stats."""
    console.print()
    console.print(f"  [bold green]✓[/bold green] [bold bright_white]Task complete[/bold bright_white]")
    console.print()
    for line in answer.strip().splitlines():
        console.print(f"  {escape(line)}")
    console.print()
    console.print(
        f"  [dim]─ {steps} steps · {messages_count} messages · {elapsed:.1f}s[/dim]"
    )
    console.print()


def print_error(msg: str):
    console.print(f"\n  [bold red]✗[/bold red] [red]{escape(msg)}[/red]\n")


# ── Core ReAct Loop ────────────────────────────────────────────────────────

def run_react_loop(task_card: TaskCard) -> tuple[str, list]:
    """
    Runs the ReAct loop for one TaskCard.
    Returns (final_answer, artifacts_list).
    Updates task_store[task_id] status throughout.
    """
    task_id = task_card.task_id
    t0      = time.time()

    # ── Discover tools ─────────────────────────────────────────────────────
    mcp_tools  = discover_tools()
    groq_tools = convert_to_groq_tools(mcp_tools)
    tool_names = [t["function"]["name"] for t in groq_tools]

    print_init(task_card.description, tool_names)

    # ── Query episodic memory ──────────────────────────────────────────────
    # If the task has an error trace, check if we've seen this before
    memory_context = ""
    if task_card.context.error_trace:
        # Extract function/file from relevant_files if available
        file_name = task_card.context.relevant_files[0] if task_card.context.relevant_files else ""
        sig = normalize_signature(
            error_trace   = task_card.context.error_trace,
            function_name = "",      # PlannerAgent doesn't know function name yet
            file_name     = file_name
        )
        memory_context = recall_as_context(sig)
        if memory_context:
            log(f"[yellow]📚 Episodic memory found for signature: {sig}[/yellow]")

    # ── Build task prompt ──────────────────────────────────────────────────
    # Combine task card fields into a clear instruction for the agent
    criteria_text = ""
    if task_card.acceptance_criteria:
        criteria_lines = "\n".join(f"  - {c}" for c in task_card.acceptance_criteria)
        criteria_text  = f"\n\nAcceptance criteria:\n{criteria_lines}"

    files_text = ""
    if task_card.context.relevant_files:
        files_text = f"\n\nRelevant files: {', '.join(task_card.context.relevant_files)}"

    error_text = ""
    if task_card.context.error_trace:
        error_text = f"\n\nKnown error trace:\n{task_card.context.error_trace}"

    task_prompt = (
        f"Task {task_id}: {task_card.description}"
        f"{files_text}"
        f"{error_text}"
        f"{criteria_text}"
    )

    # Inject episodic memory if found
    if memory_context:
        task_prompt += f"\n\n{memory_context}"

    # ── Initialize messages ────────────────────────────────────────────────
    messages = [
        {"role": "system",  "content": build_system_prompt(mcp_tools)},
        {"role": "user",    "content": task_prompt},
    ]

    files_modified  = []
    errors_seen     = []
    final_answer    = ""

    # ── ReAct Loop ─────────────────────────────────────────────────────────
    for step in range(1, MAX_STEPS + 1):

        # Summarize if context is growing too large
        if should_summarize(messages):
            log(f"[dim]Context at {len(messages)} messages — summarizing...[/dim]")
            messages = summarize_messages(messages)
            log(f"[dim]Compressed to {len(messages)} messages[/dim]")

        with console.status(f"[dim]Step {step} — thinking...[/dim]", spinner="dots"):

            # Snapshot messages before LLM call.
            # On a malformed tool call, Groq rejects the request entirely
            # (no assistant message is added to history), so we reset to
            # this snapshot to keep the conversation clean before retrying.
            messages_snapshot = list(messages)

            for attempt in range(MAX_TOOL_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model       = MODEL,
                        messages    = messages,
                        tools       = groq_tools,
                        tool_choice = "auto",
                        temperature = TEMPERATURE_AGENT,
                    )
                    break   # success — exit retry loop
                except Exception as e:
                    err_str = str(e)
                    if "tool_use_failed" in err_str or "tool call validation" in err_str:
                        if attempt < MAX_TOOL_RETRIES - 1:
                            log(f"[yellow]Tool call malformed — retrying (attempt {attempt+2})[/yellow]")
                            # Reset to snapshot so the malformed attempt is NOT in history,
                            # then add a strict corrective nudge before retrying.
                            messages = list(messages_snapshot)
                            messages.append({
                                "role":    "user",
                                "content": (
                                    "IMPORTANT: You must respond by calling one of the provided "
                                    "tools using the correct JSON format. "
                                    "Do NOT output XML tags like <function=...>. "
                                    "Do NOT output plain text. "
                                    "Call a tool now."
                                )
                            })
                            continue
                    # Non-retryable error — re-raise
                    raise


        message     = response.choices[0].message
        stop_reason = response.choices[0].finish_reason

        # ── Agent calls tools ──────────────────────────────────────────────
        if stop_reason == "tool_calls" and message.tool_calls:

            if message.content:
                print_thinking(message.content)

            messages.append({
                "role":       "assistant",
                "content":    message.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                print_tool_use(tool_name, arguments)
                print_tool_args(arguments, tool_name)

                with console.status(f"[dim]  running...[/dim]", spinner="dots", spinner_style="dim"):
                    result_str = execute_tool_call(tool_name, arguments)

                print_result(result_str)
                console.print()

                # Track files modified for artifacts
                if tool_name == "write_file" and "path" in arguments:
                    file_path = arguments["path"]
                    if file_path not in files_modified:
                        files_modified.append(file_path)

                # Track errors seen for episodic memory
                try:
                    result_dict = json.loads(result_str)
                    stderr = result_dict.get("stderr", "")
                    if stderr and result_dict.get("exit_code", 0) != 0:
                        errors_seen.append(stderr[:200])
                except Exception:
                    pass

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })

        # ── Agent is done ──────────────────────────────────────────────────
        elif stop_reason == "stop":
            elapsed      = time.time() - t0
            final_answer = message.content or "Task complete."

            print_finished(
                final_answer,
                step,
                len(messages),
                elapsed
            )

            log(f"[green]Task {task_id} completed in {step} steps[/green]")
            break

        else:
            print_error(f"Unexpected stop_reason: '{stop_reason}'")
            break

    # ── Build artifacts ────────────────────────────────────────────────────
    artifacts = [
        TaskArtifact(
            type        = "file_patch",
            file        = f,
            description = f"Modified by CoderAgent during task {task_id}",
        )
        for f in files_modified
    ]

    # ── Store episodic memory ──────────────────────────────────────────────
    if task_card.context.error_trace and files_modified:
        file_name = task_card.context.relevant_files[0] if task_card.context.relevant_files else ""
        sig = normalize_signature(
            error_trace = task_card.context.error_trace,
            file_name   = file_name
        )
        outcome = "success" if files_modified else "failure"
        store_episode(
            signature   = sig,
            context     = f"Task: {task_card.description[:100]}",
            reflection  = f"Fixed by modifying: {', '.join(files_modified)}",
            fix_applied = final_answer[:300],
            outcome     = outcome
        )
        log(f"[dim]Stored episodic memory: {sig} → {outcome}[/dim]")

    return final_answer, artifacts

# ── Escalate to Debugger ─────────────────────────────────────────────────

def escalate_to_debugger(
    task_card:    TaskCard,
    test_failure: str,
    previous_fix: str
) -> dict:
    """
    Send failed task to DebuggerAgent via A2A.
    Poll until debugger returns completed or escalate.
    """
    from config import POLL_INTERVAL_SEC, POLL_TIMEOUT_SEC

    payload = json.dumps({
        "task_card":    task_card.to_dict(),
        "test_failure": test_failure,
        "previous_fix": previous_fix,
    }).encode("utf-8")

    req = _urllib.Request(
        f"{DEBUGGER_AGENT_URL}/tasks/send",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with _urllib.urlopen(req, timeout=10) as resp:
            if resp.status != 202:
                return {"status": "failed", "error": "DebuggerAgent rejected task"}
    except Exception as e:
        return {"status": "failed", "error": f"DebuggerAgent unreachable: {e}"}

    # Poll for result
    deadline = time.time() + POLL_TIMEOUT_SEC
    while time.time() < deadline:
        try:
            with _urllib.urlopen(
                _urllib.Request(
                    f"{DEBUGGER_AGENT_URL}/tasks/{task_card.task_id}"
                ), timeout=10
            ) as resp:
                data   = json.loads(resp.read())
                status = data.get("status")
            if status in ("completed", "escalate", "failed"):
                return data
            time.sleep(POLL_INTERVAL_SEC)
        except Exception:
            time.sleep(POLL_INTERVAL_SEC)

    return {"status": "timeout", "task_id": task_card.task_id}  

# ── Background task runner ─────────────────────────────────────────────────

def run_task_background(task_card: TaskCard):
    from test_agent import run_tests

    try:
        final_answer, artifacts = run_react_loop(task_card)

        # Run tests on every modified file
        files_modified = [a.file for a in artifacts if a.file]
        test_passed    = True
        test_failure   = ""

        for file_path in files_modified:
            result = run_tests(task_card.task_id, file_path, task_card.description)
            if not result.passed():
                test_passed   = False
                test_failure  = result.failure_details
                break

        if test_passed:
            # Clean pass — done
            task_store[task_card.task_id].update({
                "status":    "completed",
                "artifacts": [a.__dict__ for a in artifacts],
                "result":    final_answer,
            })

        else:
            # Tests failed — escalate to DebuggerAgent
            log(f"[yellow]Tests failed for {task_card.task_id} "
                f"— escalating to DebuggerAgent[/yellow]")

            debug_result = escalate_to_debugger(
                task_card    = task_card,
                test_failure = test_failure,
                previous_fix = final_answer,
            )

            debug_status = debug_result.get("status")

            if debug_status == "completed":
                task_store[task_card.task_id].update({
                    "status":    "completed",
                    "artifacts": debug_result.get("artifacts", []),
                    "result":    debug_result.get("result", ""),
                    "debugged":  True,
                })
            elif debug_status == "escalate":
                # DebuggerAgent also failed — needs PlannerAgent replanning
                task_store[task_card.task_id].update({
                    "status":         "escalate",
                    "failure_report": debug_result.get("failure_report", {}),
                })
            else:
                task_store[task_card.task_id].update({
                    "status": "failed",
                    "error":  debug_result.get("error", "unknown"),
                })

    except Exception as e:
        log(f"[red]Task {task_card.task_id} crashed: {e}[/red]")
        task_store[task_card.task_id].update({
            "status": "failed", "error": str(e)
        })
        
# ── A2A Endpoints ──────────────────────────────────────────────────────────

@app.get("/.well-known/agent.json")
async def agent_card():
    """
    Agent Card — any other agent can discover what this agent does.
    Equivalent of MCP's tools/list but for agents.
    """
    return {
        "name":         "CoderAgent",
        "version":      "1.0",
        "description":  "Fixes bugs in code files inside the sandbox",
        "endpoint":     f"http://localhost:{CODER_PORT}",
        "capabilities": ["code_editing", "bug_fixing", "code_execution"],
        "accepts":      "TaskCard",
        "returns":      "TaskCard with artifacts",
    }


@app.post("/tasks/send")
async def receive_task(request: Request):
    """
    PlannerAgent sends a Task Card here.
    Immediately returns 202 Accepted — does NOT wait for the task to complete.
    Task runs in a background thread.
    PlannerAgent polls GET /tasks/{id} for completion.
    """
    try:
        data      = await request.json()
        task_card = TaskCard.from_dict(data)
    except Exception as e:
        return JSONResponse({"error": f"Invalid Task Card: {e}"}, status_code=400)

    # Store with in_progress status before starting thread
    task_store[task_card.task_id] = {
        **task_card.to_dict(),
        "status":     "in_progress",
        "received_at": datetime.utcnow().isoformat(),
    }

    # Background thread — PlannerAgent is NOT blocked
    thread = threading.Thread(
        target  = run_task_background,
        args    = (task_card,),
        daemon  = True      # thread dies if main process dies
    )
    thread.start()

    log(f"[cyan]Accepted task {task_card.task_id}[/cyan] — running in background")

    # 202 Accepted: "I have it, I'm working on it, check back later"
    return JSONResponse(
        {"task_id": task_card.task_id, "status": "accepted"},
        status_code=202
    )


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """PlannerAgent polls this until status is 'completed' or 'failed'."""
    if task_id not in task_store:
        return JSONResponse({"error": f"Task not found: {task_id}"}, status_code=404)
    return JSONResponse(task_store[task_id])


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    init_db()
    log(f"[cyan]CoderAgent starting on port {CODER_PORT}[/cyan]")
    uvicorn.run(app, host="0.0.0.0", port=CODER_PORT, log_level="warning")