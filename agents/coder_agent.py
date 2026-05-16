import os
import sys
import json
import time
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.text import Text
from rich.syntax import Syntax
from rich.markup import escape
from rich.padding import Padding
from mcp_client import call_mcp, list_tools

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_HERE, "../.env"))

# ── Configuration ──────────────────────────────────────────────────────────
GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
FILESYSTEM_MCP_URL = os.environ.get("FILESYSTEM_MCP_URL", "http://localhost:8000")
SHELL_MCP_URL      = os.environ.get("SHELL_MCP_URL",      "http://localhost:8001")

MODEL      = "llama-3.3-70b-versatile"
MAX_STEPS  = 20

client  = Groq(api_key=GROQ_API_KEY)
console = Console(highlight=False)

# ── Tool Registry ──────────────────────────────────────────────────────────

TOOL_REGISTRY = {}

def discover_tools() -> list:
    """
    Ask both MCP servers what tools they expose.
    Build TOOL_REGISTRY mapping tool_name → server_url.
    Return combined list of tool schemas for the LLM.
    """
    all_tools = []
    for server_url in [FILESYSTEM_MCP_URL, SHELL_MCP_URL]:
        tools = list_tools(server_url)
        for tool in tools:
            TOOL_REGISTRY[tool["name"]] = server_url
            all_tools.append(tool)
    return all_tools


def convert_to_groq_tools(mcp_tools: list) -> list:
    """
    MCP tool schemas and Groq tool schemas are almost identical
    but Groq wraps them in a specific structure.

    MCP format:
    {
        "name": "read_file",
        "description": "...",
        "inputSchema": { "type": "object", "properties": {...} }
    }

    Groq format:
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "...",
            "parameters": { "type": "object", "properties": {...} }
        }
    }

    Only two differences:
    1. Wrapped in {"type": "function", "function": {...}}
    2. "inputSchema" renamed to "parameters"
    """
    groq_tools = []
    for tool in mcp_tools:
        groq_tools.append({
            "type": "function",
            "function": {
                "name":        tool["name"],
                "description": tool["description"],
                "parameters":  tool["inputSchema"]
            }
        })
    return groq_tools


# ── Tool Execution ─────────────────────────────────────────────────────────

def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """
    Look up which MCP server owns this tool.
    Call it. Return result as a string for the messages list.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"})

    server_url = TOOL_REGISTRY[tool_name]
    result     = call_mcp(server_url, tool_name, arguments)
    return json.dumps(result, indent=2)


# ── Claude Code Style Terminal Output ──────────────────────────────────────
# Minimal, clean, flowing — no heavy panels. Uses ❯ markers,
# dim metadata, inline tool results, and muted colors.

TOOL_VERBS = {
    "read_file":       ("Read", "cyan"),
    "write_file":      ("Write", "magenta"),
    "list_dir":        ("List", "cyan"),
    "execute_command": ("Run", "yellow"),
    "run_python":      ("Run", "yellow"),
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
        code = args.get("code", "")
        first_line = code.split("\n")[0]
        if len(first_line) > 50:
            first_line = first_line[:47] + "..."
        return f"{verb} python"
    return f"{verb}"


def print_init(task: str, tool_names: list):
    """Print the startup banner — compact and clean."""
    w = console.width
    console.print()
    console.print(f"[bold bright_white]╭─ CoderAgent[/bold bright_white] [dim]({MODEL})[/dim]")
    console.print(f"[dim]│[/dim]")
    console.print(f"[dim]│[/dim]  {escape(task)}")
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
    """Print tool arguments inline — only for write operations or long args."""
    if tool_name == "write_file":
        content = args.get("content", "")
        if content:
            lines = content.splitlines()
            # Show as a code block, truncated to 20 lines
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
        # Not JSON — just show raw
        short = result_str[:300] if len(result_str) > 300 else result_str
        console.print(f"    [dim]{escape(short)}[/dim]")
        return

    # Determine success/failure
    is_error    = "error" in parsed
    exit_code   = parsed.get("exit_code")
    is_fail     = is_error or (exit_code is not None and exit_code != 0)

    if is_error:
        console.print(f"    [red]✗ {escape(str(parsed['error'])[:200])}[/red]")
        return

    # For command results — show stdout/stderr compactly
    if "stdout" in parsed or "stderr" in parsed:
        stdout = parsed.get("stdout", "").strip()
        stderr = parsed.get("stderr", "").strip()
        code   = parsed.get("exit_code", 0)

        if code == 0:
            marker = "[green]✓[/green]"
        else:
            marker = "[red]✗[/red]"

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

    # For file read results — show content compactly
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

    # For write results
    if parsed.get("success"):
        written = parsed.get("bytes_written", "?")
        console.print(f"    [green]✓[/green] [dim]{written} bytes written[/dim]")
        return

    # For list_dir results
    if "entries" in parsed:
        entries = parsed["entries"]
        console.print(f"    [green]✓[/green] [dim]{len(entries)} items[/dim]")
        for entry in entries[:10]:
            icon = "📁" if entry["type"] == "directory" else "📄"
            console.print(f"    [dim]{icon} {entry['name']}[/dim]")
        if len(entries) > 10:
            console.print(f"    [dim]... {len(entries) - 10} more[/dim]")
        return

    # Fallback — dump a short JSON preview
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


def build_system_prompt(mcp_tools: list) -> str:
    """
    Build system prompt dynamically.
    Tool descriptions come from MCP — never hardcoded here.
    Prompt only contains behavioral rules the tools can't express.
    """
    tool_lines = []
    for tool in mcp_tools:
        # Use the description from the MCP server schema directly
        tool_lines.append(f"- {tool['name']}: {tool['description']}")

    tools_section = "\n".join(tool_lines)

    return f"""You are CoderAgent — a precise software engineer working inside a sandbox.

Tools available:
{tools_section}

Rules:
1. Always read a file before editing it — never assume its current contents.
2. After writing a fix, always run the code to verify it works.
3. To run code, use execute_command with the appropriate command for the language:
   - Python:     python filename.py
   - Node.js:    node filename.js
   - Go:         go run filename.go
   - Shell:      bash filename.sh
   - Other:      use whatever command the project's language requires
4. Fix ALL bugs you find, not just the first one.
5. After write_file, read_file again to confirm the content is exactly what you intended.
6. When done, summarize every change you made and why."""

# ── The ReAct Loop ─────────────────────────────────────────────────────────

def run_coder_agent(task: str):
    t0 = time.time()

    # Discover tools silently, then show init banner
    mcp_tools  = discover_tools()
    groq_tools = convert_to_groq_tools(mcp_tools)
    tool_names = [t["function"]["name"] for t in groq_tools]

    print_init(task, tool_names)

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(mcp_tools)  # ← dynamic now
        },
        {
            "role": "user",
            "content": task
        }
    ]

    for step in range(1, MAX_STEPS + 1):

        # Spinner while waiting for LLM
        with console.status(
            f"[dim]Step {step} · thinking...[/dim]",
            spinner="dots",
            spinner_style="cyan"
        ):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=groq_tools,
                tool_choice="auto",
                temperature=0.2
            )

        message     = response.choices[0].message
        stop_reason = response.choices[0].finish_reason

        # ── Tool calls ────────────────────────────────────────────────
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
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                print_tool_use(tool_name, arguments)
                print_tool_args(arguments, tool_name)

                with console.status(f"[dim]  running...[/dim]", spinner="dots", spinner_style="dim"):
                    result = execute_tool_call(tool_name, arguments)

                print_result(result)
                console.print()

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result
                })

        # ── Done ──────────────────────────────────────────────────────
        elif stop_reason == "stop":
            elapsed = time.time() - t0
            print_finished(
                message.content or "Task complete.",
                step,
                len(messages),
                elapsed
            )
            return message.content

        # ── Unexpected ────────────────────────────────────────────────
        else:
            print_error(f"Unexpected stop_reason: '{stop_reason}'")
            break

    print_error(f"Agent hit {MAX_STEPS} step limit without finishing.")
    return "Agent did not complete within step limit."


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task = (
        "The file 'buggy_code.py' in the sandbox has bugs. "
        "Read the file, identify all bugs, fix them, "
        "then run the fixed file to verify it works correctly."
    )
    run_coder_agent(task)