# mcp_servers/shell/server.py

import os
import time
import subprocess
import tempfile
import sys
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ── Configuration ──────────────────────────────────────────────────────────
SANDBOX_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../../sandbox")
)

DEFAULT_TIMEOUT_MS  = 10_000   # 10 seconds default
MAX_TIMEOUT_MS      = 30_000   # 30 seconds hard ceiling
MAX_OUTPUT_CHARS    = 8_000    # truncate stdout/stderr beyond this

# Commands that get blocked immediately before any execution.
# This is a basic safety layer — not a replacement for Docker.
BLOCKED_COMMANDS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",           # formats a disk
    "dd if=/dev/zero",  # overwrites disk with zeros
    ":(){:|:&};:",    # fork bomb
    "sudo",
    "su ",
    "chmod 777 /",
    "wget",           # no network calls in sandbox
    "curl",
    "nc ",            # netcat
    "nmap",
]

# ── Tool Schemas ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "execute_command",
        "description": (
            "Run a shell command inside the sandbox directory. "
            "Use this to run files in any language: "
            "'python file.py', 'node file.js', 'go run file.go', etc. "
            "Each call is stateless — no state persists between calls. "
            "For multi-step operations, chain commands with && in a single call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute. Use && to chain multiple commands."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": f"Max execution time in milliseconds. Default: {DEFAULT_TIMEOUT_MS}. Max: {MAX_TIMEOUT_MS}."
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "run_python",
        "description": (
            "Execute a Python code string inside the sandbox. "
            "Code runs with the sandbox as the working directory. "
            "Print statements go to stdout. Exceptions go to stderr."
            "For running files in any language, use execute_command instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute."
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": f"Max execution time in milliseconds. Default: {DEFAULT_TIMEOUT_MS}."
                }
            },
            "required": ["code"]
        }
    }
]

# ── Safety Check ───────────────────────────────────────────────────────────

def check_command_safety(command: str) -> str | None:
    """
    Returns an error message if the command is blocked, None if it's safe.
    Checks against the blocklist (case-insensitive).
    """
    command_lower = command.lower().strip()
    for blocked in BLOCKED_COMMANDS:
        if blocked.lower() in command_lower:
            return f"Command blocked for safety: contains '{blocked}'"
    return None

# ── Output Truncation ──────────────────────────────────────────────────────

def truncate_output(text: str) -> dict:
    """
    If text exceeds MAX_OUTPUT_CHARS, keep only the last MAX_OUTPUT_CHARS.
    Returns a dict with the (possibly truncated) text and metadata.
    Errors appear at the END of output, so we keep the tail.
    """
    if len(text) <= MAX_OUTPUT_CHARS:
        return {
            "text": text,
            "truncated": False,
            "original_length": len(text)
        }

    return {
        "text": text[-MAX_OUTPUT_CHARS:],
        "truncated": True,
        "original_length": len(text),
        "note": f"Output truncated. Showing last {MAX_OUTPUT_CHARS} chars of {len(text)} total."
    }

# ── Tool Implementations ───────────────────────────────────────────────────

def tool_execute_command(arguments: dict) -> dict:
    command = arguments.get("command", "").strip()
    timeout_ms = int(arguments.get("timeout_ms", DEFAULT_TIMEOUT_MS))

    # Validate inputs
    if not command:
        return {"error": "Missing required argument: command"}

    # Cap timeout at the hard ceiling
    timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)
    timeout_sec = timeout_ms / 1000

    # Safety check before any execution
    safety_error = check_command_safety(command)
    if safety_error:
        return {"error": safety_error}

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            shell=True,            # run through /bin/sh so && | ; work
            capture_output=True,   # capture stdout and stderr separately
            text=True,             # decode bytes to string automatically
            timeout=timeout_sec,
            cwd=SANDBOX_ROOT       # working directory is the sandbox
        )

        runtime_ms = int((time.time() - start_time) * 1000)

        stdout_info = truncate_output(result.stdout)
        stderr_info = truncate_output(result.stderr)

        return {
            "stdout":      stdout_info["text"],
            "stderr":      stderr_info["text"],
            "exit_code":   result.returncode,
            "runtime_ms":  runtime_ms,
            "stdout_truncated": stdout_info["truncated"],
            "stderr_truncated": stderr_info["truncated"],
        }

    except subprocess.TimeoutExpired:
        runtime_ms = int((time.time() - start_time) * 1000)
        return {
            "stdout":     "",
            "stderr":     f"Process killed: exceeded timeout of {timeout_ms}ms",
            "exit_code":  -1,
            "runtime_ms": runtime_ms,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

    except Exception as e:
        return {"error": f"Execution failed: {str(e)}"}


def tool_run_python(arguments: dict) -> dict:
    code = arguments.get("code", "").strip()
    timeout_ms = int(arguments.get("timeout_ms", DEFAULT_TIMEOUT_MS))

    if not code:
        return {"error": "Missing required argument: code"}

    timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)

    temp_path = None  # ← initialize here so finally block can check it

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=SANDBOX_ROOT,
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        result = tool_execute_command({
            # sys.executable = the exact Python binary running this server
            # guarantees we use the same Python environment, not whatever
            # 'python' resolves to on the system PATH
            "command": f"{sys.executable} {temp_path}",
            "timeout_ms": timeout_ms
        })

        if result.get("stderr"):
            result["stderr"] = result["stderr"].replace(temp_path, "<python_script>")

        return result

    except Exception as e:
        return {"error": f"Failed to run python code: {str(e)}"}

    finally:
        # Runs no matter what — success, error, or crash
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

# ── Tool Dispatch ──────────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "execute_command": tool_execute_command,
    "run_python":      tool_run_python,
}

# ── JSON-RPC Helpers ───────────────────────────────────────────────────────

def make_success(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}

def make_error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

# ── Main Handler ───────────────────────────────────────────────────────────

@app.post("/")
async def handle_jsonrpc(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(make_error(None, -32700, "Parse error: invalid JSON"))

    request_id = body.get("id")
    method     = body.get("method")
    params     = body.get("params", {})

    if method == "tools/list":
        return JSONResponse(make_success(request_id, {"tools": TOOLS}))

    elif method == "tools/call":
        tool_name  = params.get("name")
        arguments  = params.get("arguments", {})

        if not tool_name:
            return JSONResponse(make_error(request_id, -32602, "Missing 'name' in params"))

        if tool_name not in TOOL_HANDLERS:
            return JSONResponse(make_error(request_id, -32601, f"Unknown tool: '{tool_name}'"))

        result = TOOL_HANDLERS[tool_name](arguments)
        return JSONResponse(make_success(request_id, result))

    else:
        return JSONResponse(make_error(request_id, -32601, f"Method not found: '{method}'"))