# mcp_servers/shell/server.py

import os
import sys
import time
import shlex          # ← NEW: safe command tokenization for Docker args
import subprocess
import tempfile
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DOCKER_IMAGE, DOCKER_MEMORY, DOCKER_CPUS, DOCKER_TMPFS

app = FastAPI()

# ── Configuration ──────────────────────────────────────────────────────────
SANDBOX_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../../sandbox")
)

DEFAULT_TIMEOUT_MS = 10_000   # 10 seconds default
MAX_TIMEOUT_MS     = 30_000   # 30 seconds hard ceiling
MAX_OUTPUT_CHARS   = 8_000    # truncate stdout/stderr beyond this

# ── Tool Schemas ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "execute_command",
        "description": (
            "Run a shell command inside an isolated Docker sandbox. "
            "Network is disabled. Memory is capped at 256MB. "
            "Filesystem is read-only except /sandbox (your working directory). "
            "Use this to run files: 'python file.py', 'pytest tests/', etc. "
            "Each call gets a fresh container — no state persists between calls. "
            "Chain commands with && for multi-step operations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to execute. Paths starting with 'sandbox/' "
                        "are automatically mapped to '/sandbox/' inside the container."
                    )
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": (
                        f"Max execution time in milliseconds. "
                        f"Default: {DEFAULT_TIMEOUT_MS}. Max: {MAX_TIMEOUT_MS}."
                    )
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "run_python",
        "description": (
            "Execute a Python code string inside the Docker sandbox. "
            "Runs in an isolated container — no network, 256MB RAM limit. "
            "Working directory is /sandbox. "
            "Print statements go to stdout. Exceptions go to stderr. "
            "For running existing files, use execute_command instead."
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
                    "description": (
                        f"Max execution time in milliseconds. "
                        f"Default: {DEFAULT_TIMEOUT_MS}. Max: {MAX_TIMEOUT_MS}."
                    )
                }
            },
            "required": ["code"]
        }
    }
]

# ── Output Truncation ──────────────────────────────────────────────────────

def truncate_output(text: str) -> dict:
    """
    If text exceeds MAX_OUTPUT_CHARS, keep only the last MAX_OUTPUT_CHARS.
    Errors appear at the END of output — keeping the tail preserves them.
    """
    if len(text) <= MAX_OUTPUT_CHARS:
        return {"text": text, "truncated": False, "original_length": len(text)}
    return {
        "text": text[-MAX_OUTPUT_CHARS:],
        "truncated": True,
        "original_length": len(text),
        "note": f"Output truncated. Showing last {MAX_OUTPUT_CHARS} of {len(text)} chars."
    }

# ── Path Rewriting ─────────────────────────────────────────────────────────

def rewrite_paths(command: str) -> str:
    """
    Translate host-relative sandbox paths to container-absolute paths.

    CoderAgent sends:  "python sandbox/calculator.py"
    Container needs:   "python /sandbox/calculator.py"

    Rule: replace "sandbox/" with "/sandbox/" anywhere in the command.
    --workdir /sandbox handles bare relative paths like "tests/" correctly.
    """
    return command.replace("sandbox/", "/sandbox/")

# ── Docker Execution ───────────────────────────────────────────────────────

def run_in_docker(command: str, timeout_sec: float) -> dict:
    """
    Run a shell command inside an isolated Docker container.

    Container config (all enforced by Linux kernel, not by Python code):
      --rm              auto-remove container after exit (no leftover containers)
      --memory          hard RAM limit — kernel OOM-kills if exceeded
      --cpus            CPU quota — kernel throttles if exceeded
      --network=none    no network interfaces — kernel drops all socket syscalls
      --read-only       container filesystem is immutable
      --tmpfs /tmp      writable RAM disk for .pyc cache, pip temp files
      -v sandbox:/sandbox  host sandbox directory mounted into container
      --workdir /sandbox   default working directory inside container

    The command runs via "sh -c" so &&, |, ; all work correctly.
    subprocess here runs a trusted system command (docker), not untrusted code.
    The untrusted code runs inside Docker with kernel-enforced limits.
    """
    rewritten = rewrite_paths(command)

    docker_cmd = [
        "docker", "run",
        "--rm",                              # destroy container on exit
        f"--memory={DOCKER_MEMORY}",         # 256m RAM limit
        f"--cpus={DOCKER_CPUS}",             # 0.5 CPU limit
        "--network=none",                    # no network
        "--read-only",                       # immutable container filesystem
        "--tmpfs", DOCKER_TMPFS,             # writable /tmp RAM disk
        "-v", f"{SANDBOX_ROOT}:/sandbox",   # mount host sandbox
        "--workdir", "/sandbox",             # working directory
        DOCKER_IMAGE,                        # python:3.11-slim
        "sh", "-c", rewritten                # run via shell
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec
            # No cwd= or shell=True here — docker_cmd is a trusted list,
            # not a shell string. shell=False is the default and is correct.
        )

        runtime_ms = int((time.time() - start_time) * 1000)
        stdout_info = truncate_output(result.stdout)
        stderr_info = truncate_output(result.stderr)

        return {
            "stdout":           stdout_info["text"],
            "stderr":           stderr_info["text"],
            "exit_code":        result.returncode,
            "runtime_ms":       runtime_ms,
            "stdout_truncated": stdout_info["truncated"],
            "stderr_truncated": stderr_info["truncated"],
        }

    except subprocess.TimeoutExpired:
        runtime_ms = int((time.time() - start_time) * 1000)
        return {
            "stdout":           "",
            "stderr":           f"Process killed: exceeded timeout of {timeout_sec * 1000:.0f}ms",
            "exit_code":        -1,
            "runtime_ms":       runtime_ms,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }

    except Exception as e:
        return {"error": f"Docker execution failed: {str(e)}"}

# ── Tool Implementations ───────────────────────────────────────────────────

def tool_execute_command(arguments: dict) -> dict:
    command    = arguments.get("command", "").strip()
    timeout_ms = int(arguments.get("timeout_ms", DEFAULT_TIMEOUT_MS))

    if not command:
        return {"error": "Missing required argument: command"}

    timeout_ms  = min(timeout_ms, MAX_TIMEOUT_MS)
    timeout_sec = timeout_ms / 1000

    return run_in_docker(command, timeout_sec)


def tool_run_python(arguments: dict) -> dict:
    code       = arguments.get("code", "").strip()
    timeout_ms = int(arguments.get("timeout_ms", DEFAULT_TIMEOUT_MS))

    if not code:
        return {"error": "Missing required argument: code"}

    timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)

    temp_path = None  # initialize before try so finally can always check it

    try:
        # Write code to a temp file inside SANDBOX_ROOT on the host.
        # SANDBOX_ROOT is volume-mounted at /sandbox inside the container,
        # so the container sees this file at /sandbox/tmp_XXXX.py automatically.
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=SANDBOX_ROOT,
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        # Get just the filename — container sees it at /sandbox/{filename}
        filename = os.path.basename(temp_path)

        result = run_in_docker(f"python /sandbox/{filename}", timeout_ms / 1000)

        # Scrub the temp filename from error messages — agent doesn't need to see it
        if result.get("stderr"):
            result["stderr"] = result["stderr"].replace(
                f"/sandbox/{filename}", "<python_script>"
            )

        return result

    except Exception as e:
        return {"error": f"Failed to run python code: {str(e)}"}

    finally:
        # Always runs — success, exception, or crash
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
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            return JSONResponse(make_error(request_id, -32602, "Missing 'name' in params"))
        if tool_name not in TOOL_HANDLERS:
            return JSONResponse(make_error(request_id, -32601, f"Unknown tool: '{tool_name}'"))

        result = TOOL_HANDLERS[tool_name](arguments)
        return JSONResponse(make_success(request_id, result))

    else:
        return JSONResponse(make_error(request_id, -32601, f"Method not found: '{method}'"))