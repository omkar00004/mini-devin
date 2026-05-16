# mcp_servers/filesystem/server.py

import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

PORT = int(os.environ.get("FILESYSTEM_MCP_PORT", 8000))


# ── Sandbox Configuration ──────────────────────────────────────────────────
# All file operations are jailed to this directory.
# Change this to wherever you want the sandbox to live.
SANDBOX_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../../sandbox")
)

# ── Tool Schemas (what tools/list returns) ─────────────────────────────────
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file inside the sandbox.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file inside the sandbox"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file inside the sandbox. Creates the file if it doesn't exist.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file inside the sandbox"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_dir",
        "description": "List files and folders inside a sandbox directory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the directory. Use '.' for sandbox root."
                }
            },
            "required": ["path"]
        }
    }
]

# ── Path Safety ────────────────────────────────────────────────────────────

def safe_path(user_path: str) -> str:
    """
    Resolves a user-provided path and ensures it stays inside SANDBOX_ROOT.
    Raises ValueError if the resolved path escapes the sandbox.
    """
    # Join with sandbox root, then resolve all symlinks and ../
    resolved = os.path.realpath(
        os.path.join(SANDBOX_ROOT, user_path)
    )
    
    # Add os.sep so "sandbox_evil" can't prefix-match "sandbox/"
    sandbox_prefix = SANDBOX_ROOT + os.sep
    
    if resolved != SANDBOX_ROOT and not resolved.startswith(sandbox_prefix):
        raise ValueError(
            f"Path escape attempt blocked: '{user_path}' resolves outside sandbox"
        )
    
    return resolved

# ── Tool Implementations ───────────────────────────────────────────────────

def tool_read_file(arguments: dict) -> dict:
    path = arguments.get("path")
    if not path:
        return {"error": "Missing required argument: path"}
    
    try:
        resolved = safe_path(path)
    except ValueError as e:
        return {"error": str(e)}
    
    if not os.path.exists(resolved):
        return {"error": f"File not found: {path}"}
    
    if not os.path.isfile(resolved):
        return {"error": f"Path is a directory, not a file: {path}"}
    
    # Size guard — refuse to read files over 100KB
    size_bytes = os.path.getsize(resolved)
    if size_bytes > 100_000:
        return {
            "error": f"File too large to read directly: {size_bytes} bytes. "
                     f"Max is 100,000 bytes."
        }
    
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "content": content,
            "path": path,
            "size_bytes": size_bytes,
            "lines": content.count("\n") + 1
        }
    except UnicodeDecodeError:
        return {"error": f"File is not valid UTF-8 text. Binary files are not supported."}


def tool_write_file(arguments: dict) -> dict:
    path = arguments.get("path")
    content = arguments.get("content")
    MAX_WRITE_BYTES = 1_000_000  # 1MB
    
    if not path:
        return {"error": "Missing required argument: path"}
    if content is None:
        return {"error": "Missing required argument: content"}
    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        return {"error": f"Content too large to write: max {MAX_WRITE_BYTES} bytes"}
    
    try:
        resolved = safe_path(path)
    except ValueError as e:
        return {"error": str(e)}
    
    # Auto-create parent directories if they don't exist
    parent_dir = os.path.dirname(resolved)
    os.makedirs(parent_dir, exist_ok=True)
    
    try:
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        return {
            "success": True,
            "path": path,
            "bytes_written": len(content.encode("utf-8"))
        }
    except Exception as e:
        return {"error": f"Write failed: {str(e)}"}



def tool_list_dir(arguments: dict) -> dict:
    path = arguments.get("path", ".")
    
    try:
        resolved = safe_path(path)
    except ValueError as e:
        return {"error": str(e)}
    
    if not os.path.exists(resolved):
        return {"error": f"Directory not found: {path}"}
    
    if not os.path.isdir(resolved):
        return {"error": f"Path is a file, not a directory: {path}"}
    
    try:
        entries = []
        for name in sorted(os.listdir(resolved)):
            full_path = os.path.join(resolved, name)
            entries.append({
                "name": name,
                "type": "directory" if os.path.isdir(full_path) else "file",
                "size_bytes": os.path.getsize(full_path) if os.path.isfile(full_path) else None
            })
        return {
            "path": path,
            "entries": entries,
            "count": len(entries)
        }
    except Exception as e:
        return {"error": f"List failed: {str(e)}"}


# ── Tool Dispatch Table ────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "read_file":  tool_read_file,
    "write_file": tool_write_file,
    "list_dir":   tool_list_dir,
}

# ── JSON-RPC Handler ───────────────────────────────────────────────────────

def make_success(request_id, result: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    }

def make_error(request_id, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }

@app.post("/")
async def handle_jsonrpc(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(make_error(None, -32700, "Parse error: invalid JSON"))
    
    request_id = body.get("id")
    method     = body.get("method")
    params     = body.get("params", {})
    
    # ── Method: tools/list ─────────────────────────────────────────────────
    if method == "tools/list":
        return JSONResponse(make_success(request_id, {"tools": TOOLS}))
    
    # ── Method: tools/call ─────────────────────────────────────────────────
    elif method == "tools/call":
        tool_name  = params.get("name")
        arguments  = params.get("arguments", {})
        
        if not tool_name:
            return JSONResponse(
                make_error(request_id, -32602, "Missing 'name' in params")
            )
        
        if tool_name not in TOOL_HANDLERS:
            return JSONResponse(
                make_error(request_id, -32601, f"Unknown tool: '{tool_name}'")
            )
        
        # Call the tool
        tool_result = TOOL_HANDLERS[tool_name](arguments)
        
        # Note: tool errors are returned as results, not JSON-RPC errors
        # (the tool call itself succeeded — it just returned an error result)
        return JSONResponse(make_success(request_id, tool_result))
    
    # ── Unknown method ─────────────────────────────────────────────────────
    else:
        return JSONResponse(
            make_error(request_id, -32601, f"Method not found: '{method}'")
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)