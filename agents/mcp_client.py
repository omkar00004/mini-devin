# agents/mcp_client.py

import json
import urllib.request
import uuid

def call_mcp(server_url: str, tool_name: str, arguments: dict) -> dict:
    """
    Send a tools/call JSON-RPC request to an MCP server.
    Returns the result dict directly.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        server_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    for attempt in range(2):   # try twice
        try:
            with urllib.request.urlopen(req, timeout=35) as response:
                data = json.loads(response.read())
            ...
            return data.get("result", {})
        except urllib.error.URLError as e:
            if attempt == 0:
                continue       # retry once
            return {"error": f"Could not reach MCP server at {server_url}: {str(e)}"}

        except Exception as e:
            return {"error": f"Unexpected error calling {tool_name}: {str(e)}"}


def list_tools(server_url: str) -> list:
    """
    Fetch the tools/list from an MCP server.
    Returns list of tool schemas.
    """
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/list",
        "params": {}
    }

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        server_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
        return data.get("result", {}).get("tools", [])
    except Exception as e:
        print(f"[MCP] Failed to list tools from {server_url}: {e}")
        return []