# run.py — starts all servers and runs the full system

import subprocess
import time
import sys
import os

def start_server(name: str, cmd: list, cwd: str) -> subprocess.Popen:
    """Start a server process and return the handle."""
    proc = subprocess.Popen(
        cmd,
        cwd    = cwd,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )
    print(f"  ✓ {name} started (pid {proc.pid})")
    return proc

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    print("\nStarting mini-devin...\n")

    procs = []
    try:
        # 1 — filesystem-mcp
        procs.append(start_server(
            "filesystem-mcp (port 8000)",
            ["uvicorn", "server:app", "--port", "8000", "--log-level", "error"],
            cwd=os.path.join(base, "mcp_servers/filesystem")
        ))

        # 2 — shell-mcp
        procs.append(start_server(
            "shell-mcp     (port 8001)",
            ["uvicorn", "server:app", "--port", "8001", "--log-level", "error"],
            cwd=os.path.join(base, "mcp_servers/shell")
        ))

        # 3 — CoderAgent A2A server
        procs.append(start_server(
            "CoderAgent    (port 9001)",
            [sys.executable, "coder_agent.py"],
            cwd=os.path.join(base, "agents")
        ))

        # Give servers time to bind their ports
        print("\nWaiting for servers to start...")
        time.sleep(3)

        # 4 — PlannerAgent runs and exits
        print("\n" + "─" * 60 + "\n")
        planner = subprocess.run(
            [sys.executable, "planner_agent.py"],
            cwd=os.path.join(base, "agents")
        )

    finally:
        print("\nShutting down servers...")
        for proc in procs:
            proc.terminate()
        print("Done.")