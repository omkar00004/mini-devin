# run.py

import subprocess, time, sys, os

def start(name, cmd, cwd):
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  ✓ {name} (pid {proc.pid})")
    return proc

if __name__ == "__main__":
    base  = os.path.dirname(os.path.abspath(__file__))
    procs = []

    print("\nStarting mini-devin...\n")

    try:
        procs.append(start(
            "filesystem-mcp  port 8000",
            ["uvicorn", "server:app", "--port", "8000", "--log-level", "error"],
            cwd=os.path.join(base, "mcp_servers/filesystem")
        ))
        procs.append(start(
            "shell-mcp       port 8001",
            ["uvicorn", "server:app", "--port", "8001", "--log-level", "error"],
            cwd=os.path.join(base, "mcp_servers/shell")
        ))
        procs.append(start(
            "CoderAgent      port 9001",
            [sys.executable, "coder_agent.py"],
            cwd=os.path.join(base, "agents")
        ))
        procs.append(start(
            "DebuggerAgent   port 9002",
            [sys.executable, "debugger_agent.py"],
            cwd=os.path.join(base, "agents")
        ))

        print("\nWaiting for servers...")
        time.sleep(3)

        # Index sandbox before running planner
        print("\nIndexing codebase...")
        subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0,'agents'); "
             "from memory.semantic import index_directory; "
             "r = index_directory('sandbox'); print(f'  ✓ {r}')"
            ],
            cwd=base
        )

        print("\n" + "─" * 60 + "\n")
        subprocess.run(
            [sys.executable, "planner_agent.py"],
            cwd=os.path.join(base, "agents")
        )

    finally:
        print("\nShutting down...")
        for p in procs:
            p.terminate()