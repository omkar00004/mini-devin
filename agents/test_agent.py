# agents/test_agent.py

import os
import sys
import json
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.markup import escape

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_client import call_mcp
from config import MODEL, TEMPERATURE_AGENT

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

client  = Groq(api_key=os.environ["GROQ_API_KEY"])
console = Console(highlight=False)

FILESYSTEM_MCP_URL = os.environ.get("FILESYSTEM_MCP_URL", "http://localhost:8000")
SHELL_MCP_URL      = os.environ.get("SHELL_MCP_URL",      "http://localhost:8001")


@dataclass
class TestResult:
    """
    Ground truth verdict from TestAgent.
    DebuggerAgent reads this to understand what specifically failed.
    """
    verdict:       str          # "pass" or "fail"
    tests_written: int
    tests_passed:  int
    tests_failed:  int
    failure_details: str        # exact pytest output on failure
    test_file:     str          # path to the test file written

    def passed(self) -> bool:
        return self.verdict == "pass"


def generate_tests(file_path: str, file_content: str, task_description: str) -> str:
    """
    Ask LLM to write pytest tests for the fixed functions.
    Tests must cover: normal case, empty/None input, missing keys.
    Returns test file content as string.
    """
    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{
            "role": "user",
            "content": (
                f"Write pytest tests for this fixed code.\n"
                f"Task that was completed: {task_description}\n\n"
                f"File: {file_path}\n"
                f"```python\n{file_content}\n```\n\n"
                f"Rules:\n"
                f"1. Test the normal case (valid inputs)\n"
                f"2. Test the edge cases that were fixed (empty list, missing keys, None)\n"
                f"3. Use assert statements, no mocking\n"
                f"4. Import the module using: import sys; sys.path.insert(0, '.')\n"
                f"5. Return ONLY the test file content, no explanation, no markdown fences"
            )
        }],
        temperature = TEMPERATURE_AGENT,
        max_tokens  = 1000,
    )
    return response.choices[0].message.content.strip()


def run_tests(task_id: str, file_path: str, task_description: str) -> TestResult:
    """
    Full TestAgent flow:
    1. Read the fixed file
    2. Generate tests via LLM
    3. Write test file to sandbox
    4. Run pytest
    5. Parse results
    6. Return typed TestResult
    """
    t0 = time.time()
    console.print(f"\n  [bold cyan]❯[/bold cyan] TestAgent — {escape(file_path)}")

    # Step 1 — read the file that was fixed
    read_result = call_mcp(FILESYSTEM_MCP_URL, "read_file", {"path": file_path})
    if "error" in read_result:
        console.print(f"    [red]✗ Cannot read {file_path}: {read_result['error']}[/red]")
        return TestResult(
            verdict="fail", tests_written=0, tests_passed=0,
            tests_failed=1,
            failure_details=f"Could not read file: {read_result['error']}",
            test_file=""
        )

    file_content = read_result["content"]

    # Step 2 — generate tests
    console.print(f"    [dim]Generating tests...[/dim]", end="\r")
    test_content = generate_tests(file_path, file_content, task_description)

    # Step 3 — write test file to sandbox
    test_file_path = f"test_{task_id}_{file_path.replace('/', '_')}"
    write_result   = call_mcp(
        FILESYSTEM_MCP_URL, "write_file",
        {"path": test_file_path, "content": test_content}
    )
    if "error" in write_result:
        return TestResult(
            verdict="fail", tests_written=0, tests_passed=0,
            tests_failed=1,
            failure_details=f"Could not write test file: {write_result['error']}",
            test_file=test_file_path
        )

    # Step 4 — run pytest, capture output
    run_result = call_mcp(
        SHELL_MCP_URL, "execute_command",
        {
            "command":    f"python -m pytest {test_file_path} -v --tb=short 2>&1",
            "timeout_ms": 30000
        }
    )

    stdout   = run_result.get("stdout", "")
    exit_code = run_result.get("exit_code", 1)

    # Step 5 — parse pytest output for counts
    tests_passed = stdout.count(" PASSED")
    tests_failed = stdout.count(" FAILED") + stdout.count(" ERROR")
    tests_written = tests_passed + tests_failed

    verdict = "pass" if exit_code == 0 and tests_failed == 0 else "fail"

    elapsed = time.time() - t0
    icon    = "[green]✓[/green]" if verdict == "pass" else "[red]✗[/red]"

    console.print(
        f"    {icon} {tests_passed}/{tests_written} tests passed "
        f"[dim]· {elapsed:.1f}s[/dim]"
    )

    if verdict == "fail" and stdout:
        # Show first 400 chars of failure output
        preview = stdout[:400].strip()
        console.print(f"    [dim]{escape(preview)}[/dim]")

    return TestResult(
        verdict        = verdict,
        tests_written  = tests_written,
        tests_passed   = tests_passed,
        tests_failed   = tests_failed,
        failure_details = stdout if verdict == "fail" else "",
        test_file      = test_file_path,
    )