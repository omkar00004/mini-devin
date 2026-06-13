# mcp_servers/shell/test_docker.py
# Run from mini-devin/ root: python mcp_servers/shell/test_docker.py

import json
import urllib.request

BASE = "http://localhost:8001"

def call(method, params=None):
    body = json.dumps({
        "jsonrpc": "2.0", "id": "test",
        "method": method, "params": params or {}
    }).encode()
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["result"]

def test(label, result, expect_exit=0, expect_in_stdout=None, expect_error=None):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    if "error" in result:
        if expect_error:
            print(f"  ✓ Got expected error: {result['error']}")
        else:
            print(f"  ✗ Unexpected error: {result['error']}")
        return
    exit_ok = result["exit_code"] == expect_exit
    stdout_ok = (expect_in_stdout is None) or (expect_in_stdout in result["stdout"])
    print(f"  exit_code: {result['exit_code']}  {'✓' if exit_ok else '✗'}")
    print(f"  stdout:    {result['stdout'][:200]!r}")
    if result["stderr"]:
        print(f"  stderr:    {result['stderr'][:200]!r}")
    print(f"  runtime:   {result['runtime_ms']}ms")
    if expect_in_stdout:
        print(f"  contains {expect_in_stdout!r}: {'✓' if stdout_ok else '✗'}")
    if exit_ok and stdout_ok:
        print(f"  → PASS")
    else:
        print(f"  → FAIL")

# ── TEST 1: tools/list ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1: tools/list")
tools = call("tools/list")["tools"]
names = [t["name"] for t in tools]
print(f"  tools: {names}")
assert "execute_command" in names and "run_python" in names
print("  → PASS")

# ── TEST 2: basic Python ───────────────────────────────────────────────────
r = call("tools/call", {"name": "execute_command", "arguments": {
    "command": "python -c \"print('hello from docker')\""
}})
test("basic python execution", r, expect_exit=0, expect_in_stdout="hello from docker")

# ── TEST 3: sandbox path rewriting ────────────────────────────────────────
r = call("tools/call", {"name": "execute_command", "arguments": {
    "command": "python sandbox/calculator.py"
}})
test("sandbox/ path rewriting (calculator.py)", r)
# exit_code may be 1 if calculator.py has bugs — that's expected and correct

# ── TEST 4: network is blocked ────────────────────────────────────────────
r = call("tools/call", {"name": "execute_command", "arguments": {
    "command": "python -c \"import urllib.request; urllib.request.urlopen('http://example.com')\""
}})
test("network blocked (--network=none)", r, expect_exit=1)
# Should fail with socket error — no network in container

# ── TEST 5: run_python ────────────────────────────────────────────────────
r = call("tools/call", {"name": "run_python", "arguments": {
    "code": "x = 2 + 2\nprint(f'result: {x}')"
}})
test("run_python basic", r, expect_exit=0, expect_in_stdout="result: 4")

# ── TEST 6: run_python exception ──────────────────────────────────────────
r = call("tools/call", {"name": "run_python", "arguments": {
    "code": "raise ValueError('deliberate error')"
}})
test("run_python exception captured", r, expect_exit=1)
print(f"  stderr contains 'ValueError': {'ValueError' in r.get('stderr','')}")

# ── TEST 7: pytest on sandbox tests ──────────────────────────────────────
r = call("tools/call", {"name": "execute_command", "arguments": {
    "command": "python -m pytest tests/test_calculator.py -v --tb=short"
}})
test("pytest on sandbox tests", r)
print(f"  stdout preview: {r.get('stdout','')[:400]}")

print("\n" + "="*60)
print("ALL TESTS DONE")