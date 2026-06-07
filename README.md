<div align="center">

# 🤖 mini-devin

**A multi-agent autonomous code repair system powered by Groq + MCP**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%28llama--3.3--70b%29-F55036?style=flat-square)](https://console.groq.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MCP](https://img.shields.io/badge/Protocol-MCP%20JSON--RPC-blueviolet?style=flat-square)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

*A miniature, production-style implementation of an AI software engineer - plans, codes, tests, debugs, and escalates autonomously.*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Agent Roles](#-agent-roles)
- [Memory System](#-memory-system)
- [MCP Servers](#-mcp-servers)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running](#running)
- [Configuration Reference](#-configuration-reference)
- [How It Works - End to End](#-how-it-works--end-to-end)
- [Extending mini-devin](#-extending-mini-devin)
- [Roadmap](#-roadmap)

---

## 🔍 Overview

**mini-devin** is a from-scratch, multi-agent system that autonomously fixes bugs in a sandboxed codebase. Inspired by [Devin](https://www.cognition.ai/blog/introducing-devin), it implements the core planning → coding → testing → debugging loop without relying on any agent framework.

Key design principles:

- **No magic frameworks** - agents are plain Python processes communicating over HTTP
- **MCP-native** - all file and shell access goes through [Model Context Protocol](https://modelcontextprotocol.io) servers
- **A2A communication** - agents discover and delegate to each other via `/.well-known/agent.json`
- **Structured memory** - three memory layers (episodic, semantic, import graph) prevent repeated mistakes
- **Escalation chains** - failures propagate up from CoderAgent → DebuggerAgent → PlannerAgent, not silently dropped

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         mini-devin System                           │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      PlannerAgent                              │ │
│  │   • Decomposes issues into 2–4 atomic TaskCards               │ │
│  │   • Resolves dependency ordering (depends_on_indices)          │ │
│  │   • Polls CoderAgent and reports final results                 │ │
│  └───────────────────────────┬────────────────────────────────────┘ │
│                              │  POST /tasks/send (TaskCard JSON)    │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                       CoderAgent  :9001                        │ │
│  │   • ReAct loop (read → think → write → run → verify)          │ │
│  │   • Queries episodic + semantic memory before acting           │ │
│  │   • On test failure → escalates to DebuggerAgent               │ │
│  └───────────┬──────────────────────────┬──────────────────────── ┘ │
│              │                          │                            │
│   MCP calls  │          POST /tasks/send│ (on test failure)         │
│              ▼                          ▼                            │
│  ┌──────────────────┐   ┌──────────────────────────────────────────┐│
│  │  Filesystem MCP  │   │            DebuggerAgent  :9002          ││
│  │     :8000        │   │  • Structured reflection before fixing   ││
│  │  Shell MCP :8001 │   │  • Failure classification + policy       ││
│  └──────────────────┘   │  • Escalates with FailureReport if stuck ││
│                          └──────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

All inter-agent calls are synchronous HTTP (no message queue). PlannerAgent uses a poll loop; CoderAgent and DebuggerAgent respond immediately with `202 Accepted` and process tasks on background threads.

---

## 🤝 Agent Roles

### PlannerAgent
> `agents/planner_agent.py`

Receives a free-text issue description and a list of relevant files. Uses the LLM to decompose the issue into an ordered list of **TaskCards**, each representing one atomic unit of work. Dispatches tasks sequentially (respecting `depends_on` ordering), polls for completion, and prints a rich summary.

**Key behaviour:**
- Calls `/.well-known/agent.json` to discover the CoderAgent at startup
- Validates JSON output from the LLM and strips markdown fences defensively
- Second-pass dependency resolution maps LLM-returned indices to real task IDs

---

### CoderAgent
> `agents/coder_agent.py`

The primary code-editing agent. Runs a **ReAct loop** (Reasoning + Acting) for each incoming TaskCard:

1. Queries **episodic memory** for similar past errors
2. **Reads** the target file via Filesystem MCP
3. **Thinks** about the fix (LLM reasoning)
4. **Writes** the corrected file
5. **Runs** the file to verify (Shell MCP)
6. Repeats until the task is done or `MAX_STEPS` is reached

On completion, it runs the project's existing test suite. If tests fail, it escalates to the **DebuggerAgent** rather than retrying blindly.

**Tool output style:** Minimal, Claude Code–style terminal flow - coloured `❯` markers, dim metadata, inline diffs.

---

### DebuggerAgent
> `agents/debugger_agent.py`

Receives escalations from CoderAgent. Distinct from CoderAgent in one key way: **it reflects before it acts**.

For each attempt:
1. **Classifies** the failure (`FORMAT_ERROR`, `ENVIRONMENT`, `SCOPE_TOO_NARROW`, `WRONG_APPROACH`)
2. **Generates a structured reflection** - forces the LLM to answer: *What assumption was wrong? What does the test prove? What is the corrected approach?*
3. Injects the reflection into the ReAct prompt so the fix is theory-driven, not pattern-matched
4. Runs tests again after the fix; if still failing, escalates with a `FailureReport`

---

### TestRunner
> `agents/test_runner.py`

Discovers and runs *existing* repo tests before any fix is attempted. Provides CoderAgent with the exact runtime failure - not a static code reading. Parses pytest output into a structured `ExecutionResult` with `error_type`, `error_location`, and `error_message`.

---

## 🧠 Memory System

mini-devin implements three complementary memory layers:

### Episodic Memory
> `agents/memory/episodic.py` · SQLite (`memory/episodes.db`)

Stores `(signature, reflection, fix_applied, outcome)` tuples indexed by a stable **error signature** derived from the error type, function name, and file name - stripping ephemeral line numbers and values.

```
"KeyError: 'timeout' at config.py:34 in parse_config"
→ signature: "KeyError::parse_config::config.py"
```

On future similar errors, the agent is shown past reflections as *hints* (not raw code) to avoid copy-paste without understanding.

### Semantic Memory
> `agents/memory/semantic.py` · ChromaDB + `all-MiniLM-L6-v2`

Indexes the entire sandbox codebase as function/class-level chunks using AST parsing + sentence-transformer embeddings. At startup, `run.py` calls `index_directory('sandbox')` so PlannerAgent can find relevant files by semantic similarity rather than exact name matching.

### Import Graph
> `agents/memory/import_graph.py`

Builds a static bidirectional import graph from Python AST analysis. Given a buggy file, the graph finds all files it imports **and** all files that import it - up to N hops - so the planner can include transitively-related files in the task context.

---

## 🔌 MCP Servers

Both servers implement the [Model Context Protocol](https://modelcontextprotocol.io) JSON-RPC 2.0 spec.

| Server | Port | Tools |
|--------|------|-------|
| `mcp_servers/filesystem/` | `8000` | `read_file`, `write_file`, `list_dir` |
| `mcp_servers/shell/` | `8001` | `execute_command` |
| `mcp_servers/github/` | `8002` | *(planned)* |

Agents call tools via the shared `agents/mcp_client.py`:

```python
result = call_mcp("http://localhost:8000", "read_file", {"path": "sandbox/bug.py"})
```

---

## 📁 Project Structure

```
mini-devin/
├── run.py                        # Orchestrator - starts all services, then runs PlannerAgent
├── config.py                     # Single source of truth for all hyperparameters
├── .env                          # API keys (gitignored)
├── .env.example                  # Template for .env
├── requirements.txt
│
├── agents/
│   ├── planner_agent.py          # Issue decomposition + A2A orchestration
│   ├── coder_agent.py            # ReAct coding loop (FastAPI server on :9001)
│   ├── debugger_agent.py         # Reflection-driven debug loop (FastAPI on :9002)
│   ├── test_runner.py            # Discovers + runs existing repo tests
│   ├── test_agent.py             # Writes NEW tests to validate a fix
│   ├── mcp_client.py             # Shared JSON-RPC MCP client
│   │
│   ├── memory/
│   │   ├── episodic.py           # SQLite-backed episode store + recall
│   │   ├── semantic.py           # ChromaDB vector index of codebase chunks
│   │   ├── import_graph.py       # Static AST-based import graph builder
│   │   ├── summarizer.py         # Context window compression via LLM
│   │   └── watcher.py            # (optional) file watcher for live re-indexing
│   │
│   └── models/
│       ├── task_card.py          # TaskCard, TaskContext, TaskArtifact dataclasses
│       └── failure.py            # FailureType, FailureReport, ESCALATION_POLICY
│
├── mcp_servers/
│   ├── filesystem/server.py      # File read/write/list MCP server
│   ├── shell/server.py           # Shell command execution MCP server
│   └── github/                   # (planned) GitHub MCP server
│
├── memory/
│   ├── episodes.db               # Persistent episodic memory (SQLite)
│   └── codebase_index/           # Persistent vector index (ChromaDB)
│
├── sandbox/                      # Buggy files for agents to fix
└── scripts/
    └── test_import_graph.py      # Unit tests for the import graph module
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11+**
- A [Groq API key](https://console.groq.com) (free tier works)
- `pip` or `uv`

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/mini-devin.git
cd mini-devin/mini-devin

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90 MB) on first run. It is cached locally after that.

### Configuration

Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_...               # Required - from console.groq.com
FILESYSTEM_MCP_PORT=8000           # Default, change if port is occupied
SHELL_MCP_PORT=8001
DEBUGGER_AGENT_URL=http://localhost:9002
DEBUGGER_AGENT_PORT=9002
```

### Running

**Option 1 - Full system (recommended)**

`run.py` starts all services in the correct order, indexes the sandbox, then launches PlannerAgent:

```bash
python run.py
```

Expected startup output:
```
Starting mini-devin...

  ✓ filesystem-mcp  port 8000 (pid 12345)
  ✓ shell-mcp       port 8001 (pid 12346)
  ✓ CoderAgent      port 9001 (pid 12347)
  ✓ DebuggerAgent   port 9002 (pid 12348)

Waiting for servers...

Indexing codebase...
  ✓ {'files_indexed': 3, 'chunks_indexed': 27}

────────────────────────────────────────────────────────────

╭─ PlannerAgent (llama-3.3-70b-versatile)
│
│  The file multi_bug.py has multiple independent bugs ...
╰──────────────────────────────────────────────────────────
```

**Option 2 - Individual agents**

Start services manually for development:

```bash
# Terminal 1 - MCP servers
cd mcp_servers/filesystem && uvicorn server:app --port 8000
cd mcp_servers/shell && uvicorn server:app --port 8001

# Terminal 2 - Agents
cd agents && python coder_agent.py        # starts on :9001
cd agents && python debugger_agent.py     # starts on :9002

# Terminal 3 - Run the planner
cd agents && python planner_agent.py
```

---

## ⚙️ Configuration Reference

All tuneable parameters live in a single file: `config.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL` | `llama-3.3-70b-versatile` | Groq model used by all agents |
| `MAX_STEPS` | `20` | Max ReAct iterations per task |
| `MAX_TOOL_RETRIES` | `2` | Retries on malformed tool call JSON |
| `SUMMARIZE_THRESHOLD` | `15` | Message count that triggers context compression |
| `KEEP_RECENT` | `4` | Messages kept intact after summarization |
| `MAX_TOKENS_SUMMARY` | `600` | Token budget for the context summary |
| `POLL_INTERVAL_SEC` | `2` | PlannerAgent polling interval |
| `POLL_TIMEOUT_SEC` | `300` | Max wait before polling times out |
| `TEMPERATURE_AGENT` | `0.2` | CoderAgent temperature (low = precise) |
| `TEMPERATURE_PLANNER` | `0.1` | PlannerAgent temperature (deterministic) |
| `TEMPERATURE_SUMMARY` | `0.1` | Summarizer temperature (factual) |
| `MAX_DEBUG_ATTEMPTS` | `3` | DebuggerAgent retry limit before escalation |
| `DEBUGGER_AGENT_PORT` | `9002` | Port for DebuggerAgent HTTP server |

---

## 🔄 How It Works - End to End

```
1. User edits `planner_agent.py` → sets the `issue` string describing the bug(s)

2. PlannerAgent calls Groq LLM:
   "Break this issue into 2–4 atomic tasks, return JSON"
   
3. For each TaskCard (in dependency order):
   a. PlannerAgent POST /tasks/send → CoderAgent :9001
      └─ Returns 202 immediately; task runs in background thread

   b. CoderAgent queries episodic memory for similar past errors
   c. CoderAgent runs ReAct loop:
      Read file → Reason → Write fix → Run code → Verify output
   d. TestRunner runs existing pytest tests on modified files

   e. If tests PASS:
      CoderAgent marks task "completed" → PlannerAgent polls → prints ✓

   f. If tests FAIL:
      CoderAgent POST /tasks/send → DebuggerAgent :9002
      DebuggerAgent:
        1. Classifies failure type
        2. Generates structured reflection (what assumption was wrong?)
        3. Runs its own ReAct loop with the reflection injected
        4. Re-runs tests
        If still failing after 3 attempts → returns FailureReport with escalation=True
      
   g. Episodic memory updated with outcome (success / failure)

4. PlannerAgent prints summary: N/M tasks completed · files modified · elapsed time
```

---

## 🧩 Extending mini-devin

### Add a new MCP tool

1. Add the handler to `mcp_servers/filesystem/server.py` or `mcp_servers/shell/server.py` following the existing JSON-RPC pattern.
2. Agents auto-discover tools at startup via `list_tools()` - no agent code changes needed.

### Add a new Agent

1. Create `agents/your_agent.py` as a FastAPI app with:
   - `GET /.well-known/agent.json` - agent discovery card
   - `POST /tasks/send` - accepts TaskCard JSON, returns `202 Accepted`
   - `GET /tasks/{task_id}` - returns task status
2. Register it in `run.py`'s startup sequence.

### Change the LLM model

Update `MODEL` in `config.py`. Any [Groq-supported model](https://console.groq.com/docs/models) works. For higher quality on complex tasks, try `llama-3.1-405b-reasoning`.

### Point at a real codebase (not sandbox)

Change the `sandbox/` path in `run.py`'s `index_directory()` call and update the `issue` string in `planner_agent.py` with a real bug description.

---

## 🗺 Roadmap

- [ ] **GitHub MCP server** - read issues and PRs directly from a repository
- [ ] **PlannerAgent replanning** - on `FailureReport` escalation, replan with narrower scope
- [ ] **Parallel task execution** - run independent tasks concurrently (currently sequential)
- [ ] **Web UI** - real-time task progress dashboard
- [ ] **Multi-language support** - extend Shell MCP with Node, Go, Java runners
- [ ] **Langfuse tracing** - full observability for every agent step

---

## 📄 License

MIT - see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ as a learning project. Not affiliated with Cognition AI or the official Devin project.</sub>
</div>
