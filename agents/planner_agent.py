# agents/planner_agent.py

import os, sys, json, time
import urllib.request
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.markup import escape

sys.path.insert(0, os.path.dirname(__file__))
from models.task_card import TaskCard, TaskContext
from memory.episodic import init_db

# Import all hyperparameters from one place
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL,
    MAX_STEPS,
    TEMPERATURE_AGENT,
    SUMMARIZE_THRESHOLD,
    KEEP_RECENT,
    TEMPERATURE_PLANNER,
    POLL_INTERVAL_SEC,
    POLL_TIMEOUT_SEC,
)


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
CODER_AGENT_URL = os.environ.get("CODER_AGENT_URL", "http://localhost:9001")

client  = Groq(api_key=GROQ_API_KEY)
console = Console(highlight=False)

# ── Banner ─────────────────────────────────────────────────────────────────

def print_init(issue: str):
    w = console.width
    console.print(f"\n[bold bright_white]╭─ PlannerAgent ({MODEL})[/bold bright_white]")
    console.print(f"[bold bright_white]│[/bold bright_white]")
    for line in issue.strip().splitlines():
        console.print(f"[bold bright_white]│[/bold bright_white]  {escape(line.strip())}")
    console.print(f"[bold bright_white]╰{'─' * (w - 2)}[/bold bright_white]\n")

# ── Planning ───────────────────────────────────────────────────────────────

def plan_tasks(issue: str, repo_files: list) -> list[TaskCard]:
    """
    Ask LLM to decompose issue into atomic tasks.
    Key fix: depends_on must reference task_ids, not descriptions.
    We assign IDs first, then let the model reference them by index.
    """
    files_list = "\n".join(f"  - {f}" for f in repo_files) if repo_files else "  (unknown)"

    prompt = f"""You are PlannerAgent. Break this issue into 2-4 atomic tasks.

Issue:
{issue}

Files in sandbox:
{files_list}

Rules:
- Each task must be completable independently by a single agent
- depends_on must use the task's index number (0, 1, 2...) NOT its description
- relevant_files must name exact files to edit
- acceptance_criteria must be specific and testable
- Return ONLY valid JSON array, no markdown fences, no explanation

[
  {{
    "description": "...",
    "context": {{
      "relevant_files": ["filename.py"],
      "error_trace": "describe the specific error",
      "issue_summary": "one sentence summary"
    }},
    "acceptance_criteria": ["specific testable criterion"],
    "depends_on_indices": []
  }}
]"""

    response = client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = TEMPERATURE_PLANNER,
        max_tokens  = 1500,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model added them
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
    raw = raw.strip()

    try:
        tasks_data = json.loads(raw)
    except json.JSONDecodeError as e:
        console.print(f"  [red]✗ Planner returned invalid JSON: {e}[/red]")
        return []

    # Create all cards first so we have their IDs
    task_cards = []
    for t in tasks_data:
        card = TaskCard(
            description         = t["description"],
            context             = TaskContext(
                relevant_files  = t.get("context", {}).get("relevant_files", []),
                error_trace     = t.get("context", {}).get("error_trace"),
                issue_summary   = t.get("context", {}).get("issue_summary"),
            ),
            acceptance_criteria = t.get("acceptance_criteria", []),
            depends_on          = [],   # filled in next pass
        )
        task_cards.append(card)

    # Second pass — resolve depends_on_indices → actual task_ids
    # This is the fix: model returns indices [0,1,2], we map to real IDs
    for i, (card, t) in enumerate(zip(task_cards, tasks_data)):
        indices = t.get("depends_on_indices", [])
        card.depends_on = [
            task_cards[idx].task_id
            for idx in indices
            if isinstance(idx, int) and 0 <= idx < len(task_cards) and idx != i
        ]

    return task_cards

# ── A2A Communication ──────────────────────────────────────────────────────

def send_task(task_card: TaskCard) -> bool:
    payload = json.dumps(task_card.to_dict()).encode("utf-8")
    req     = urllib.request.Request(
        f"{CODER_AGENT_URL}/tasks/send",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 202
    except Exception as e:
        console.print(f"  [red]✗ Failed to send {task_card.task_id}: {e}[/red]")
        return False


def poll_task(task_id: str) -> dict:
    """Poll every POLL_INTERVAL_SEC seconds. No LLM calls — pure HTTP GET."""
    url      = f"{CODER_AGENT_URL}/tasks/{task_id}"
    deadline = time.time() + POLL_TIMEOUT_SEC
    dots     = 0

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url), timeout=10
            ) as resp:
                data   = json.loads(resp.read())
                status = data.get("status")

            if status in ("completed", "failed"):
                return data

            dots = (dots % 3) + 1
            console.print(
                f"  [dim]waiting{'.' * dots}[/dim]   ",
                end="\r"
            )
            time.sleep(POLL_INTERVAL_SEC)

        except Exception as e:
            console.print(f"\n  [red]✗ Poll error: {e}[/red]")
            time.sleep(POLL_INTERVAL_SEC)

    return {"status": "timeout", "task_id": task_id}

# ── Orchestration ──────────────────────────────────────────────────────────

def run_planner(issue: str, repo_files: list = None):
    t0 = time.time()
    print_init(issue)

    # Discover CoderAgent
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{CODER_AGENT_URL}/.well-known/agent.json"),
            timeout=5
        ) as resp:
            agent_card = json.loads(resp.read())
        console.print(f"  [green]✓[/green] Found {agent_card['name']}: "
                      f"[dim]{agent_card['description']}[/dim]\n")
    except Exception as e:
        console.print(f"  [red]✗ CoderAgent not reachable: {e}[/red]")
        console.print(f"  [dim]Start it first: python coder_agent.py[/dim]")
        return

    # Plan
    console.print("  [dim]Planning...[/dim]")
    task_cards = plan_tasks(issue, repo_files or [])

    if not task_cards:
        console.print("  [red]✗ No tasks generated[/red]")
        return

    # Print plan
    console.print(f"  [bold]Plan:[/bold] {len(task_cards)} tasks\n")
    for i, card in enumerate(task_cards):
        deps = f" [dim]← depends on {card.depends_on}[/dim]" if card.depends_on else ""
        console.print(f"    {i+1}. [white]{card.task_id}[/white] — "
                      f"{escape(card.description)}{deps}")

    w = console.width
    console.print(f"\n  [dim]{'─' * (w - 4)}[/dim]\n")

    # Execute — sequential, respecting depends_on
    completed_ids: set[str] = set()
    all_artifacts            = []

    for card in task_cards:
        # Check dependencies — skip if any unmet
        unmet = [d for d in card.depends_on if d not in completed_ids]
        if unmet:
            console.print(f"  [yellow]⏸[/yellow] {card.task_id} "
                          f"[dim]skipped — unmet deps: {unmet}[/dim]")
            continue

        console.print(f"  [bold cyan]❯[/bold cyan] "
                      f"[white]{card.task_id}[/white] — {escape(card.description)}")

        if not send_task(card):
            console.print(f"    [red]✗ rejected[/red]")
            continue

        result = poll_task(card.task_id)
        console.print()  # clear the \r polling line

        status = result.get("status")

        if status == "completed":
            completed_ids.add(card.task_id)
            artifacts = result.get("artifacts", [])
            all_artifacts.extend(artifacts)
            files = [a["file"] for a in artifacts if a.get("file")]
            console.print(f"    [green]✓ completed[/green]"
                          + (f" [dim]— {', '.join(files)}[/dim]" if files else ""))

        elif status == "failed":
            err = str(result.get("error", "unknown"))[:120]
            console.print(f"    [red]✗ failed[/red] [dim]{escape(err)}[/dim]")

        else:
            console.print(f"    [yellow]⚠ {status}[/yellow]")

    # Summary
    elapsed      = time.time() - t0
    unique_files = list({a["file"] for a in all_artifacts if a.get("file")})
    console.print(
        f"\n  [green]✓[/green] [bold]{len(completed_ids)}/{len(task_cards)} tasks completed[/bold]"
        f"  [dim]·  files: {unique_files}  ·  {elapsed:.1f}s[/dim]\n"
    )


if __name__ == "__main__":
    init_db()
    issue = """
    The file multi_bug.py has multiple independent bugs across different functions.
    1. calculate_stats() crashes with empty input
    2. load_user_data() crashes when 'email' field is missing
    3. process_config() crashes when optional keys are absent
    """
    run_planner(issue, repo_files=["multi_bug.py"])