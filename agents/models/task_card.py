# agents/models/task_card.py

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class TaskContext:
    relevant_files: List[str]        = field(default_factory=list)
    error_trace:    Optional[str]    = None
    issue_summary:  Optional[str]    = None


@dataclass
class TaskArtifact:
    type:        str            # "file_patch", "test_result", "error"
    file:        Optional[str]  = None
    description: Optional[str]  = None
    content:     Optional[str]  = None


@dataclass
class TaskCard:
    description:         str
    task_id:             str             = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_by:          str             = "PlannerAgent"
    assigned_to:         str             = "CoderAgent"
    status:              str             = "pending"   # pending→in_progress→completed→failed
    context:             TaskContext     = field(default_factory=TaskContext)
    acceptance_criteria: List[str]       = field(default_factory=list)
    depends_on:          List[str]       = field(default_factory=list)
    artifacts:           List[TaskArtifact] = field(default_factory=list)
    created_at:          str             = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "task_id":             self.task_id,
            "created_by":          self.created_by,
            "assigned_to":         self.assigned_to,
            "status":              self.status,
            "description":         self.description,
            "context": {
                "relevant_files":  self.context.relevant_files,
                "error_trace":     self.context.error_trace,
                "issue_summary":   self.context.issue_summary,
            },
            "acceptance_criteria": self.acceptance_criteria,
            "depends_on":          self.depends_on,
            "artifacts": [
                {
                    "type":        a.type,
                    "file":        a.file,
                    "description": a.description,
                    "content":     a.content,
                }
                for a in self.artifacts
            ],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskCard":
        ctx = data.get("context", {})
        return cls(
            task_id             = data.get("task_id", str(uuid.uuid4())[:8]),
            created_by          = data.get("created_by", "PlannerAgent"),
            assigned_to         = data.get("assigned_to", "CoderAgent"),
            status              = data.get("status", "pending"),
            description         = data["description"],
            context             = TaskContext(
                relevant_files  = ctx.get("relevant_files", []),
                error_trace     = ctx.get("error_trace"),
                issue_summary   = ctx.get("issue_summary"),
            ),
            acceptance_criteria = data.get("acceptance_criteria", []),
            depends_on          = data.get("depends_on", []),
            artifacts           = [
                TaskArtifact(
                    type        = a.get("type", "unknown"),
                    file        = a.get("file"),
                    description = a.get("description"),
                    content     = a.get("content"),
                )
                for a in data.get("artifacts", [])
            ],
            created_at          = data.get("created_at", datetime.utcnow().isoformat()),
        )