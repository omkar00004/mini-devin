# agents/models/failure.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class FailureType(Enum):
    FORMAT_ERROR     = "format_error"      # malformed tool call — retry locally
    WRONG_APPROACH   = "wrong_approach"    # fix didn't work — escalate after 2
    SCOPE_TOO_NARROW = "scope_too_narrow"  # need more files — escalate immediately
    ENVIRONMENT      = "environment"       # missing dep, wrong python — escalate
    UNKNOWN          = "unknown"           # can't classify — escalate after 3


# Per-type policy: how many local retries before escalating
ESCALATION_POLICY = {
    FailureType.FORMAT_ERROR:     {"max_local_retries": 3, "escalate": False},
    FailureType.WRONG_APPROACH:   {"max_local_retries": 2, "escalate": True},
    FailureType.SCOPE_TOO_NARROW: {"max_local_retries": 0, "escalate": True},
    FailureType.ENVIRONMENT:      {"max_local_retries": 0, "escalate": True},
    FailureType.UNKNOWN:          {"max_local_retries": 3, "escalate": True},
}


@dataclass
class FailureReport:
    """
    What CoderAgent or DebuggerAgent sends back to PlannerAgent
    when a task cannot be completed.
    Contains everything PlannerAgent needs to replan intelligently.
    """
    task_id:          str
    failure_type:     FailureType
    attempts:         int
    last_error:       str                    # exact stderr or exception
    reflection:       str                    # what assumption was wrong
    files_seen:       List[str] = field(default_factory=list)
    suggested_scope:  List[str] = field(default_factory=list)  # files Planner should add
    should_escalate:  bool = True

    def to_dict(self) -> dict:
        return {
            "task_id":         self.task_id,
            "failure_type":    self.failure_type.value,
            "attempts":        self.attempts,
            "last_error":      self.last_error,
            "reflection":      self.reflection,
            "files_seen":      self.files_seen,
            "suggested_scope": self.suggested_scope,
            "should_escalate": self.should_escalate,
        }