"""The audit log: every guard event as one line of JSON (no extra packages needed).

Each line is one window: its verdict (attack type, confidence, all
probabilities, suspicious class pairs), the state the policy saw, and the
action taken. Plain JSON lines can be read back by any tool, or with `read`.
"""

import json
from pathlib import Path


class AuditLog:
    """Writes events to `path` (JSON lines). With `path=None` it only keeps them in memory."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.events: list[dict] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")  # start a fresh log

    def log(self, event: dict) -> None:
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a") as file:
                file.write(json.dumps(event) + "\n")

    def close(self) -> None:
        pass  # every event is already written

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        """All events of an audit log file."""
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
