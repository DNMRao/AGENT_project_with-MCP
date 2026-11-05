"""Short-term conversation memory and durable case memory for follow-up turns.

Memory only answers "which ticket/customer does this follow-up refer to?".
It is never used as policy evidence, so it cannot override retrieved policy.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.data import SeedMemory, load_seed_memory


@dataclass(frozen=True)
class Turn:
    prompt: str
    ticket_id: str
    customer_id: str
    status: str


@dataclass
class ShortTermMemory:
    """Turns of the current conversation; lives in RAM for one session only."""

    max_turns: int = 10
    turns: list[Turn] = field(default_factory=list)

    def add(self, turn: Turn) -> None:
        self.turns = [*self.turns, turn][-self.max_turns:]  # keep only the most recent turns

    def last(self) -> Turn | None:
        return self.turns[-1] if self.turns else None


@dataclass
class CaseMemory:
    """Last ticket and customer of one conversation, saved to disk between runs."""

    conversation_id: str
    last_ticket_id: str | None = None
    last_customer_id: str | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_seed(cls, seed: SeedMemory | None = None) -> "CaseMemory":
        seed = seed or load_seed_memory()
        return cls(seed.conversation_id, seed.last_ticket_id, seed.last_customer_id, list(seed.notes))

    @classmethod
    def load(cls, path: Path) -> "CaseMemory":
        """Saved memory if it exists, otherwise the seeded fixture."""
        if path.exists():
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        return cls.from_seed()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def remember(self, ticket_id: str, customer_id: str, note: str | None = None) -> None:
        """Record the latest ticket; forget notes that belong to a different customer."""
        if customer_id != self.last_customer_id:
            self.notes = []  # prevents one customer's context leaking into another's
        self.last_ticket_id, self.last_customer_id = ticket_id, customer_id
        if note:
            self.notes.append(note)
