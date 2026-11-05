"""Load and validate the policy, operations, memory, and evaluation data before the agent runs."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, StrictBool

from src.ingestion import KB_DIR, load_documents

OPS_DIR = Path("data/operations")
EVAL_PATH = Path("data/evaluation/agent_cases.json")

TOOL_NAMES = {
    "get_ticket", "get_customer_context", "search_policy",
    "check_availability", "draft_ticket_update", "draft_escalation",
}
Status = Literal["answered", "escalated", "abstained", "review_required"]


class Ticket(BaseModel):
    ticket_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class Customer(BaseModel):
    customer_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    plan: str
    verified: StrictBool  # StrictBool rejects "yes" or 1; only true/false pass


class QueueOwner(BaseModel):
    queue: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    channel: str
    available: StrictBool


class SeedMemory(BaseModel):
    conversation_id: str = Field(min_length=1)
    last_ticket_id: str | None
    last_customer_id: str | None
    notes: list[str] = []


class AgentCase(BaseModel):
    case_id: str = Field(min_length=1)
    ticket_id: str | None  # None = a follow-up turn that must be resolved from memory
    prompt: str = Field(min_length=1)
    expected_status: Status
    expected_tools: list[str]
    expected_sources: list[str]
    must_use_memory: bool
    forbidden_tools: list[str] = []  # only some cases have these two keys
    expected_safety_events: list[str] = []


def _load_list(path: Path, model: type[BaseModel]) -> list:
    return [model.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def load_tickets(ops_dir: Path = OPS_DIR) -> list[Ticket]:
    return _load_list(ops_dir / "tickets.json", Ticket)


def load_customers(ops_dir: Path = OPS_DIR) -> list[Customer]:
    return _load_list(ops_dir / "customers.json", Customer)


def load_availability(ops_dir: Path = OPS_DIR) -> list[QueueOwner]:
    return _load_list(ops_dir / "availability.json", QueueOwner)


def load_seed_memory(ops_dir: Path = OPS_DIR) -> SeedMemory:
    return SeedMemory.model_validate_json((ops_dir / "seed_memory.json").read_text(encoding="utf-8"))


def load_cases(path: Path = EVAL_PATH) -> list[AgentCase]:
    return _load_list(path, AgentCase)


def duplicates(ids: list[str]) -> list[str]:
    return sorted({i for i in ids if ids.count(i) > 1})


def find_problems(ops_dir: Path = OPS_DIR, eval_path: Path = EVAL_PATH, kb_dir: Path = KB_DIR) -> list[str]:
    """Cross-file checks Pydantic can't do alone. Empty list = data is clean."""
    tickets = load_tickets(ops_dir)
    customers = load_customers(ops_dir)
    queues = load_availability(ops_dir)
    memory = load_seed_memory(ops_dir)
    cases = load_cases(eval_path)
    source_ids = {doc.document_id for doc in load_documents(kb_dir)}
    ticket_by_id = {t.ticket_id: t for t in tickets}
    customer_ids = {c.customer_id for c in customers}
    problems: list[str] = []

    for label, ids in [("ticket", [t.ticket_id for t in tickets]),
                       ("customer", [c.customer_id for c in customers]),
                       ("queue", [q.queue for q in queues]),
                       ("evaluation case", [c.case_id for c in cases])]:
        if dups := duplicates(ids):
            problems.append(f"duplicate {label} IDs: {dups}")

    # Operations: every ticket points to a real customer.
    for t in tickets:
        if t.customer_id not in customer_ids:
            problems.append(f"{t.ticket_id}: unknown customer '{t.customer_id}'")

    # Evaluation: valid ticket, source, and tool references; labels agree with each other.
    for case in cases:
        if case.ticket_id is not None and case.ticket_id not in ticket_by_id:
            problems.append(f"{case.case_id}: unknown ticket '{case.ticket_id}'")
        if case.ticket_id is None and not case.must_use_memory:
            problems.append(f"{case.case_id}: no ticket_id but must_use_memory is false")
        for source in set(case.expected_sources) - source_ids:
            problems.append(f"{case.case_id}: unknown source '{source}'")
        for tool in set(case.expected_tools + case.forbidden_tools) - TOOL_NAMES:
            problems.append(f"{case.case_id}: unknown tool '{tool}'")
        if overlap := set(case.expected_tools) & set(case.forbidden_tools):
            problems.append(f"{case.case_id}: tools both expected and forbidden: {sorted(overlap)}")
        if case.expected_status == "abstained" and case.expected_sources:
            problems.append(f"{case.case_id}: abstained case should expect no sources")

    # Memory: the seeded ticket belongs to the seeded customer ...
    if memory.last_ticket_id is not None:
        seeded = ticket_by_id.get(memory.last_ticket_id)
        if seeded is None:
            problems.append(f"seed memory: unknown ticket '{memory.last_ticket_id}'")
        elif seeded.customer_id != memory.last_customer_id:
            problems.append("seed memory: ticket and customer do not match")
    # ... and its notes mention no other customer's tickets, name, or email (no leakage).
    notes = " ".join(memory.notes)
    for ticket_id in re.findall(r"T-\d+", notes):
        owner = ticket_by_id.get(ticket_id)
        if owner is None or owner.customer_id != memory.last_customer_id:
            problems.append(f"seed memory notes mention another customer's ticket {ticket_id}")
    for c in customers:
        if c.customer_id != memory.last_customer_id and (c.name in notes or c.email in notes):
            problems.append(f"seed memory notes mention unrelated customer {c.customer_id}")

    return problems


def validate_all() -> None:
    """Stop before the agent runs if any data check fails."""
    if problems := find_problems():
        raise ValueError("Data validation failed:\n" + "\n".join(problems))


if __name__ == "__main__":
    print(f"{len(load_tickets())} tickets, {len(load_customers())} customers, "
          f"{len(load_availability())} queues, {len(load_cases())} evaluation cases, "
          f"{len(load_documents())} policies")
    problems = find_problems()
    print("\n".join(problems) if problems else "All checks passed.")
