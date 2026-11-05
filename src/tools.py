"""Typed operation tools over local fixtures, with a trace and a call budget.

The JSON files stand in for real systems: helpdesk (tickets), CRM (customers),
on-call schedule (availability). Draft tools return text only; nothing is sent.
"""

from pydantic import BaseModel

from src.config import MAX_TOOL_CALLS
from src.data import Customer, QueueOwner, Ticket, load_availability, load_customers, load_tickets
from src.retrieval import Retriever


class ToolBudgetExceeded(RuntimeError):
    """The next call would go past MAX_TOOL_CALLS."""


class ToolError(RuntimeError):
    """A tool could not complete (unknown ID, rejected escalation, ...)."""


class ToolCall(BaseModel):
    """One trace entry: what was called, with which arguments, and how it ended."""

    name: str
    args: dict[str, str | list[str]]
    ok: bool = True
    note: str = ""


class Evidence(BaseModel):
    source_id: str
    chunk_id: str
    score: float
    text: str  # paragraph only, title removed


class PolicySearchResult(BaseModel):
    evidence: list[Evidence]
    safety_events: list[str]


class Draft(BaseModel):
    kind: str  # "ticket_update" or "escalation"
    ticket_id: str
    recipient: str  # customer name, or queue owner for escalations
    text: str
    citations: list[str]


class ToolRegistry:
    def __init__(self, retriever: Retriever, call_limit: int = MAX_TOOL_CALLS):
        self.retriever = retriever
        self.call_limit = call_limit
        self.tickets = {t.ticket_id: t for t in load_tickets()}
        self.customers = {c.customer_id: c for c in load_customers()}
        self.queues = {q.queue: q for q in load_availability()}
        self.trace: list[ToolCall] = []

    def reset_trace(self) -> None:
        self.trace = []

    def _record(self, name: str, **args) -> ToolCall:
        """Log the call before running it; refuse once the budget is spent."""
        if len(self.trace) >= self.call_limit:
            raise ToolBudgetExceeded(f"tool budget of {self.call_limit} calls exhausted before {name}")
        call = ToolCall(name=name, args=args)
        self.trace.append(call)
        return call

    @staticmethod
    def _fail(call: ToolCall, message: str) -> ToolError:
        call.ok, call.note = False, message
        return ToolError(message)

    def get_ticket(self, ticket_id: str) -> Ticket:
        call = self._record("get_ticket", ticket_id=ticket_id)
        if ticket_id not in self.tickets:
            raise self._fail(call, f"unknown ticket {ticket_id}")
        return self.tickets[ticket_id]

    def get_customer_context(self, customer_id: str) -> Customer:
        call = self._record("get_customer_context", customer_id=customer_id)
        if customer_id not in self.customers:
            raise self._fail(call, f"unknown customer {customer_id}")
        return self.customers[customer_id]

    def search_policy(self, query: str) -> PolicySearchResult:
        call = self._record("search_policy", query=query)
        hits, events = self.retriever.retrieve_evidence(query)
        call.note = ", ".join(f"{h.chunk.chunk_id}={h.score:.3f}" for h in hits) or "no evidence"
        return PolicySearchResult(
            evidence=[
                Evidence(
                    source_id=h.chunk.document_id,
                    chunk_id=h.chunk.chunk_id,
                    score=round(h.score, 3),
                    text=h.chunk.text.removeprefix(h.chunk.title).strip(),
                )
                for h in hits
            ],
            safety_events=events,
        )

    def check_availability(self, queue: str) -> QueueOwner:
        call = self._record("check_availability", queue=queue)
        owner = self.queues.get(queue)
        if owner is None:  # an unknown queue is treated as "nobody available"
            owner = QueueOwner(queue=queue, owner="unassigned", channel="", available=False)
        call.note = f"{owner.owner} available={owner.available}"
        return owner

    def draft_ticket_update(self, ticket_id: str, customer_name: str, text: str, citations: list[str]) -> Draft:
        self._record("draft_ticket_update", ticket_id=ticket_id, citations=citations)
        return Draft(kind="ticket_update", ticket_id=ticket_id, recipient=customer_name,
                     text=text, citations=citations)

    def draft_escalation(self, ticket_id: str, queue: str, owner: str, text: str, citations: list[str]) -> Draft:
        call = self._record("draft_escalation", ticket_id=ticket_id, queue=queue, owner=owner, citations=citations)
        # Re-check here as well: the tool must not trust its caller's view of availability.
        current = self.queues.get(queue)
        if current is None or not current.available:
            raise self._fail(call, f"no available owner for {queue}")
        if current.owner != owner:
            raise self._fail(call, f"{owner} is not the owner of {queue}")
        return Draft(kind="escalation", ticket_id=ticket_id, recipient=f"{queue} ({owner})",
                     text=text, citations=citations)
