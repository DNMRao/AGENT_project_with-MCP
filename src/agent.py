"""Bounded operations agent: plan one action, run it, repeat, then validate the answer.

The planner here is a fixed set of rules, so every run is reproducible. A
language model could replace _plan() without changing the tools, budgets,
memory, or citation checks around it.
"""

import argparse
import re
from dataclasses import dataclass, field

from src.config import MAX_AGENT_STEPS, MAX_TOOL_CALLS, MEMORY_PATH
from src.data import AgentCase, Customer, QueueOwner, Ticket, load_cases, validate_all
from src.ingestion import build_chunks, load_documents
from src.memory import CaseMemory, ShortTermMemory, Turn
from src.retrieval import Retriever, is_unsafe
from src.tools import Draft, Evidence, ToolBudgetExceeded, ToolCall, ToolError, ToolRegistry

CITATION_PATTERN = re.compile(r"\[([a-z0-9-]+)\]")  # same pattern as Module 5 generation.py

# Why each queue receives a handoff; the wording follows escalation-rules.md and
# makes the second policy search find that file.
ESCALATION_REASONS = {
    "incident-response": "production incident response",
    "privacy": "privacy processing",
    "identity-review": "identity review",
}


def route(ticket: Ticket, customer: Customer) -> str | None:
    """Queue that must handle this ticket, or None if support may answer directly."""
    if ticket.category == "incident":
        return "incident-response"
    if ticket.category == "privacy":
        return "privacy"
    if ticket.category == "account" and not customer.verified:
        return "identity-review"
    return None


@dataclass
class AgentResult:
    case_id: str
    status: str  # answered, escalated, abstained, review_required, invalid_citations
    answer: str
    citations: list[str]
    memory_used: bool
    safety_events: list[str]
    trace: list[ToolCall]
    steps: int

    @property
    def tools_used(self) -> list[str]:
        return [call.name for call in self.trace]


@dataclass
class RunState:
    """Everything the planner knows so far in one run."""

    request: str
    ticket_id: str | None
    memory_used: bool
    remembered_customer: str | None
    ticket: Ticket | None = None
    customer: Customer | None = None
    evidence: list[Evidence] | None = None  # None = not searched yet; [] = searched, nothing found
    escalation_evidence: list[Evidence] | None = None
    owner: QueueOwner | None = None
    draft: Draft | None = None
    events: list[str] = field(default_factory=list)

    def all_evidence(self) -> list[Evidence]:
        return [*(self.evidence or []), *(self.escalation_evidence or [])]


@dataclass(frozen=True)
class Action:
    tool: str
    args: dict


@dataclass(frozen=True)
class Finish:
    status: str
    answer: str
    event: str | None = None


def sources(evidence: list[Evidence]) -> list[str]:
    """Unique source IDs in retrieval order."""
    return list(dict.fromkeys(e.source_id for e in evidence))


def first_sentence(text: str) -> str:
    return text.split(". ")[0].rstrip(".") + "."


def compose_update(ticket: Ticket, customer: Customer, evidence: list[Evidence]) -> str:
    best = {e.source_id: e for e in reversed(evidence)}  # keep the best-scoring chunk per source
    lines = [f"Draft update for {customer.name} on {ticket.ticket_id} ({ticket.subject})."]
    lines += [f"Next step per [{s}]: {best[s].text}" for s in sources(evidence)]
    lines.append("Draft only: a support colleague reviews it before anything is sent.")
    return "\n".join(lines)


def compose_escalation(state: RunState, queue: str) -> str:
    t, c, owner = state.ticket, state.customer, state.owner
    evidence = state.all_evidence()
    best = {e.source_id: e for e in reversed(evidence)}
    lines = [
        f"Escalate {t.ticket_id} to {queue} (owner: {owner.owner}, {owner.channel}).",
        f"Customer {c.customer_id}, {c.email}, plan {c.plan}, verified={c.verified}.",
        f"Reported issue ({t.category}): {t.subject}. Customer wrote: \"{t.body}\"",
        "Evidence that triggered the handoff:",
    ]
    lines += [f"- [{s}] {first_sentence(best[s].text)}" for s in sources(evidence)]
    return "\n".join(lines)


def _plan(state: RunState) -> Action | Finish:
    """Choose the next action from what is known so far. No side effects."""
    if state.ticket_id is None:
        return Finish("review_required", "No ticket in the request or in memory; request human review.", "no_ticket")
    if state.ticket is None:
        return Action("get_ticket", {"ticket_id": state.ticket_id})
    if state.memory_used and state.ticket.customer_id != state.remembered_customer:
        return Finish("review_required", "Memory does not match the ticket's customer; request human review.",
                      "memory_customer_mismatch")
    if state.customer is None:
        return Action("get_customer_context", {"customer_id": state.ticket.customer_id})
    if state.evidence is None:
        t = state.ticket
        return Action("search_policy", {"query": f"{state.request} {t.subject}. {t.body}"})
    if not state.evidence:
        return Finish("abstained", f"No approved Northstar policy covers this request on "
                                   f"{state.ticket.ticket_id}; route it to human review.")

    queue = route(state.ticket, state.customer)
    if queue is None:  # support may answer from the evidence
        if state.draft is None:
            return Action("draft_ticket_update", {
                "ticket_id": state.ticket.ticket_id, "customer_name": state.customer.name,
                "text": compose_update(state.ticket, state.customer, state.evidence),
                "citations": sources(state.evidence)})
        return Finish("answered", state.draft.text)

    # A human queue must take over: confirm an owner before drafting anything.
    if state.owner is None:
        return Action("check_availability", {"queue": queue})
    if not state.owner.available:
        cited = " ".join(f"[{s}]" for s in sources(state.evidence))
        return Finish("review_required", f"{queue} has no available owner ({state.owner.owner}); "
                                         f"request human review. Evidence: {cited}", "owner_unavailable")
    if state.escalation_evidence is None:
        return Action("search_policy", {"query": f"escalation rules for {ESCALATION_REASONS[queue]}: "
                                                 "responsible queue and handoff evidence"})
    if state.draft is None:
        return Action("draft_escalation", {
            "ticket_id": state.ticket.ticket_id, "queue": queue, "owner": state.owner.owner,
            "text": compose_escalation(state, queue), "citations": sources(state.all_evidence())})
    return Finish("escalated", state.draft.text)


class OperationsAgent:
    def __init__(self, retriever: Retriever | None = None, case_memory: CaseMemory | None = None,
                 max_steps: int = MAX_AGENT_STEPS, max_tool_calls: int = MAX_TOOL_CALLS,
                 memory_path=None):
        validate_all()  # refuse to run on inconsistent data
        self.retriever = retriever or Retriever(build_chunks(load_documents()))
        self.tools = ToolRegistry(self.retriever, max_tool_calls)
        self.case_memory = case_memory or CaseMemory.from_seed()
        self.short_term = ShortTermMemory()
        self.max_steps = max_steps
        self.memory_path = memory_path  # None = keep memory in RAM only

    def run_case(self, case: AgentCase) -> AgentResult:
        return self.run(case.prompt, case.ticket_id, case.case_id)

    def run(self, request: str, ticket_id: str | None = None, case_id: str = "ad-hoc") -> AgentResult:
        self.tools.reset_trace()
        state = self._start_state(request, ticket_id)
        steps = 0
        try:
            while steps < self.max_steps:
                steps += 1
                decision = _plan(state)
                if isinstance(decision, Finish):
                    return self._finish(case_id, state, decision, steps)
                self._execute(state, decision)
            outcome = Finish("review_required", "Step budget exhausted; request human review.", "step_budget_exhausted")
        except ToolBudgetExceeded:
            outcome = Finish("review_required", "Tool budget exhausted; request human review.",
                             "execution_budget_exhausted")
        except ToolError as error:
            outcome = Finish("review_required", f"A tool failed ({error}); request human review.",
                             f"tool_failed:{self.tools.trace[-1].name}")
        return self._finish(case_id, state, outcome, steps)

    def _start_state(self, request: str, ticket_id: str | None) -> RunState:
        """A request without a ticket ID is a follow-up: resolve it from memory."""
        if ticket_id is not None:
            return RunState(request, ticket_id, memory_used=False, remembered_customer=None)
        last = self.short_term.last()
        if last is not None:  # this session's previous turn wins over saved memory
            return RunState(request, last.ticket_id, memory_used=True, remembered_customer=last.customer_id)
        return RunState(request, self.case_memory.last_ticket_id, memory_used=True,
                        remembered_customer=self.case_memory.last_customer_id)

    def _execute(self, state: RunState, action: Action) -> None:
        """Run one tool and store its typed output in the state."""
        result = getattr(self.tools, action.tool)(**action.args)
        match action.tool:
            case "get_ticket":
                state.ticket = result
                if is_unsafe(f"{result.subject} {result.body}"):
                    state.events.append("unsafe_ticket_text")  # treated as data, never obeyed
            case "get_customer_context":
                state.customer = result
            case "search_policy":
                state.events += result.safety_events
                if state.evidence is None:
                    state.evidence = result.evidence
                else:
                    state.escalation_evidence = result.evidence
            case "check_availability":
                state.owner = result
            case "draft_ticket_update" | "draft_escalation":
                state.draft = result

    def _finish(self, case_id: str, state: RunState, outcome: Finish, steps: int) -> AgentResult:
        status, answer, events = outcome.status, outcome.answer, list(state.events)
        if outcome.event:
            events.append(outcome.event)
        # Citations: every [id] in the answer must come from evidence retrieved in this run.
        accepted = set(sources(state.all_evidence()))
        cited = list(dict.fromkeys(CITATION_PATTERN.findall(answer)))
        if invalid := [c for c in cited if c not in accepted]:
            status, events = "invalid_citations", [*events, f"invalid_citations:{','.join(invalid)}"]
        # Remember the ticket only if this run actually inspected it for the right customer.
        if state.customer is not None:
            self.case_memory.remember(state.ticket.ticket_id, state.customer.customer_id)
            self.short_term.add(Turn(state.request, state.ticket.ticket_id, state.customer.customer_id, status))
            if self.memory_path is not None:
                self.case_memory.save(self.memory_path)
        return AgentResult(case_id, status, answer, cited, state.memory_used, events,
                           list(self.tools.trace), steps)


def print_result(result: AgentResult) -> None:
    print(f"case:      {result.case_id}")
    print(f"status:    {result.status}   (steps {result.steps}, tool calls {len(result.trace)})")
    print(f"citations: {', '.join(result.citations) or '-'}   memory used: {result.memory_used}")
    print(f"safety:    {', '.join(result.safety_events) or '-'}")
    print("trace:")
    for i, call in enumerate(result.trace, 1):
        args = ", ".join(f"{k}={v}" for k, v in call.args.items() if k != "text")
        print(f"  {i}. {call.name}({args}){'' if call.ok else '  FAILED'}  {call.note}")
    print(f"answer:\n{result.answer}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Northstar operations agent.")
    sub = parser.add_subparsers(dest="command", required=True)
    case_cmd = sub.add_parser("case", help="run one evaluation case from seed memory")
    case_cmd.add_argument("case_id")
    ask_cmd = sub.add_parser("ask", help="free request; durable memory is saved between runs")
    ask_cmd.add_argument("request")
    ask_cmd.add_argument("--ticket", help="ticket ID; omit for a follow-up resolved from memory")
    args = parser.parse_args()

    if args.command == "case":
        cases = {c.case_id: c for c in load_cases()}
        print_result(OperationsAgent().run_case(cases[args.case_id]))
    else:
        agent = OperationsAgent(case_memory=CaseMemory.load(MEMORY_PATH), memory_path=MEMORY_PATH)
        print_result(agent.run(args.request, args.ticket))
