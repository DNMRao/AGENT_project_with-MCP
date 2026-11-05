import pytest

from src.tools import ToolBudgetExceeded, ToolError, ToolRegistry


def test_unknown_ticket_raises_tool_error_and_is_traced():
    tools = ToolRegistry(retriever=None, call_limit=6)
    with pytest.raises(ToolError):
        tools.get_ticket("T-9999")
    assert len(tools.trace) == 1
    assert not tools.trace[0].ok, "Unknown ticket id"
    # TODO: assert the trace has 1 entry, and that entry has ok == False


def test_budget_stops_the_next_call():
    tools = ToolRegistry(retriever=None, call_limit=2)
    tools.get_ticket("T-1001")
    tools.get_ticket("T-1001")
    with pytest.raises(ToolBudgetExceeded):
         tools.get_ticket("T-1001")
    # TODO: use pytest.raises to check that a THIRD call raises ToolBudgetExceeded


def test_escalation_to_unavailable_queue_is_refused():
    tools = ToolRegistry(retriever=None)
    #tools.draft_escalation("T-1001", "identity-review", "Nadine Keller")
    with pytest.raises(ToolError):
         tools.draft_escalation("T-1001", "identity-review", "Nadine Keller", "X", [])
         #tools.draft_escalation("T-1001", "privacy", "Greta Holm", "X", [])
    # TODO: look in data/operations/availability.json for a queue with "available": false,
    #       call draft_escalation for it, and check that it raises ToolError

