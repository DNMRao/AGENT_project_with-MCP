from src.data import SeedMemory
from src.memory import CaseMemory, ShortTermMemory, Turn


def test_short_term_memory_keeps_only_recent_turns():
    memory = ShortTermMemory(max_turns=2)
    for i in range(3):
        memory.add(Turn(prompt=f"q{i}", ticket_id=f"T-{i}", customer_id="C-001", status="answered"))
    assert len(memory.turns) == 2, "oldest turn was not dropped"
    assert memory.last().ticket_id == "T-2"


def test_notes_are_cleared_when_customer_changes():
    memory = CaseMemory("demo", "T-1002", "C-002", ["Maya asked for MFA help."])
    memory.remember("T-1001", "C-001")
    assert memory.notes == [], "notes leaked to a different customer"
    assert (memory.last_ticket_id, memory.last_customer_id) == ("T-1001", "C-001")


def test_notes_are_kept_for_the_same_customer():
    memory = CaseMemory("demo", "T-1002", "C-002", ["Maya asked for MFA help."])
    memory.remember("T-1002", "C-002", note="Sent the checklist.")
    assert memory.notes == ["Maya asked for MFA help.", "Sent the checklist."]


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "memory.json"
    original = CaseMemory("demo", "T-1001", "C-001", ["note"])
    original.save(path)
    assert CaseMemory.load(path) == original


def test_from_seed_copies_notes():
    seed = SeedMemory(conversation_id="demo", last_ticket_id="T-1002", last_customer_id="C-002", notes=["a"])
    memory = CaseMemory.from_seed(seed)
    memory.remember("T-1002", "C-002", note="x")
    assert seed.notes == ["a"], "remember() changed the seed fixture (missing defensive copy)"
