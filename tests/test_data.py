import json
import shutil
from pathlib import Path

from src.data import find_problems


def test_supplied_data_is_clean():
    assert find_problems() == []


def test_ticket_with_unknown_customer_is_reported(tmp_path):
    # tmp_path is a fresh empty folder that pytest creates for this test only,
    # so we can break a copy of the data without touching the real files.
    ops = tmp_path / "operations"
    shutil.copytree(Path("data/operations"), ops)

    tickets = json.loads((ops / "tickets.json").read_text())
    # TODO: change the customer_id of the first ticket to "C-999"
    tickets[0]["customer_id"] = "C-999"
    (ops / "tickets.json").write_text(json.dumps(tickets))

    problems = find_problems(ops_dir=ops)
    #assert problems is [], "customer_id is not matching"
    # TODO: assert that one of the problems mentions "C-999"
    assert any("C-999" in p for p in problems), f"C-999 not reported; got: {problems}"

