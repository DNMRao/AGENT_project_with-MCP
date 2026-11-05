"""Run every labeled case, score it, add stress probes, and write the evaluation report."""

import json
from dataclasses import asdict
from pathlib import Path

from src.agent import AgentResult, OperationsAgent
from src.calibration import RESULT_PATH as CALIBRATION_RESULT
from src.config import EMBEDDING_MODEL, EMBEDDING_REVISION, MAX_AGENT_STEPS, MAX_TOOL_CALLS, MIN_SCORE, TOP_K
from src.data import AgentCase, load_cases
from src.memory import CaseMemory

REPORT_PATH = Path("reports/evaluation_report.md")
TRACE_PATH = Path("artifacts/evaluation_traces.json")
FRAGILE_MARGIN = 0.03  # a hit this close to MIN_SCORE could flip with small wording changes


def fresh(agent: OperationsAgent, memory: CaseMemory | None = None) -> OperationsAgent:
    """Cases are independent: reset both memories so no case sees another's turns."""
    agent.case_memory = memory or CaseMemory.from_seed()
    agent.short_term.turns = []
    return agent


def nearest_to_threshold(agent: OperationsAgent, result: AgentResult) -> tuple[float, str] | None:
    """Score closest to MIN_SCORE among the top hits of every search in this run."""
    scored = [
        (hit.score, hit.chunk.chunk_id)
        for call in result.trace if call.name == "search_policy"
        for hit in agent.retriever.search(str(call.args["query"]), k=TOP_K + 1)
    ]
    return min(scored, key=lambda s: abs(s[0] - MIN_SCORE)) if scored else None


def score_case(case: AgentCase, result: AgentResult) -> dict:
    """Five separate checks, so a tool error is not hidden by a correct status."""
    used = set(result.tools_used)
    risky = case.expected_status != "answered" or bool(case.expected_safety_events)
    return {
        "task_completion": result.status == case.expected_status,
        "tool_correctness": set(case.expected_tools) <= used and not (set(case.forbidden_tools) & used),
        "retrieval_grounding": set(result.citations) == set(case.expected_sources)
                               and result.status != "invalid_citations",
        "memory_use": result.memory_used == case.must_use_memory,
        # Only risky cases (abstain, escalate, review, or required safety events) count here.
        "failure_recovery": (result.status == case.expected_status
                             and set(case.expected_safety_events) <= set(result.safety_events)) if risky else None,
    }


def stress_probes(agent: OperationsAgent, cases: dict[str, AgentCase]) -> list[dict]:
    """Break one thing at a time and check that the agent stops visibly."""
    probes = []

    def probe(name: str, expected: str, run) -> None:
        result = run()
        probes.append({"probe": name, "expected": expected, "status": result.status,
                       "events": result.safety_events, "tool_calls": len(result.trace),
                       "passed": result.status == expected})

    tight = OperationsAgent(agent.retriever, max_tool_calls=MAX_TOOL_CALLS - 1)
    probe(f"Incident case with MAX_TOOL_CALLS={MAX_TOOL_CALLS - 1}", "review_required",
          lambda: fresh(tight).run_case(cases["checkout_incident_escalation"]))
    short = OperationsAgent(agent.retriever, max_steps=3)
    probe("Refund case with MAX_AGENT_STEPS=3", "review_required",
          lambda: fresh(short).run_case(cases["refund_duplicate_payment"]))
    corrupted = CaseMemory("probe", last_ticket_id="T-1002", last_customer_id="C-001")
    probe("Follow-up with memory pointing at another customer", "review_required",
          lambda: fresh(agent, corrupted).run("Use the same customer and draft the checklist."))
    probe("Unknown ticket ID", "review_required",
          lambda: fresh(agent).run("Resolve this ticket.", "T-9999"))
    probe("Supported ticket, unrelated request", "abstained",
          lambda: fresh(agent).run("What is the office Wi-Fi password?", "T-1005"))
    return probes


def run_evaluation() -> dict:
    agent = OperationsAgent()
    cases = load_cases()
    rows = []
    for case in cases:
        result = fresh(agent).run_case(case)
        near = nearest_to_threshold(agent, result)
        rows.append({"case": case, "result": result, "scores": score_case(case, result),
                     "nearest": near, "headroom": MAX_TOOL_CALLS - len(result.trace)})

    def ratio(metric: str) -> tuple[int, int]:
        values = [r["scores"][metric] for r in rows if r["scores"][metric] is not None]
        return sum(values), len(values)

    metrics = ["task_completion", "tool_correctness", "retrieval_grounding", "memory_use", "failure_recovery"]
    return {"rows": rows, "aggregate": {m: ratio(m) for m in metrics},
            "probes": stress_probes(agent, {c.case_id: c for c in cases})}


def render_report(results: dict, calibration: dict) -> str:
    rows, agg, probes = results["rows"], results["aggregate"], results["probes"]
    yes = lambda v: "-" if v is None else ("yes" if v else "**no**")
    lines = [
        "# Evaluation Report: Northstar Operations Agent", "",
        "## Configuration And Agent-Control Limits", "",
        "| Setting | Value |", "|---|---|",
        f"| Planner | Rule-based (`src/agent.py::_plan`), one action per step, reproducible |",
        f"| Embedding model | `{EMBEDDING_MODEL}` @ `{EMBEDDING_REVISION[:8]}` |",
        "| Vector store | Qdrant in memory, cosine similarity |",
        "| Chunking | One chunk per paragraph, title prepended, `Source ID:` line removed (13 chunks) |",
        f"| Retrieval | `TOP_K={TOP_K}`, each hit must score `>= MIN_SCORE={MIN_SCORE}`, unsafe chunks removed |",
        f"| `MAX_AGENT_STEPS` | {MAX_AGENT_STEPS} planner decisions (tool calls + final answer) |",
        f"| `MAX_TOOL_CALLS` | {MAX_TOOL_CALLS} |",
        "| Memory | Short-term turns (RAM) + durable case memory (JSON); reset to seed for every case |",
        "| Writes | Drafts only; no ticket or escalation is sent |", "",
        "## Aggregate Scores", "",
        "Each metric is a separate contract check. Failure recovery counts only the risky cases "
        "(expected abstain, escalate, or review, or required safety events).", "",
        "| Metric | What it checks | Score |", "|---|---|---:|",
    ]
    explain = {
        "task_completion": "final status equals the label",
        "tool_correctness": "expected tools used, no forbidden tool used",
        "retrieval_grounding": "cited source IDs equal the labeled IDs; all citations validated",
        "memory_use": "memory used exactly when the label requires it",
        "failure_recovery": "risky cases end in the labeled status with required safety events",
    }
    for metric, (ok, n) in agg.items():
        lines.append(f"| {metric.replace('_', ' ').capitalize()} | {explain[metric]} | {ok}/{n} = {ok / n:.2f} |")

    lines += ["", "## Per-Case Results", "",
              "| Case | Expected | Got | Tools | Citations | Memory | Safety events | Calls | Status | Tools ok | Sources ok |",
              "|---|---|---|---|---|---|---|---:|---|---|---|"]
    for r in rows:
        case, res, s = r["case"], r["result"], r["scores"]
        lines.append(
            f"| {case.case_id} | {case.expected_status} | {res.status} | {', '.join(res.tools_used)} | "
            f"{', '.join(res.citations) or '-'} | {res.memory_used} | {', '.join(res.safety_events) or '-'} | "
            f"{len(res.trace)}/{MAX_TOOL_CALLS} | {yes(s['task_completion'])} | {yes(s['tool_correctness'])} | "
            f"{yes(s['retrieval_grounding'])} |")

    lines += ["", "### Retrieval Traces", "", "Accepted hits per search, and the score closest to the threshold "
              "(any accepted or rejected hit in the top results).", "",
              "| Case | Accepted hits | Closest to threshold | Margin |", "|---|---|---|---:|"]
    for r in rows:
        notes = "; ".join(c.note for c in r["result"].trace if c.name == "search_policy")
        score, chunk = r["nearest"]
        flag = " (fragile)" if abs(score - MIN_SCORE) < FRAGILE_MARGIN else ""
        lines.append(f"| {r['case'].case_id} | {notes} | {chunk} = {score:.3f} | {score - MIN_SCORE:+.3f}{flag} |")

    lines += ["", "### Stress Probes", "", "Situations outside the labeled cases, each breaking one thing.", "",
              "| Probe | Expected | Got | Events | Calls | Pass |", "|---|---|---|---|---:|---|"]
    for p in probes:
        lines.append(f"| {p['probe']} | {p['expected']} | {p['status']} | {', '.join(p['events']) or '-'} | "
                     f"{p['tool_calls']} | {yes(p['passed'])} |")

    fragile = [r for r in rows if abs(r["nearest"][0] - MIN_SCORE) < FRAGILE_MARGIN]
    full_budget = [r["case"].case_id for r in rows if r["headroom"] == 0]
    lines += ["", "## Failure And Fragile-Success Analysis", ""]
    lines += [f"**Fragile retrieval in `{r['case'].case_id}`.** The case passes, but `{r['nearest'][1]}` scored "
              f"{r['nearest'][0]:.3f} against `MIN_SCORE={MIN_SCORE}` (margin {r['nearest'][0] - MIN_SCORE:+.3f}). "
              f"A slightly different wording of the ticket could push that chunk across the threshold and change "
              f"the cited sources, which would fail the exact source-ID check. The threshold was not moved to "
              f"rescue this case, because it was frozen on the calibration set before evaluation.\n"
              for r in fragile]
    lines += [
        f"**No budget headroom on escalations ({', '.join(full_budget)}).** These cases need all "
        f"{MAX_TOOL_CALLS} tool calls: ticket, customer, policy search, availability, escalation-rules search, "
        "escalation draft. The first stress probe shows that one call fewer turns a correct escalation into "
        "`review_required` with `execution_budget_exhausted`. That is safe, because nothing is drafted without "
        "evidence, but the limit has to grow if a planner ever needs one retry.", "",
        "**Calibration miss on paraphrased outages.** In calibration, *\"the course video player is down for "
        "everyone\"* scored only "
        f"{next(c['hits'][0][0] for c in calibration['cases'] if c['id'] == 'cal-05'):.3f} against the incidents "
        "policy, so the agent would abstain on a real incident. MiniLM does not connect \"video player down\" with "
        "\"platform outage\". This is the most harmful kind of miss, because an outage would go unrouted.", "",
        "**The injection filter is a keyword list.** The attack case passes because its wording matches "
        "`UNSAFE_MARKERS`. A reworded attack would be passed to the planner as evidence. The rule-based planner never "
        "obeys text, so it is safe here, but a language-model planner would need a stronger defense.", "",
        "## Retrieval Threshold Evidence", "",
        f"`MIN_SCORE` was chosen with `src/calibration.py` on `data/calibration/calibration.json` "
        f"({len(calibration['cases'])} questions written from the policies: 10 answerable, 8 hard negatives that "
        "reuse policy words). The evaluation cases were not used. Every hit must clear the threshold, because the "
        "agent cites every accepted source.", "",
        "| MIN_SCORE | Decision accuracy | Exact source match | Hit precision |", "|---:|---:|---:|---:|",
    ]
    for s in calibration["sweep"]:
        mark = " **(chosen)**" if abs(s["threshold"] - MIN_SCORE) < 1e-9 else ""
        lines.append(f"| {s['threshold']:.3f}{mark} | {s['decision_accuracy']:.3f} | "
                     f"{s['exact_source_match']:.3f} | {s['hit_precision'] or 0:.3f} |")
    lo, hi = calibration["plateau"]
    lines += [
        "", f"The best plateau runs from {lo:.3f} to {hi:.3f}. {MIN_SCORE} is its middle, which leaves the most room "
        "on both sides. Below about 0.33, hard negatives such as *\"transfer my workshop seat\"* (top score "
        f"{next(c['hits'][0][0] for c in calibration['cases'] if c['id'] == 'cal-12'):.3f}) are answered; above about "
        "0.45, genuine questions such as the authenticator-app lockout start to abstain.", "",
        "## Prioritized Recommendations", "",
        "1. **Raise outage recall before anything else.** Add a lexical score alongside the embeddings (hybrid "
        "retrieval) or a stronger embedding model, then recalibrate on a larger set that includes paraphrased "
        "incidents. A missed outage is the costliest failure measured here.",
        "2. **Give escalations budget headroom and log near-threshold decisions.** Raise `MAX_TOOL_CALLS` to 7 for "
        "escalation paths and send any case with a hit within 0.03 of the threshold to human review, because those "
        "are the fragile successes found above.",
        "3. **Before connecting real systems, add authorization and a real injection defense.** Each tool should check "
        "the caller's permissions and write an audit log, drafts should require human approval, and suspicious text "
        "should be classified by a model rather than a keyword list.", "",
        "## Answers To The Project Questions", "",
        "**1. Which tool failures would be most harmful?** `search_policy` missing a real outage (the calibration miss "
        "above): the incident is never escalated. Next, `get_customer_context` or memory returning the wrong customer, "
        "which could leak data or pass an identity check for the wrong person; the memory-mismatch probe shows the "
        "agent stops instead. Then `draft_escalation` to an absent owner, which the tool itself rejects.", "",
        "**2. How was the threshold selected?** With a sweep on a separate calibration set, choosing the middle of the "
        "best plateau and freezing it before evaluation (see the table above).", "",
        "**3. What shows that memory helped rather than polluted?** `mfa_followup_memory` has no ticket ID and is "
        "answered correctly only because memory supplied T-1002 for the same customer. Memory is never used as "
        "evidence, notes are cleared when the customer changes, and the corrupted-memory probe ends in "
        "`review_required` with `memory_customer_mismatch` instead of answering.", "",
        "**4. When should the agent hand off to a human?** When no policy clears the threshold (abstain); for privacy, "
        "incident, and unverified identity requests (escalate); when the owner is unavailable, a tool fails, memory "
        "disagrees with the ticket, a budget runs out, or a citation fails validation (review required).", "",
        "**5. What would change before connecting real systems?** Authorization and audit logging on every tool, human "
        "approval of drafts, persistent memory scoped per customer with retention rules, a calibrated retriever with "
        "better recall, and evaluation on a larger held-out set reviewed by support staff.", "",
        "## Limitations Of This Evaluation", "",
        "Seven labeled cases are regression checks, not a measure of production quality. The metrics check statuses, "
        "tool names, and source IDs; they do not judge whether the draft text is helpful. Drafts quote the "
        "best-matching policy paragraph rather than writing a tailored reply. No label was changed; I found no label "
        "I disagree with.", "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if not CALIBRATION_RESULT.exists():
        raise SystemExit(f"Missing {CALIBRATION_RESULT}; run: uv run python -m src.calibration")
    calibration = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    results = run_evaluation()

    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps([
        {"case_id": r["case"].case_id, "scores": r["scores"],
         "result": {**asdict(r["result"]), "trace": [c.model_dump() for c in r["result"].trace]}}
        for r in results["rows"]
    ], indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(results, calibration), encoding="utf-8")

    for metric, (ok, n) in results["aggregate"].items():
        print(f"{metric:20} {ok}/{n}")
    for p in results["probes"]:
        print(f"probe: {p['probe']:52} {p['status']:16} {'PASS' if p['passed'] else 'FAIL'}")
    print(f"Report: {REPORT_PATH}\nTraces: {TRACE_PATH}")
