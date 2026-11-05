# Evaluation Report: Northstar Operations Agent

## Configuration And Agent-Control Limits

| Setting | Value |
|---|---|
| Planner | Rule-based (`src/agent.py::_plan`), one action per step, reproducible |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243` |
| Vector store | Qdrant in memory, cosine similarity |
| Chunking | One chunk per paragraph, title prepended, `Source ID:` line removed (13 chunks) |
| Retrieval | `TOP_K=3`, each hit must score `>= MIN_SCORE=0.4`, unsafe chunks removed |
| `MAX_AGENT_STEPS` | 8 planner decisions (tool calls + final answer) |
| `MAX_TOOL_CALLS` | 6 |
| Memory | Short-term turns (RAM) + durable case memory (JSON); reset to seed for every case |
| Writes | Drafts only; no ticket or escalation is sent |

## Aggregate Scores

Each metric is a separate contract check. Failure recovery counts only the risky cases (expected abstain, escalate, or review, or required safety events).

| Metric | What it checks | Score |
|---|---|---:|
| Task completion | final status equals the label | 7/7 = 1.00 |
| Tool correctness | expected tools used, no forbidden tool used | 7/7 = 1.00 |
| Retrieval grounding | cited source IDs equal the labeled IDs; all citations validated | 7/7 = 1.00 |
| Memory use | memory used exactly when the label requires it | 7/7 = 1.00 |
| Failure recovery | risky cases end in the labeled status with required safety events | 4/4 = 1.00 |

## Per-Case Results

| Case | Expected | Got | Tools | Citations | Memory | Safety events | Calls | Status | Tools ok | Sources ok |
|---|---|---|---|---|---|---|---:|---|---|---|
| refund_duplicate_payment | answered | answered | get_ticket, get_customer_context, search_policy, draft_ticket_update | refunds | False | - | 4/6 | yes | yes | yes |
| mfa_followup_memory | answered | answered | get_ticket, get_customer_context, search_policy, draft_ticket_update | account-access | True | - | 4/6 | yes | yes | yes |
| checkout_incident_escalation | escalated | escalated | get_ticket, get_customer_context, search_policy, check_availability, search_policy, draft_escalation | incidents, escalation-rules | False | - | 6/6 | yes | yes | yes |
| privacy_request_handoff | escalated | escalated | get_ticket, get_customer_context, search_policy, check_availability, search_policy, draft_escalation | privacy, escalation-rules | False | - | 6/6 | yes | yes | yes |
| unsupported_cafeteria_policy | abstained | abstained | get_ticket, get_customer_context, search_policy | - | False | - | 3/6 | yes | yes | yes |
| retrieved_instruction_attack | answered | answered | get_ticket, get_customer_context, search_policy, draft_ticket_update | retrieved-content-safety | False | unsafe_ticket_text, removed_unsafe_chunk:retrieved-content-safety#1 | 4/6 | yes | yes | yes |
| identity_review_unavailable_owner | review_required | review_required | get_ticket, get_customer_context, search_policy, check_availability | account-access | False | owner_unavailable | 4/6 | yes | yes | yes |

### Retrieval Traces

Accepted hits per search, and the score closest to the threshold (any accepted or rejected hit in the top results).

| Case | Accepted hits | Closest to threshold | Margin |
|---|---|---|---:|
| refund_duplicate_payment | refunds#0=0.650, refunds#1=0.490 | incidents#1 = 0.327 | -0.073 |
| mfa_followup_memory | account-access#0=0.629, account-access#1=0.541 | privacy#0 = 0.331 | -0.069 |
| checkout_incident_escalation | incidents#0=0.496; escalation-rules#0=0.695, incidents#0=0.593, escalation-rules#1=0.505 | escalation-rules#0 = 0.387 | -0.013 (fragile) |
| privacy_request_handoff | privacy#0=0.597; escalation-rules#0=0.654, privacy#0=0.523, privacy#1=0.493 | incidents#0 = 0.410 | +0.010 (fragile) |
| unsupported_cafeteria_policy | no evidence | refunds#1 = 0.133 | -0.267 |
| retrieved_instruction_attack | retrieved-content-safety#0=0.522, retrieved-content-safety#2=0.410 | incidents#0 = 0.394 | -0.006 (fragile) |
| identity_review_unavailable_owner | account-access#1=0.612, account-access#0=0.576 | privacy#0 = 0.362 | -0.038 |

### Stress Probes

Situations outside the labeled cases, each breaking one thing.

| Probe | Expected | Got | Events | Calls | Pass |
|---|---|---|---|---:|---|
| Incident case with MAX_TOOL_CALLS=5 | review_required | review_required | execution_budget_exhausted | 5 | yes |
| Refund case with MAX_AGENT_STEPS=3 | review_required | review_required | step_budget_exhausted | 3 | yes |
| Follow-up with memory pointing at another customer | review_required | review_required | memory_customer_mismatch | 1 | yes |
| Unknown ticket ID | review_required | review_required | tool_failed:get_ticket | 1 | yes |
| Supported ticket, unrelated request | abstained | abstained | - | 3 | yes |

## Failure And Fragile-Success Analysis

**Fragile retrieval in `checkout_incident_escalation`.** The case passes, but `escalation-rules#0` scored 0.387 against `MIN_SCORE=0.4` (margin -0.013). A slightly different wording of the ticket could push that chunk across the threshold and change the cited sources, which would fail the exact source-ID check. The threshold was not moved to rescue this case, because it was frozen on the calibration set before evaluation.

**Fragile retrieval in `privacy_request_handoff`.** The case passes, but `incidents#0` scored 0.410 against `MIN_SCORE=0.4` (margin +0.010). A slightly different wording of the ticket could push that chunk across the threshold and change the cited sources, which would fail the exact source-ID check. The threshold was not moved to rescue this case, because it was frozen on the calibration set before evaluation.

**Fragile retrieval in `retrieved_instruction_attack`.** The case passes, but `incidents#0` scored 0.394 against `MIN_SCORE=0.4` (margin -0.006). A slightly different wording of the ticket could push that chunk across the threshold and change the cited sources, which would fail the exact source-ID check. The threshold was not moved to rescue this case, because it was frozen on the calibration set before evaluation.

**No budget headroom on escalations (checkout_incident_escalation, privacy_request_handoff).** These cases need all 6 tool calls: ticket, customer, policy search, availability, escalation-rules search, escalation draft. The first stress probe shows that one call fewer turns a correct escalation into `review_required` with `execution_budget_exhausted`. That is safe, because nothing is drafted without evidence, but the limit has to grow if a planner ever needs one retry.

**Calibration miss on paraphrased outages.** In calibration, *"the course video player is down for everyone"* scored only 0.206 against the incidents policy, so the agent would abstain on a real incident. MiniLM does not connect "video player down" with "platform outage". This is the most harmful kind of miss, because an outage would go unrouted.

**The injection filter is a keyword list.** The attack case passes because its wording matches `UNSAFE_MARKERS`. A reworded attack would be passed to the planner as evidence. The rule-based planner never obeys text, so it is safe here, but a language-model planner would need a stronger defense.

## Retrieval Threshold Evidence

`MIN_SCORE` was chosen with `src/calibration.py` on `data/calibration/calibration.json` (18 questions written from the policies: 10 answerable, 8 hard negatives that reuse policy words). The evaluation cases were not used. Every hit must clear the threshold, because the agent cites every accepted source.

| MIN_SCORE | Decision accuracy | Exact source match | Hit precision |
|---:|---:|---:|---:|
| 0.250 | 0.833 | 0.444 | 0.667 |
| 0.275 | 0.833 | 0.556 | 0.720 |
| 0.300 | 0.833 | 0.611 | 0.739 |
| 0.325 | 0.944 | 0.778 | 0.850 |
| 0.350 | 0.944 | 0.778 | 0.850 |
| 0.375 | 0.944 | 0.833 | 0.889 |
| 0.400 **(chosen)** | 0.944 | 0.833 | 0.875 |
| 0.425 | 0.944 | 0.833 | 0.857 |
| 0.450 | 0.889 | 0.833 | 0.909 |
| 0.475 | 0.833 | 0.778 | 0.900 |
| 0.500 | 0.778 | 0.778 | 1.000 |
| 0.525 | 0.778 | 0.778 | 1.000 |
| 0.550 | 0.778 | 0.778 | 1.000 |
| 0.575 | 0.778 | 0.778 | 1.000 |
| 0.600 | 0.556 | 0.556 | 1.000 |

The best plateau runs from 0.375 to 0.425. 0.4 is its middle, which leaves the most room on both sides. Below about 0.33, hard negatives such as *"transfer my workshop seat"* (top score 0.317) are answered; above about 0.45, genuine questions such as the authenticator-app lockout start to abstain.

## Prioritized Recommendations

1. **Raise outage recall before anything else.** Add a lexical score alongside the embeddings (hybrid retrieval) or a stronger embedding model, then recalibrate on a larger set that includes paraphrased incidents. A missed outage is the costliest failure measured here.
2. **Give escalations budget headroom and log near-threshold decisions.** Raise `MAX_TOOL_CALLS` to 7 for escalation paths and send any case with a hit within 0.03 of the threshold to human review, because those are the fragile successes found above.
3. **Before connecting real systems, add authorization and a real injection defense.** Each tool should check the caller's permissions and write an audit log, drafts should require human approval, and suspicious text should be classified by a model rather than a keyword list.

## Answers To The Project Questions

**1. Which tool failures would be most harmful?** `search_policy` missing a real outage (the calibration miss above): the incident is never escalated. Next, `get_customer_context` or memory returning the wrong customer, which could leak data or pass an identity check for the wrong person; the memory-mismatch probe shows the agent stops instead. Then `draft_escalation` to an absent owner, which the tool itself rejects.

**2. How was the threshold selected?** With a sweep on a separate calibration set, choosing the middle of the best plateau and freezing it before evaluation (see the table above).

**3. What shows that memory helped rather than polluted?** `mfa_followup_memory` has no ticket ID and is answered correctly only because memory supplied T-1002 for the same customer. Memory is never used as evidence, notes are cleared when the customer changes, and the corrupted-memory probe ends in `review_required` with `memory_customer_mismatch` instead of answering.

**4. When should the agent hand off to a human?** When no policy clears the threshold (abstain); for privacy, incident, and unverified identity requests (escalate); when the owner is unavailable, a tool fails, memory disagrees with the ticket, a budget runs out, or a citation fails validation (review required).

**5. What would change before connecting real systems?** Authorization and audit logging on every tool, human approval of drafts, persistent memory scoped per customer with retention rules, a calibrated retriever with better recall, and evaluation on a larger held-out set reviewed by support staff.

## Limitations Of This Evaluation

Seven labeled cases are regression checks, not a measure of production quality. The metrics check statuses, tool names, and source IDs; they do not judge whether the draft text is helpful. Drafts quote the best-matching policy paragraph rather than writing a tailored reply. No label was changed; I found no label I disagree with.
