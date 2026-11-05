"""Choose MIN_SCORE from data/calibration/ only (never from the evaluation cases)."""

import json
from pathlib import Path

from src.config import MIN_SCORE, TOP_K
from src.ingestion import build_chunks, load_documents
from src.retrieval import Retriever

CALIBRATION_PATH = Path("data/calibration/calibration.json")
RESULT_PATH = Path("artifacts/calibration.json")  # read by src/evaluation.py for the report
THRESHOLDS = [round(0.25 + 0.025 * i, 3) for i in range(15)]  # 0.25 ... 0.60


def load_cases(path: Path = CALIBRATION_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_threshold(ranked: list[tuple[dict, list[tuple[float, str]]]], t: float) -> dict:
    """Decision accuracy, exact source match, and hit precision at threshold t."""
    decisions = exact = good_hits = all_hits = 0
    for case, hits in ranked:
        accepted = [source for score, source in hits[:TOP_K] if score >= t]
        expected = set(case["expected_sources"])
        decisions += (not accepted) == case["should_abstain"]
        exact += set(accepted) == expected
        good_hits += sum(source in expected for source in accepted)
        all_hits += len(accepted)
    n = len(ranked)
    return {
        "threshold": t,
        "decision_accuracy": round(decisions / n, 3),
        "exact_source_match": round(exact / n, 3),
        "hit_precision": round(good_hits / all_hits, 3) if all_hits else None,
    }


def main() -> None:
    cases = load_cases()
    retriever = Retriever(build_chunks(load_documents()))
    # Retrieval is the same at every threshold, so search once and reuse the scores.
    ranked = [
        (c, [(h.score, h.chunk.document_id) for h in retriever.search(c["question"], k=TOP_K)])
        for c in cases
    ]

    print(f"{'id':7} {'expected':26} top-{TOP_K} hits")
    for case, hits in ranked:
        label = ",".join(case["expected_sources"]) or "(abstain)"
        print(f"{case['id']:7} {label:26} " + " | ".join(f"{s:.3f} {d}" for s, d in hits))

    answer_tops = [hits[0][0] for c, hits in ranked if not c["should_abstain"]]
    abstain_tops = [hits[0][0] for c, hits in ranked if c["should_abstain"]]
    print(f"\nLowest top score among answer cases:   {min(answer_tops):.3f}")
    print(f"Highest top score among abstain cases: {max(abstain_tops):.3f}")

    sweep = [score_threshold(ranked, t) for t in THRESHOLDS]
    print(f"\n{'MIN_SCORE':>9} {'decision':>9} {'exact src':>10} {'precision':>10}")
    for row in sweep:
        print(f"{row['threshold']:9.3f} {row['decision_accuracy']:9.3f} "
              f"{row['exact_source_match']:10.3f} {row['hit_precision'] or 0:10.3f}")

    # Best = highest decision accuracy, then exact match; report the whole tied plateau.
    best_key = max((r["decision_accuracy"], r["exact_source_match"]) for r in sweep)
    plateau = [r["threshold"] for r in sweep if (r["decision_accuracy"], r["exact_source_match"]) == best_key]
    print(f"\nBest plateau: {plateau[0]:.3f} to {plateau[-1]:.3f} "
          f"(decision {best_key[0]:.3f}, exact source match {best_key[1]:.3f})")
    frozen = score_threshold(ranked, MIN_SCORE)
    print(f"Frozen MIN_SCORE = {MIN_SCORE}: {frozen}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({
        "top_k": TOP_K,
        "frozen_min_score": MIN_SCORE,
        "lowest_answer_top": round(min(answer_tops), 3),
        "highest_abstain_top": round(max(abstain_tops), 3),
        "plateau": [plateau[0], plateau[-1]],
        "sweep": sweep,
        "cases": [{"id": c["id"], "expected": c["expected_sources"],
                   "hits": [[round(s, 3), d] for s, d in hits]} for c, hits in ranked],
    }, indent=2), encoding="utf-8")
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
