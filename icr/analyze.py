"""Summarise an ICR run: CR, PR, SI, SR and SCR per condition.

Metrics are computed over the revision records that exist for each condition:

    CR   receiver correct after revision | sender right, receiver wrong
    PR   receiver correct after revision | sender wrong, receiver right
    SI   (CR + PR) / 2
    SR   receiver correct after revision | both wrong
    SCR  receiver correct after revision | both right

SR and SCR are reported only where such records were generated (a
``run_icr_mixed.sh`` run has none). Intervals are 95% percentile intervals of a
paired item-cluster bootstrap: every resample draws items with replacement and
recomputes every condition on the same items, so differences are paired.

Usage:
    python -m icr.analyze --run LABEL=ROOT[,ROOT...] [--run ...] --out DIR

Several roots may be listed for one run when a supplementary condition was
written to its own root; they must share the first root's phase-1 beliefs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("none", "true_answer", "true_text", "true_statebridge", "true_latentmas")
LABELS = {
    "none": "No Message",
    "true_answer": "Answer Only",
    "true_text": "Full Text",
    "true_statebridge": "StateBridge",
    "true_latentmas": "LatentMAS",
}
STRATA = {
    "correction_opportunity": "CR",
    "destruction_risk": "PR",
    "both_wrong": "SR",
    "both_correct": "SCR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, help="LABEL=ROOT[,ROOT...]")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    # Split on "\n" only: str.splitlines() also breaks on Unicode separators
    # that can occur inside generated text.
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line]


def load_records(root: Path) -> list[dict]:
    merged = root / "revisions" / "merged.jsonl"
    if merged.is_file():
        return read_jsonl(merged)
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("revisions/*/records/item_*.json"))
    ]


def load_run(roots: list[Path]) -> dict[str, dict[tuple[int, str], dict]]:
    beliefs = read_jsonl(roots[0] / "prebeliefs" / "merged.jsonl")
    reference = {(r["item_id"], r["agent_id"]): r["correct"] for r in beliefs}
    by_condition: dict[str, dict[tuple[int, str], dict]] = {}
    for root in roots:
        other = read_jsonl(root / "prebeliefs" / "merged.jsonl")
        if {(r["item_id"], r["agent_id"]): r["correct"] for r in other} != reference:
            raise RuntimeError(f"{root} does not share the beliefs of {roots[0]}")
        for row in load_records(root):
            if row["condition"] not in CONDITIONS:
                continue
            key = (int(row["item_id"]), row["direction"])
            cell = by_condition.setdefault(row["condition"], {})
            if key in cell and cell[key]["receiver_post_correct"] != row["receiver_post_correct"]:
                raise RuntimeError(f"conflicting records for {row['condition']} {key}")
            cell[key] = row
    return by_condition


class Counts:
    """Per-item success/denominator arrays for every (condition, stratum)."""

    def __init__(self, by_condition: dict[str, dict[tuple[int, str], dict]]):
        items = sorted({item for cell in by_condition.values() for item, _ in cell})
        self.items = items
        index = {item: i for i, item in enumerate(items)}
        self.ok: dict[tuple[str, str], np.ndarray] = {}
        self.den: dict[tuple[str, str], np.ndarray] = {}
        for condition, cell in by_condition.items():
            for stratum in STRATA.values():
                self.ok[(condition, stratum)] = np.zeros(len(items))
                self.den[(condition, stratum)] = np.zeros(len(items))
            for (item, _), row in cell.items():
                stratum = STRATA[row["pair_classification"]]
                self.den[(condition, stratum)][index[item]] += 1
                self.ok[(condition, stratum)][index[item]] += bool(row["receiver_post_correct"])

    def rate(self, condition: str, stratum: str, idx=None):
        ok, den = self.ok[(condition, stratum)], self.den[(condition, stratum)]
        if idx is None:
            total = den.sum()
            return 100 * ok.sum() / total if total else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100 * ok[idx].sum(axis=-1) / den[idx].sum(axis=-1)

    def metrics(self, condition: str, idx=None) -> dict:
        cr, pr = self.rate(condition, "CR", idx), self.rate(condition, "PR", idx)
        return {
            "CR": cr, "PR": pr, "SI": (cr + pr) / 2,
            "SR": self.rate(condition, "SR", idx), "SCR": self.rate(condition, "SCR", idx),
        }


def interval(samples) -> str:
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return ""
    low, high = np.percentile(samples, [2.5, 97.5])
    return f"[{low:.2f}, {high:.2f}]"


def main() -> None:
    cli = parse_args()
    rng = np.random.default_rng(cli.bootstrap_seed)
    cli.out.mkdir(parents=True, exist_ok=True)
    summary, paired = [], []
    for spec in cli.run:
        label, roots = spec.split("=", 1)
        by_condition = load_run([Path(r) for r in roots.split(",")])
        counts = Counts(by_condition)
        boot = rng.integers(0, len(counts.items), size=(cli.num_bootstrap, len(counts.items)))
        present = [c for c in CONDITIONS if c in by_condition]
        point = {c: counts.metrics(c) for c in present}
        dist = {c: counts.metrics(c, boot) for c in present}
        for c in present:
            rows = by_condition[c].values()
            row = {
                "run": label, "condition": LABELS[c], "records": len(rows),
                "non_eos": sum(not r.get("hit_eos", True) for r in rows),
                "unparsed": sum(r.get("parsed_answer") is None for r in rows),
            }
            for m in ("CR", "PR", "SI", "SR", "SCR"):
                stratum = m if m != "SI" else "CR"
                row[f"n_{m}"] = int(counts.den[(c, stratum)].sum()) if m != "SI" else ""
                row[m] = "" if np.isnan(point[c][m]) else round(float(point[c][m]), 2)
                row[f"{m}_ci"] = interval(dist[c][m]) if row[m] != "" else ""
            summary.append(row)
            if c != "none" and "none" in present:
                for m in ("CR", "PR", "SI"):
                    delta = point[c][m] - point["none"][m]
                    paired.append({
                        "run": label, "condition": LABELS[c], "metric": f"delta {m} vs No Message",
                        "delta": round(float(delta), 2), "ci95": interval(dist[c][m] - dist["none"][m]),
                    })
    for name, rows in (("summary.csv", summary), ("paired_vs_no_message.csv", paired)):
        if rows:
            with open(cli.out / name, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    for r in summary:
        print(f"{r['run']:6s} {r['condition']:12s} n={r['records']:5d}  "
              f"CR={r['CR']} {r['CR_ci']}  PR={r['PR']} {r['PR_ci']}  SI={r['SI']} {r['SI_ci']}  "
              f"SR={r['SR']}  SCR={r['SCR']}  non-EOS={r['non_eos']}  unparsed={r['unparsed']}")
    for r in paired:
        print(f"{r['run']:6s} {r['condition']:12s} {r['metric']:24s} {r['delta']:+.2f} {r['ci95']}")


if __name__ == "__main__":
    main()
