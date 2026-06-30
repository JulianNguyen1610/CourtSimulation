"""Head-to-head comparison: fixed vs judge_mediated orchestrator predictions."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.error_analysis import classify_error


def load(path: Path) -> dict[str, dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {r["case_id"]: r for r in rows}


def main() -> None:
    root = Path("docs/experiments/orchestrator_ablation/20260629T123354Z")
    fixed = load(root / "orchestrator_fixed" / "predictions.csv")
    mediated = load(root / "orchestrator_judge_mediated" / "predictions.csv")

    ids = sorted(set(fixed) & set(mediated))
    assert len(ids) == 53

    def em(row: dict) -> bool:
        return float(row["exact_match"]) >= 1.0

    fixed_hits = sum(1 for i in ids if em(fixed[i]))
    med_hits = sum(1 for i in ids if em(mediated[i]))

    mediated_wins = [i for i in ids if em(mediated[i]) and not em(fixed[i])]
    fixed_wins = [i for i in ids if em(fixed[i]) and not em(mediated[i])]
    both_wrong = [i for i in ids if not em(fixed[i]) and not em(mediated[i])]
    both_right = [i for i in ids if em(fixed[i]) and em(mediated[i])]

    err_types_fixed = Counter()
    err_types_med = Counter()
    for i in ids:
        if not em(fixed[i]):
            err_types_fixed[classify_error(pd.Series(fixed[i]))] += 1
        if not em(mediated[i]):
            err_types_med[classify_error(pd.Series(mediated[i]))] += 1

    out = root / "error_analysis_head_to_head.md"
    lines = [
        "# Orchestrator Ablation — Head-to-Head Error Analysis",
        "",
        "Run: `20260629T123354Z` | Split: validation 53",
        "",
        "## Overall",
        "",
        "| Orchestrator | EM | Hits |",
        "|---|---:|---:|",
        f"| fixed | {fixed_hits/53:.4f} | {fixed_hits}/53 |",
        f"| judge_mediated | {med_hits/53:.4f} | {med_hits}/53 |",
        f"| **Δ EM** | **{(med_hits-fixed_hits)/53:+.4f}** | **{med_hits-fixed_hits:+d} cases** |",
        "",
        "## Head-to-head",
        "",
        f"- Both correct: {len(both_right)}",
        f"- Both wrong: {len(both_wrong)}",
        f"- **Judge-mediated fixes fixed's errors**: {len(mediated_wins)}",
        f"- **Judge-mediated regresses fixed's hits**: {len(fixed_wins)}",
        "",
        "### Judge-mediated wins (fixed wrong → mediated correct)",
        "",
    ]
    for cid in mediated_wins:
        lines.append(f"- **{cid}**")
        lines.append(f"  - gold: `{fixed[cid]['gold_answer'][:120].replace(chr(10), ' ')}`")
        lines.append(f"  - fixed: `{fixed[cid]['predicted_answer'][:120].replace(chr(10), ' ')}`")
        lines.append(f"  - mediated: `{mediated[cid]['predicted_answer'][:120].replace(chr(10), ' ')}`")

    lines += ["", "### Judge-mediated regressions (fixed correct → mediated wrong)", ""]
    for cid in fixed_wins:
        lines.append(f"- **{cid}**")
        lines.append(f"  - gold: `{fixed[cid]['gold_answer'][:120].replace(chr(10), ' ')}`")
        lines.append(f"  - fixed: `{fixed[cid]['predicted_answer'][:120].replace(chr(10), ' ')}`")
        lines.append(f"  - mediated: `{mediated[cid]['predicted_answer'][:120].replace(chr(10), ' ')}`")

    lines += [
        "",
        "## Error taxonomy (errors only)",
        "",
        "| Type | fixed | judge_mediated |",
        "|---|---:|---:|",
    ]
    all_types = sorted(set(err_types_fixed) | set(err_types_med))
    for t in all_types:
        lines.append(f"| {t} | {err_types_fixed.get(t, 0)} | {err_types_med.get(t, 0)} |")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
