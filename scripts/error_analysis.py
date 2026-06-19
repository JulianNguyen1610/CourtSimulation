"""Error analysis cho Phase 1 ViLQA debate experiments.

Phân loại lỗi từ predictions.csv (và optional debate_result.json) thành taxonomy
đặc thù cho legal QA debate, không dùng gold answer để gán nhãn lỗi (tránh leak).

Cách dùng:
    python -m scripts.error_analysis outputs/vilqa_multi_agent_baseline/<run_dir>
    python -m scripts.error_analysis outputs/vilqa_multi_agent_baseline/<run_dir> --compare direct debate

Output: in báo cáo markdown ra stdout + ghi error_analysis.md vào run_dir.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Taxonomy lỗi đặc thù cho legal QA debate (không dùng gold answer)
# --------------------------------------------------------------------------- #
UNKNOWN_TOKENS = {"không xác định", "khong xac dinh", "không có", "n/a", "none", "null", ""}

# Pattern phát hiện answer bị paraphrase sang tiếng Anh (ViLQA gold luôn là tiếng Việt)
ENGLISH_HINTS = re.compile(
    r"\b(the|to|from|and|or|of|is|are|by|dong|vnd|years?|months?|days?|million|billion)\b",
    flags=re.IGNORECASE,
)

# Pattern phát hiện answer lấy nhầm số năm/tháng (không phải span tiền/phạt)
DURATION_PATTERN = re.compile(r"^\s*\d{1,3}\s*(năm|tháng|ngày|tuần)\s*$", flags=re.IGNORECASE)
MONEY_PATTERN = re.compile(r"\d{1,3}(?:[.,]\d{3})*\s*đồng", flags=re.IGNORECASE)


def classify_error(row: pd.Series) -> str:
    """Gán 1 nhãn lỗi cho mỗi prediction. Không dùng gold_answer để gán nhãn.

    Quy tắc ưu tiên:
    1. UNKNOWN_ANSWER: model trả token "Không xác định" hoặc rỗng.
    2. EMPTY_PARSE: fallback do JSON parse fail (không có predicted_answer).
    3. ENGLISH_PARAPHRASE: answer chứa từ tiếng Anh đáng kể (gold luôn tiếng Việt).
    4. WRONG_SPAN_TYPE: câu hỏi hỏi tiền nhưng trả số năm, hoặc ngược lại.
    5. OVER_EXTRACTION: answer dài bất thường (>12 từ) với câu hỏi cần span ngắn.
    6. NUMERIC_MISMATCH: answer có số nhưng khác loại đơn vị so với question.
    7. PARTIAL_SPAN: answer ngắn nhưng F1 thấp (span sai đoạn).
    8. CORRECT: EM=1.
    """
    pred = str(row.get("predicted_answer", "")).strip()
    gold = str(row.get("gold_answer", "")).strip()
    question = str(row.get("question", "")).lower()
    em = float(row.get("exact_match", 0.0))
    f1 = float(row.get("f1", 0.0))

    if em >= 1.0:
        return "CORRECT"

    norm_pred = pred.lower().strip()
    if norm_pred in UNKNOWN_TOKENS:
        return "UNKNOWN_ANSWER"

    if not pred:
        return "EMPTY_PARSE"

    # Phát hiện paraphrase tiếng Anh
    english_matches = len(ENGLISH_HINTS.findall(pred))
    if english_matches >= 2 and not re.search(r"đồng|năm|tháng|ngày", pred, flags=re.IGNORECASE):
        return "ENGLISH_PARAPHRASE"

    # Wrong span type: hỏi tiền nhưng trả thời gian, hoặc ngược lại
    asks_money = bool(re.search(r"tiền|đồng|bao nhiêu.*tiền", question))
    asks_duration = bool(re.search(r"bao lâu|bao nhiêu năm|bao nhiêu tháng|thời hạn", question))

    if asks_money and DURATION_PATTERN.match(pred) and not MONEY_PATTERN.search(pred):
        return "WRONG_SPAN_TYPE"
    if asks_duration and MONEY_PATTERN.search(pred) and not DURATION_PATTERN.match(pred):
        return "WRONG_SPAN_TYPE"

    # Over-extraction: answer dài nhưng câu hỏi yêu cầu span ngắn
    word_count = len(pred.split())
    if word_count > 12:
        return "OVER_EXTRACTION"

    # Partial span: F1 > 0 nhưng < 0.7, answer ngắn
    if 0 < f1 < 0.7 and word_count <= 12:
        return "PARTIAL_SPAN"

    # Numeric mismatch: có số nhưng không khớp loại
    if 0 < f1 < 0.5:
        return "NUMERIC_MISMATCH"

    return "OTHER"


def analyze_run(run_dir: Path, methods: list[str] | None = None) -> dict:
    """Phân tích 1 run dir, trả về dict thống kê."""
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {predictions_path}")

    df = pd.read_csv(predictions_path)
    df["error_type"] = df.apply(classify_error, axis=1)

    if methods:
        df = df[df["method"].isin(methods)]

    report: dict = {
        "run_dir": str(run_dir),
        "total_predictions": len(df),
        "methods": sorted(df["method"].unique().tolist()),
        "by_method": {},
        "overall_error_taxonomy": {},
    }

    # Thống kê theo method
    for method, group in df.groupby("method"):
        total = len(group)
        correct = int((group["exact_match"] >= 1.0).sum())
        avg_f1 = float(group["f1"].mean())
        em_rate = float(group["exact_match"].mean())

        error_counts = Counter(group["error_type"].tolist())
        # Loại CORRECT khỏi taxonomy lỗi, giữ dạng Counter để sort
        error_only = Counter(
            {k: v for k, v in error_counts.items() if k != "CORRECT"}
        )

        report["by_method"][method] = {
            "total": total,
            "correct": correct,
            "em_rate": round(em_rate, 4),
            "avg_f1": round(avg_f1, 4),
            "error_taxonomy": dict(error_only.most_common()),
            "error_rate": round(sum(error_only.values()) / max(total, 1), 4),
        }

    # Taxonomy tổng hợp (tất cả methods)
    all_errors = Counter(df[df["error_type"] != "CORRECT"]["error_type"].tolist())
    report["overall_error_taxonomy"] = dict(all_errors.most_common())

    # Phân tích chéo direct vs debate (nếu có cả 2)
    if {"direct", "debate"}.issubset(set(df["method"].unique())):
        report["cross_method"] = _cross_method_analysis(df)

    return report


def _cross_method_analysis(df: pd.DataFrame) -> dict:
    """So sánh direct vs debate trên cùng case_id để tìm pattern thắng/thua."""
    direct = df[df["method"] == "direct"].set_index("case_id")[
        ["predicted_answer", "exact_match", "f1", "error_type"]
    ].rename(columns=lambda c: f"direct_{c}")
    debate = df[df["method"] == "debate"].set_index("case_id")[
        ["predicted_answer", "exact_match", "f1", "error_type"]
    ].rename(columns=lambda c: f"debate_{c}")

    merged = direct.join(debate, how="inner")

    direct_wins = int(
        ((merged["direct_exact_match"] > merged["debate_exact_match"])).sum()
    )
    debate_wins = int(
        ((merged["debate_exact_match"] > merged["direct_exact_match"])).sum()
    )
    both_wrong = int(
        ((merged["direct_exact_match"] == 0) & (merged["debate_exact_match"] == 0)).sum()
    )
    both_right = int(
        ((merged["direct_exact_match"] == 1) & (merged["debate_exact_match"] == 1)).sum()
    )

    # Cases debate sửa được lỗi của direct
    debate_fixes_direct = merged[
        (merged["direct_error_type"] != "CORRECT")
        & (merged["debate_error_type"] == "CORRECT")
    ].index.tolist()

    # Cases debate làm hỏng answer đúng của direct
    debate_breaks_direct = merged[
        (merged["direct_error_type"] == "CORRECT")
        & (merged["debate_error_type"] != "CORRECT")
    ].index.tolist()

    # Cùng lỗi (debate không thêm giá trị)
    same_error = merged[
        (merged["direct_error_type"] == merged["debate_error_type"])
        & (merged["direct_error_type"] != "CORRECT")
    ].index.tolist()

    return {
        "total_compared": len(merged),
        "direct_wins": direct_wins,
        "debate_wins": debate_wins,
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "debate_fixed_direct_errors": debate_fixes_direct[:10],
        "debate_broke_direct_correct": debate_breaks_direct[:10],
        "same_error_no_value": same_error[:10],
    }


def format_report(report: dict) -> str:
    """Format report thành markdown."""
    lines: list[str] = []
    lines.append("# Error Analysis Report")
    lines.append("")
    lines.append(f"**Run:** `{report['run_dir']}`")
    lines.append(f"**Total predictions:** {report['total_predictions']}")
    lines.append(f"**Methods:** {', '.join(report['methods'])}")
    lines.append("")

    lines.append("## 1. Overall Metrics by Method")
    lines.append("")
    lines.append("| Method | N | EM rate | Avg F1 | Correct | Error rate |")
    lines.append("|---|---|---|---|---|---|")
    for method, stats in report["by_method"].items():
        lines.append(
            f"| {method} | {stats['total']} | {stats['em_rate']} | "
            f"{stats['avg_f1']} | {stats['correct']} | {stats['error_rate']} |"
        )
    lines.append("")

    lines.append("## 2. Error Taxonomy by Method")
    lines.append("")
    for method, stats in report["by_method"].items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Error type | Count | % of errors |")
        lines.append("|---|---|---|")
        total_errors = sum(stats["error_taxonomy"].values()) or 1
        for etype, count in stats["error_taxonomy"].items():
            pct = round(100 * count / total_errors, 1)
            lines.append(f"| {etype} | {count} | {pct}% |")
        lines.append("")

    if "cross_method" in report:
        cross = report["cross_method"]
        lines.append("## 3. Cross-Method Analysis (direct vs debate)")
        lines.append("")
        lines.append(f"- **Total compared:** {cross['total_compared']}")
        lines.append(f"- **Direct wins:** {cross['direct_wins']}")
        lines.append(f"- **Debate wins:** {cross['debate_wins']}")
        lines.append(f"- **Both correct:** {cross['both_correct']}")
        lines.append(f"- **Both wrong:** {cross['both_wrong']}")
        lines.append("")
        lines.append("### Cases debate FIXED direct's errors:")
        for cid in cross["debate_fixed_direct_errors"]:
            lines.append(f"- {cid}")
        lines.append("")
        lines.append("### Cases debate BROKE direct's correct answers:")
        for cid in cross["debate_broke_direct_correct"]:
            lines.append(f"- {cid}")
        lines.append("")
        lines.append("### Cases with same error (debate added no value):")
        for cid in cross["same_error_no_value"]:
            lines.append(f"- {cid}")
        lines.append("")

    lines.append("## 4. Error Type Definitions")
    lines.append("")
    lines.append("- **CORRECT**: Exact match = 1.0")
    lines.append("- **UNKNOWN_ANSWER**: Model trả 'Không xác định' hoặc rỗng")
    lines.append("- **EMPTY_PARSE**: Fallback do JSON parse fail")
    lines.append("- **ENGLISH_PARAPHRASE**: Answer chứa từ tiếng Anh (gold luôn tiếng Việt)")
    lines.append("- **WRONG_SPAN_TYPE**: Hỏi tiền trả thời gian, hoặc ngược lại")
    lines.append("- **OVER_EXTRACTION**: Answer > 12 từ cho câu hỏi cần span ngắn")
    lines.append("- **PARTIAL_SPAN**: Answer ngắn nhưng F1 thấp (sai đoạn)")
    lines.append("- **NUMERIC_MISMATCH**: Có số nhưng không khớp loại đơn vị")
    lines.append("- **OTHER**: Lỗi chưa phân loại")

    return "\n".join(lines)


def main() -> int:
    # Ép stdout/stderr dùng UTF-8 để in tiếng Việt trên Windows console (cp1252)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Error analysis cho Phase 1 ViLQA debate")
    parser.add_argument("run_dir", type=Path, help="Đường dẫn tới run dir chứa predictions.csv")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Chỉ phân tích các method cụ thể (vd: direct debate)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ghi báo cáo markdown ra file (mặc định: <run_dir>/error_analysis.md)",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Lỗi: {args.run_dir} không tồn tại", file=sys.stderr)
        return 1

    report = analyze_run(args.run_dir, methods=args.compare)
    markdown = format_report(report)

    print(markdown)

    output_path = args.output or (args.run_dir / "error_analysis.md")
    output_path.write_text(markdown, encoding="utf-8")
    print(f"\n---\nĐã ghi báo cáo: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
