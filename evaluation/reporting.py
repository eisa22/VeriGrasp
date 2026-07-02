"""CSV and LaTeX table export for §9.1."""

from __future__ import annotations

import csv
from pathlib import Path


METRIC_COLS = [
    "ap_macro",
    "ap50_macro",
    "ap75_macro",
    "mean_iou",
    "pq",
    "precision",
    "recall",
    "f1",
    "box_recall",
    "n_scenes",
    "n_gt_eval",
    "obj_per_scene",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "---"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def latex_per_stage(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Segmentation metrics per pipeline stage (Experiment~1).}",
        r"\label{tab:exp1-per-stage}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Stage & AP & AP$_{50}$ & AP$_{75}$ & mIoU & PQ & P & R & F1 \\",
        r"\midrule",
    ]
    for r in rows:
        pt = r["label"]
        if pt == "D":
            br = r.get("box_recall")
            lines.append(
                f"{pt} & --- & --- & --- & --- & --- & --- & {_fmt(br)} & --- \\\\"
            )
        else:
            lines.append(
                f"{pt} & {_fmt(r['ap_macro'])} & {_fmt(r['ap50_macro'])} & "
                f"{_fmt(r['ap75_macro'])} & {_fmt(r['mean_iou'])} & {_fmt(r['pq'])} & "
                f"{_fmt(r['precision'])} & {_fmt(r['recall'])} & {_fmt(r['f1'])} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_per_category(rows: list[dict], footnotes: dict[str, str] | None = None) -> str:
    footnotes = footnotes or {}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Segmentation at stage~F by dataset category ($n$ scenes, objects/scene).}",
        r"\label{tab:exp1-per-category}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Category & $n$ & obj/sc & AP & AP$_{50}$ & AP$_{75}$ & mIoU & PQ & R & F1 \\",
        r"\midrule",
    ]
    for r in rows:
        label = r["label"]
        fn = footnotes.get(label, "")
        lines.append(
            f"{label}{fn} & {r['n_scenes']} & {_fmt(r['obj_per_scene'], 1)} & "
            f"{_fmt(r['ap_macro'])} & {_fmt(r['ap50_macro'])} & {_fmt(r['ap75_macro'])} & "
            f"{_fmt(r['mean_iou'])} & {_fmt(r['pq'])} & {_fmt(r['recall'])} & {_fmt(r['f1'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_reporting_tables(run_dir: Path, summary: dict) -> None:
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    latex = run_dir / "latex"
    latex.mkdir(parents=True, exist_ok=True)

    per_stage = summary.get("per_stage", [])
    per_cat = summary.get("per_category_F", [])
    per_class = summary.get("per_class_recall_F", [])
    failure = summary.get("failure_modes", [])

    write_csv(tables / "per_stage.csv", per_stage)
    write_csv(tables / "per_category_F.csv", per_cat)
    write_csv(tables / "per_class_recall_F.csv", per_class)
    write_csv(tables / "failure_modes.csv", failure)

    footnotes = {"tilted": r"\footnotemark", "edge": r"\footnotemark"}
    with open(latex / "table_9_1a.tex", "w", encoding="utf-8") as f:
        f.write(latex_per_stage(per_stage))
    with open(latex / "table_9_1b.tex", "w", encoding="utf-8") as f:
        f.write(latex_per_category(per_cat, footnotes))
