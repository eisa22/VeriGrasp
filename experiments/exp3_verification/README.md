# Experiment 3 — Verifikations-Entscheidungsqualität

Bewertet die Verifikationsschicht auf den 616 primären Griffen: Accept-
Precision, False-Accept-/False-Reject-Rate, per-Check-Verhalten und die
Soft-Score-ROC (Thesis §sec:results:verif-eff). Offline-Evaluation: liest
die Szenen des Datensatzes, rekonstruiert die Kaskaden-Verdikte aus den
persistierten per-Check-Margins und vergleicht gegen das geometrische
Validitäts-Orakel.

## Ausführung

```bash
python -m experiments.exp3_verification.evaluate \
  --data-root Data/blender_dataset \
  --out-dir Results/exp3/<run> \
  --visibility strict
```

Optionen: `--visibility {primary,strict,both}`, `--limit N`,
`--scenes a,b,...`, `--secondary-sample` (alle gerankten Griffe,
nicht in den Thesis-Tabellen).

## Ausgabe

`exp3_per_grasp.csv` (ein Griff pro Zeile: Orakel, Kaskaden-Verdikt,
per-Check-Margins, Soft-Score), `exp3_summary.json` (Headline-Raten,
Wilson-Intervalle), `figures/exp3_roc.pdf`.

## Hinweise

- Der Thesis-Lauf ist `Results/exp3/full_2026-07-05`, archiviert in
  `results_archive/exp3_verification/`; Code-Stand Commit `c9663d5`
  (Branch-Historie `feature/experiment3`, Runner-Commit `9c8aa92`).
  Für eine exakte Reproduktion diesen Stand auschecken; auf `main`
  wurden die `evaluation/`-Module später für Exp. 5/6 weiterentwickelt.
- Experiment 6 patcht `exp3_per_grasp.csv` nachträglich um
  `n_blocking`-Spalten (`scripts/patch_exp3_n_blocking.py`).
