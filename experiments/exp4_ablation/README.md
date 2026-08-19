# Experiment 4 — Ablation der Verifikationsschicht

Leave-one-out auf Kriteriums- und Check-Ebene plus Greedy-Forward-
Selection über die persistierten Exp.-3-Entscheidungen
(Thesis §sec:results:ablation). Rein offline, kein Modell-Lauf.

## Ausführung

```bash
python -m experiments.exp4_ablation.evaluate \
  --exp3-dir Results/exp3/full_2026-07-05 \
  --out-dir Results/exp4/<run>
```

(Kanonischer Aufruf: `scripts/exp4_evaluate.sh`.)

## Ausgabe

`exp4_loo.csv` (Leave-one-out je Kriterium/Check), `exp4_summary.json`
(inkl. Greedy-Pfad, Acht-Check-Teilmenge), `figures/exp4_greedy_path.pdf`.

Thesis-Lauf: `Results/exp4/full_2026-07-05`, archiviert in
`results_archive/exp4_ablation/`.
