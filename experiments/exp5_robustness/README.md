# Experiment 5 — Robustheit über Szenario-Bänder

Szenen-Level-Funnel (Detektion → Kandidat → Freigabe → valide) je
Kategorie-Band, aus den Exp.-3-Ergebnissen und der Ground Truth
(Thesis §sec:results:robustness). Rein offline.

## Ausführung

```bash
python -m experiments.exp5_robustness.evaluate \
  --data-root Data/blender_dataset \
  --exp3-dir Results/exp3/full_2026-07-05 \
  --out-dir Results/exp5/<run>
```

(Kanonischer Aufruf: `scripts/exp5_evaluate.sh`.)

## Ausgabe

`exp5_per_scene.csv`, `exp5_summary.json` (Funnel je Band).

Thesis-Lauf: `Results/exp5/full_2026-07-05`, archiviert in
`results_archive/exp5_robustness/`.
