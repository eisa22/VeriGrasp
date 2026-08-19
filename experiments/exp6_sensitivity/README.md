# Experiment 6 — Schwellenwert-Sensitivität

Sweept die Schwellen der einzelnen Checks über die persistierten
per-Check-Margins von Exp. 3 und kartiert Operating-Ranges und
paarweise Interaktionen (Thesis §sec:results:threshold). Rein offline.

## Ausführung

```bash
./scripts/exp6_evaluate.sh
```

Das Skript patcht zuerst `exp3_per_grasp.csv` um `n_blocking`-Spalten
(`scripts/patch_exp3_n_blocking.py`) und ruft dann:

```bash
python -m experiments.exp6_sensitivity.evaluate \
  --exp3-dir Results/exp3/full_2026-07-05 \
  --out-dir Results/exp6/<run>
```

## Ausgabe

`exp6_curves.csv` (Sensitivitätskurven je Check), `exp6_pairwise.csv`
(Interaktions-Spotchecks), `exp6_summary.json`, `figures/`.

Thesis-Lauf: `Results/exp6/full_2026-07-05`, archiviert in
`results_archive/exp6_sensitivity/`. Tests: `pytest tests/test_exp6_sensitivity.py`.
