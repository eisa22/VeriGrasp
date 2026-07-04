# Experiment 2 — Centroid- und Normalen-Genauigkeit

Misst die geometrische Genauigkeit der Grasp-Kandidaten-Stufe gegen die
synthetische Ground Truth (Thesis §sec:exp:grasp-acc). Bedingt auf Detektion:
nur Kandidaten mit GT-Match (IoU ≥ 0.5, Exp1-Matcher) werden bewertet.

Es ist ein **reines Offline-Metrik-Skript** — es liest die persistierten
Pro-Stufen-JSONs eines existierenden Pipeline-Laufs und ruft weder den
Detektor noch die Pipeline auf.

---

## Eingaben pro Szene

| Datei | Quelle | Inhalt |
|-------|--------|--------|
| `ground_truth.json`, `instance_mask.npy` | Datensatz | GT-Boxen (8 Ecken), Klassen, Sichtbarkeit |
| `stage_prep_context.json` | Pipeline (Stage 0) | Gefittete Palettenebene, Workspace-Maske, Intrinsics |
| `stage8_candidates.json` | Pipeline (Stage 8) | Alle Kandidaten: Maske (RLE), Centroid, OBB, Bottom-Inferenz |
| `stage10_selected_target.json` | Pipeline (Stage 10) | Primärziel |
| `stage11_suction_grasps.json` | Pipeline (Stage 11) | Approach-Normale des Primärgrasps |

Die Pipeline-JSONs entstehen durch einen Voll-Lauf:

```bash
python main.py --test [--resume]     # --resume überspringt Szenen mit stage8_candidates.json
```

## Referenzrahmen

Alle Höhen und lateralen Positionen relativ zur **gefitteten Palettenebene der
Pipeline** (nicht zur wahren Ebene). GT und Prediction werden mit derselben
Ebene projiziert (`heights_above_plane`, `project_to_plane_xy`).

## Metriken

Alle Längen in mm, alle Winkel in Grad. Aggregation: n / Median / P95
(bei vorzeichenbehafteten Metriken: Median signiert, P95 vom Absolutwert).

- `e_lat` — laterale Centroid-Distanz in der Ebene
- `e_top` — signierter Top-Höhen-Fehler (95. Perzentil vs. GT-Top-Fläche)
- `theta` — Winkel zwischen Grasp-Normale und GT-Top-Normale (nur Primärgrasp)
- Extent-Fehler (relativ, long/short/height) + Yaw (Faltung 180°, near-square 90°)
- `e_bottom` — signierter Bottom-Höhen-Fehler, aufgeschlüsselt nach Bottom-Cue
- Raten: `rate_lat_30` (≤ 30 mm), `rate_theta_12` / `rate_theta_30`

Soft-Klassen (11 Klassen, siehe `eval_config.yaml`) sind aus den Band-Zeilen
der Centroid-Tabelle ausgeschlossen und werden in einer `soft`-Zeile gepoolt.

## Ausführen

```bash
# Evaluation (beide Sichtbarkeitsvarianten wie Exp1: primary + strict)
./scripts/exp2_evaluate.sh --out-dir Results/exp2/<run>

# GT-Selbsttest (alle Fehler müssen < 1e-6 sein)
./scripts/exp2_evaluate.sh --gt-self-test --limit 20 --out-dir /tmp/exp2_selftest

# Spot-Check-Overlays (5 zufällige Matches aus verschiedenen Bändern)
python -m experiments.exp2_grasp.visualize_spotcheck --n 5
```

## Ausgaben

```
Results/exp2/<run>/
  primary/
    exp2_per_candidate.csv   # eine Zeile pro gematchtem Kandidaten
    exp2_per_grasp.csv       # eine Zeile pro Szene (status: evaluated / no_target / target_unmatched / no_grasp)
    exp2_summary.json        # Zahlen für tab:exp2-centroid / -normal / -bottom
  strict/
    ... (gleiche drei Dateien, strenge Sichtbarkeit)
  figures/exp2_normal_deviation.pdf   # optional, nur bei ausreichender theta-Stichprobe
```

## Tests

```bash
pytest tests/test_exp2_gt.py tests/test_exp2_yaw.py tests/test_exp2_metrics.py
```
