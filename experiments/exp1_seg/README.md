# Experiment 1 — Segmentierungs-Performance

Quantifiziert, wie gut die **Perception-Pipeline** einzelne Pakete segmentiert, und **an welcher Stufe** Masken verloren gehen. Beantwortet die erste Hälfte von RQ1 und liefert die Tabellen für §9.1.

Es werden **nur** der Perception-Block und die 2D-Ground-Truth (`instance_mask.npy`) verwendet — **kein** Grasp, keine Verifikation.

---

## Was wird gemessen?

**Class-agnostic Instance Segmentation:** Die Pipeline klassifiziert nicht in die 35 Dataset-Klassen. Alle GT-Instanzen sind eine Vordergrundklasse („parcel“). Klassenlabels dienen nur der **diagnostischen Recall-Aufschlüsselung**, nicht dem AP.

### Vier Messpunkte in der Pipeline

| Punkt | Stufe | Artefakt | Metriken |
|-------|--------|----------|----------|
| **D** | Post Grounding DINO | Boxen + Scores | Box-Recall @ IoU 0.5 |
| **S** | Post Sobel | `refined_masks` | Volle Mask-Metriken |
| **M** | Post Matching (Stage 5) | `closed_matches` | Volle Mask-Metriken |
| **F** | Post SAM3D | finale Masken | **Headline** — volle Mask-Metriken |

Die Deltas **S → M → F** zeigen, ob Masken an der Gradient-Stufe, am Matching-Filter oder bei SAM3D verloren gehen.

### Metriken (S, M, F)

- COCO-AP (IoU 0.50:0.95), AP50, AP75  
- Mean IoU, Panoptic Quality (PQ, SQ, RQ)  
- Precision / Recall / F1 @ IoU 0.5  
- Failure-Mode-Raten: Over-Segmentation, Merge, Hallucination, Miss  
- Recall pro GT-Klasse (nur @ F, diagnostisch)

GT und Predictions werden auf den **Workspace** geclippt (x ∈ [0.15 W, 0.85 W]). Instanzen mit **0 sichtbaren Pixeln** zählen nicht als False Negatives.

---

## Architektur: zwei Phasen

```
Phase A (GPU, einmal)          Phase B (CPU, wiederholbar)
run_inference.py        →      evaluate.py
preds/scene_XXXX.npz           metrics/ + summary.json + tables/
```

Die Pipeline läuft einmal; Metriken können offline neu berechnet werden, ohne GPU.

---

## Voraussetzungen

1. **Datensatz:** `Data/blender_dataset/` mit 728 Szenen `scene_000` … `scene_727`  
2. **Python-Umgebung:** Projekt-`.venv` im Repo-Root (enthält `torch`, `transformers`, …)

```bash
cd /pfad/zu/master_thesis
source .venv/bin/activate    # Prompt: (master_thesis)
```

Falls `(workspace)` oder System-Python aktiv ist und `ModuleNotFoundError: torch` erscheint:

```bash
source .venv/bin/activate
# oder direkt:
.venv/bin/python -m experiments.exp1_seg.run_inference ...
```

`run_inference.py` wechselt bei fehlendem `torch` automatisch zu `.venv/bin/python`, sofern vorhanden.

---

## Phase A — Inference

```bash
python -m experiments.exp1_seg.run_inference [OPTIONEN]
```

### Optionen

| Flag | Beschreibung |
|------|----------------|
| `--test-set NAME` | Nur benanntes Teilmengen-Set (siehe unten) |
| `--gui` | Open3D-Visualisierung pro Szene (Perception D→S→M→F); Fenster schließen → nächste Szene |
| `--limit N` | Nur die ersten N Szenen (nach Test-Set-Filter) |
| `--resume` | Szenen mit existierender `.npz` überspringen |
| `--run-dir PATH` | Fester Ausgabeordner |
| `--data-root PATH` | Anderer Datensatz-Root |

### Test-Sets (`eval_config.yaml`)

| Name | Inhalt |
|------|--------|
| `smoke` | `scene_000`–`004` (5 Szenen) — schneller Check |
| `baseline` | `scene_000`–`055` (56 Szenen) — Dev-Band |
| `diverse` | 11 Szenen, je eine pro Kategorie-Band |

Eigene Sets in [`eval_config.yaml`](eval_config.yaml) unter `test_sets` ergänzen.

### Beispiele

```bash
# Vollständiger Lauf (728 Szenen, ohne GUI)
python -m experiments.exp1_seg.run_inference

# Smoke-Test mit GUI
python -m experiments.exp1_seg.run_inference --test-set smoke --gui

# Nur --gui → automatisch test-set smoke
python -m experiments.exp1_seg.run_inference --gui

# Baseline-Band, fortsetzen nach Abbruch
python -m experiments.exp1_seg.run_inference --test-set baseline --resume
```

### Hilfsskript

```bash
./scripts/exp1_inference.sh --test-set smoke --gui
```

### Ausgabe (Phase A)

```
Results/exp1_seg/<timestamp>[_test-NAME]/
  env.json                 # Modell, Git-Commit, Paketversionen
  config_snapshot.yaml     # Pipeline- + Eval-Konfiguration
  test_set.json            # bei --test-set
  inference_manifest.json
  preds/scene_XXXX.npz     # Masken S/M/F (RLE), Boxen D, Scores, workspace_mask
```

---

## Phase B — Evaluation

```bash
python -m experiments.exp1_seg.evaluate \
  --run-dir Results/exp1_seg/<timestamp>[_test-NAME] \
  [--test-set NAME]
```

`--test-set` filtert die Auswertung auf dieselben Szenen wie beim Inference-Lauf (empfohlen bei Teilmengen).

```bash
RUN_DIR=$(ls -td Results/exp1_seg/*_test-smoke | head -1)
python -m experiments.exp1_seg.evaluate --run-dir "$RUN_DIR" --test-set smoke
```

### Hilfsskript

```bash
./scripts/exp1_evaluate.sh --run-dir "$RUN_DIR" --test-set smoke
```

### Ausgabe (Phase B)

```
Results/exp1_seg/<run>/
  metrics/scene_XXXX.json
  summary.json
  tables/
    per_stage.csv           # Table 9.1a — D/S/M/F
    per_category_F.csv      # Table 9.1b — 11 Kategorie-Zeilen @ F
    per_class_recall_F.csv
    failure_modes.csv
  latex/
    table_9_1a.tex
    table_9_1b.tex
```

---

## Qualitative Figuren (optional)

Overlays GT + Predictions je Failure-Mode (keine Metrik-GUI):

```bash
python -m experiments.exp1_seg.visualize_failures --run-dir "$RUN_DIR"
```

→ `figures/over_segmentation_scene_XXX.png`, …

---

## Typischer Workflow

```bash
cd master_thesis
source .venv/bin/activate

# 1) Inference (klein + GUI zum Anschauen)
python -m experiments.exp1_seg.run_inference --test-set smoke --gui

# 2) Evaluieren
RUN_DIR=$(ls -td Results/exp1_seg/*_test-smoke | head -1)
python -m experiments.exp1_seg.evaluate --run-dir "$RUN_DIR" --test-set smoke

# 3) Ergebnis
cat "$RUN_DIR/summary.json"
cat "$RUN_DIR/tables/per_stage.csv"

# 4) Optional: voller Lauf ohne GUI
python -m experiments.exp1_seg.run_inference --resume
python -m experiments.exp1_seg.evaluate --run-dir "$(ls -td Results/exp1_seg/20* | head -1)"
```

---

## Relevante Dateien

| Pfad | Rolle |
|------|--------|
| [`run_inference.py`](run_inference.py) | Phase A CLI |
| [`evaluate.py`](evaluate.py) | Phase B CLI |
| [`eval_config.yaml`](eval_config.yaml) | Test-Sets, Size-Buckets, Kategorien |
| [`test_set.py`](test_set.py) | Test-Set-Auflösung |
| [`../../perception/pipeline.py`](../../perception/pipeline.py) | Perception D→S→M→F |
| [`../../evaluation/`](../../evaluation/) | GT, Metriken, Aggregation, LaTeX |

---

## Hinweise

- **AP-Scores:** SAM3D liefert keinen eigenen Confidence-Wert. AP nutzt propagierte DINO-/`closure`-Scores (siehe `score_note` in `summary.json`).
- **GUI nur in Phase A:** `evaluate` hat kein `--gui`; Metriken sind JSON/CSV/LaTeX.
- **Branch:** `feature/experiment1`
- **Tests:** `pytest tests/test_exp1_metrics.py tests/test_exp1_test_set.py`
