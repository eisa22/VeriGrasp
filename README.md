# VeriGrasp — Zero-Shot Perception mit deterministischer Grasp-Verifikation

Code zur Masterarbeit *"Zero-Shot Foundation Models for Industrial
Robotics"* (Samuel Einspieler, TU Wien, ACIN, 2026). VeriGrasp kombiniert
einen Open-Vocabulary-Detektor mit einer deterministischen,
geometriebasierten Verarbeitungskette für vakuumbasiertes Greifen in
Top-down-Palettierszenen, evaluiert auf dem synthetischen Benchmark
**SynDePal** (728 Szenen, 19.834 annotierte Pakete).

## Pipeline

1. **Detektion** — Grounding DINO (box-Prompts, Workspace-/Größenfilter,
   relative NMS) — `GroundingSAM/grounding_sam.py`
2. **Segmentierung** — deterministische Tiefengradienten-Segmentierung
   (Sobel, per-Box-Otsu) — `Segmentation/`
3. **Matching** — Closure-Matching der Gradienten-Segmente gegen die
   DINO-Boxen — `Segmentation/` + `Visualization/`
4. **3D-Verfeinerung** — DBSCAN-basiertes Splitting/Outlier-Removal —
   `Sam3D/sam3d.py` (Modulname historisch; implementiert die
   DBSCAN-Verfeinerung, nicht SAM3D von Yang et al.)
5. **Grasp-Generierung** — Sauggreif-Kandidaten, Bottom-Plane-Inferenz,
   Extraction-Corridor — `perception/`
6. **Verifikation** — deterministische Kaskade aus zwanzig geometrischen
   Prüfungen mit Audit-Record — `verification/`

**SAM-Vergleichsvariante** (Experiment 1): ersetzt Stufen 2–3 durch
box-gepromptete SAM-Masken (ViT-B) + IoU-Dedup, Stufe 4 unverändert —
`perception/pipeline_sam3d.py`, aktivierbar mit `--variant sam3d`.

## Projektstruktur

```
config.py               zentrale Konfiguration (Modelle, Schwellen)
main.py                 Gesamt-Pipeline (eine Szene / --test Batch)
GroundingSAM/           Grounding-DINO-Detektion + SAM-Maskengenerierung
Segmentation/           Tiefengradienten-Segmentierung + Matching
Sam3D/                  DBSCAN-3D-Verfeinerung + Masken-Dedup
perception/             Exp1-Pipelines (D→S→M→F), Grasp-Generierung
verification/           20-Check-Verifikationskaskade
evaluation/             Metriken, GT-Handling, Aggregation (Exp1–6)
experiments/            Runner der sechs Experimente (je mit README)
scripts/                Hilfsskripte pro Experiment
results_archive/        versionierte Messergebnisse aller Experimente
Visualization/          Open3D-Visualisierung
LLMOrchestrator/        optionaler LLM-Orchestrator (in den Experimenten
                        nicht verwendet)
tests/                  Unit-/Integrationstests
```

## Ergebnisse (results_archive/)

Alle in der Thesis berichteten Messergebnisse liegen versioniert in
[`results_archive/`](results_archive/README.md): ein Verzeichnis pro
Lauf (Exp. 1 Standard + SAM-Variante inkl. per-Szene-Predictions,
Equal-Height-Analyse, Exp. 2–6), ein README mit dem Mapping jeder
Thesis-Tabelle/-Abbildung auf ihre Datei sowie `MANIFEST.sha256` über
alle Dateien.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision transformers open3d numpy pillow \
    opencv-python scikit-learn scipy matplotlib pyyaml
```

Der optionale LLM-Orchestrator liest `OPENAI_API_KEY` aus der
Umgebungsvariable; für Pipeline und Experimente wird kein Key benötigt.

## Verwendung

**Datensatz:** `Data/blender_dataset/scene_000` … `scene_727` (SynDePal;
wird mit der Thesis verteilt, nicht Teil dieses Repos).

```bash
# Gesamt-Pipeline, eine Szene mit Visualisierung
python main.py

# Batch über alle Szenen (schreibt schlanke JSONs nach Results/)
python main.py --test

# Experiment 1: Segmentierung (Standard-Pipeline)
python -m experiments.exp1_seg.run_inference --test-set smoke
python -m experiments.exp1_seg.evaluate --run-dir <run-dir> --test-set smoke

# Experiment 1: SAM-Vergleichsvariante
python -m experiments.exp1_seg.run_inference --variant sam3d

# Equal-Height-Analyse aus den archivierten Predictions
python results_archive/exp1_equal_height/equal_height_analysis.py
```

Die Protokolle, Optionen und Ausgabeformate der sechs Experimente sind in
`experiments/<exp>/README.md` dokumentiert; Reproduzierbarkeits-Identitäten
(Config-Hash, Commit) je Lauf stehen in `results_archive/README.md` und in
der Thesis (Kapitel *Experimental Setup*).

## Referenzen

- Grounding DINO — Liu et al., ECCV 2024, [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- Segment Anything (SAM) — Kirillov et al., 2023, [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- SAM3D (konzeptuelle Inspiration der 3D-Verfeinerung; nicht ausgeführt) — Yang et al., 2023, [arXiv:2306.03908](https://arxiv.org/abs/2306.03908)
- DBSCAN — Ester et al., KDD 1996

## Autor und Lizenz

Samuel Einspieler — Masterarbeit, TU Wien (ACIN). Code und Ergebnisarchiv
stehen unter der MIT-Lizenz (siehe [LICENSE](LICENSE)).
