# VeriGrasp — Zero-Shot Perception with Deterministic Grasp Verification

Code accompanying the master's thesis *"Zero-Shot Foundation Models for
Industrial Robotics"* (Samuel Einspieler, TU Wien, ACIN, 2026). VeriGrasp
combines an open-vocabulary detector with a deterministic, geometry-based
processing chain for vacuum-based grasping in top-down palletising scenes,
evaluated on the synthetic benchmark **SynDePal** (728 scenes, 19,834
annotated parcels).

## Pipeline

1. **Detection** — Grounding DINO (box prompts, workspace and size filters,
   relative NMS) — `GroundingSAM/grounding_sam.py`
2. **Segmentation** — deterministic depth-gradient segmentation (Sobel,
   per-box Otsu) — `Segmentation/`
3. **Matching** — closure matching of the gradient segments against the
   DINO boxes — `Segmentation/` + `Visualization/`
4. **3D refinement** — DBSCAN-based splitting and outlier removal —
   `Sam3D/sam3d.py` (the module name is historical; it implements the
   DBSCAN refinement, not SAM3D by Yang et al.)
5. **Grasp generation** — suction grasp candidates, bottom-plane inference,
   extraction corridor — `perception/`
6. **Verification** — deterministic cascade of twenty geometric checks with
   an audit record — `verification/`

**SAM comparison variant** (Experiment 1): replaces stages 2–3 with
box-prompted SAM masks (ViT-B) plus IoU deduplication, stage 4 unchanged —
`perception/pipeline_sam3d.py`, enabled with `--variant sam3d`.

## Project structure

```
config.py               central configuration (models, thresholds)
main.py                 full pipeline (single scene / --test batch)
GroundingSAM/           Grounding DINO detection + SAM mask generation
Segmentation/           depth-gradient segmentation + matching
Sam3D/                  DBSCAN 3D refinement + mask deduplication
perception/             Exp. 1 pipelines (D→S→M→F), grasp generation
verification/           20-check verification cascade
evaluation/             metrics, ground-truth handling, aggregation (Exp. 1–6)
experiments/            runners of the six experiments (each with a README)
scripts/                helper scripts per experiment
results_archive/        versioned measurement results of all experiments
Visualization/          Open3D visualisation
LLMOrchestrator/        optional LLM orchestrator (not used in the experiments)
tests/                  unit and integration tests
```

## Results (results_archive/)

All measurement results reported in the thesis are versioned in
[`results_archive/`](results_archive/README.md): one directory per run
(Exp. 1 standard and SAM variant including per-scene predictions, the
equal-height analysis, Exp. 2–6), a README that maps every thesis table and
figure to the file carrying it, and a `MANIFEST.sha256` covering all files.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision transformers open3d numpy pillow \
    opencv-python scikit-learn scipy matplotlib pyyaml
```

The optional LLM orchestrator reads `OPENAI_API_KEY` from the environment;
no key is required for the pipeline or the experiments.

## Usage

**Dataset:** `Data/blender_dataset/scene_000` … `scene_727` (SynDePal) is
versioned in this repository — RGB, depth, instance masks, exact 3D ground
truth, and the persisted pipeline records per scene. Only the derivable
`pointcloud.ply` files are omitted; regenerate them once with:

```bash
python scripts/regenerate_pointclouds.py
```

(verified to reproduce the originals to float precision).

```bash
# Full pipeline, single scene with visualisation
python main.py

# Batch over all scenes (writes compact JSON records to Results/)
python main.py --test

# Experiment 1: segmentation (standard pipeline)
python -m experiments.exp1_seg.run_inference --test-set smoke
python -m experiments.exp1_seg.evaluate --run-dir <run-dir> --test-set smoke

# Experiment 1: SAM comparison variant
python -m experiments.exp1_seg.run_inference --variant sam3d

# Equal-height analysis from the archived predictions
python results_archive/exp1_equal_height/equal_height_analysis.py
```

The protocols, options, and output formats of the six experiments are
documented in `experiments/<exp>/README.md`; the reproducibility identifiers
(configuration hash, commit) per run are listed in
`results_archive/README.md` and in the thesis (chapter *Experiments*).

## References

- Grounding DINO — Liu et al., ECCV 2024, [arXiv:2303.05499](https://arxiv.org/abs/2303.05499)
- Segment Anything (SAM) — Kirillov et al., 2023, [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- SAM3D (conceptual inspiration for the 3D refinement; not executed) — Yang et al., 2023, [arXiv:2306.03908](https://arxiv.org/abs/2306.03908)
- DBSCAN — Ester et al., KDD 1996

## Author and licence

Samuel Einspieler — master's thesis, TU Wien (ACIN). The code and the results
archive are released under the MIT licence (see [LICENSE](LICENSE)).
