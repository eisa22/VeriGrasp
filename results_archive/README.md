# Results Archive — Master Thesis "Zero-Shot Foundation Models for Industrial Robotics"

This directory is the version-controlled scientific record of every
experimental result reported in the thesis. Each subdirectory is the
unmodified output of one evaluation run of the VeriGrasp pipeline on the
SynDePal benchmark (728 scenes, 19,834 annotated parcels). Every number in
the thesis' result chapters traces to a file in this archive; the mapping
is given below.

Integrity: `MANIFEST.sha256` lists a SHA-256 checksum for every file in
this archive. Verify with:

```bash
cd results_archive && shasum -a 256 -c MANIFEST.sha256
```

## Run identity

The configuration hashes and code revisions below are the ones printed in
the thesis (reproducibility table, chapter *Experimental Setup*).

| Archive directory | Thesis run | Config hash | Git commit | Run date |
|---|---|---|---|---|
| `exp1_segmentation_standard/` | Exp. 1, standard pipeline | `0a80b036` | `ce76019` | 2026-06-24 |
| `exp1_segmentation_sam_variant/` | Exp. 1, SAM-based variant | `b00accdd` | `7af2750` (variant code committed as `c78ca1b`) | 2026-08-05 |
| `exp1_equal_height/` | Equal-height case analysis | derived from the two Exp.-1 runs above | `c78ca1b` | 2026-08-17 |
| `exp2_grasp_accuracy/` | Exp. 2 | `d2f94c72` | `153dc83` | 2026-07-04 |
| `exp3_verification/` | Exp. 3 | `f1612eee` | `c9663d5` | 2026-07-05 |
| `exp4_ablation/` | Exp. 4 | `f1612eee` | `c9663d5` | 2026-07-05 |
| `exp5_robustness/` | Exp. 5 | `f1612eee` | `c9663d5` | 2026-07-05 |
| `exp6_sensitivity/` | Exp. 6 | `f1612eee` | `c9663d5` | 2026-07-05 |

Each Exp.-1 run directory carries its own `env.json` (package versions,
device, config hash, git commit), `config_snapshot.yaml`, and
`inference_manifest.json` as written by the run itself.

## Mapping: thesis tables and figures → archive files

Thesis references use LaTeX label names, which are stable across
renumbering.

| Thesis element | Archive file |
|---|---|
| Per-stage metrics, standard (`tab:exp1-per-stage`, `fig:exp1-per-stage`) | `exp1_segmentation_standard/tables/per_stage.csv` |
| Failure-mode rates (`fig:exp1-failure-modes`) | `exp1_segmentation_standard/tables/failure_modes.csv` |
| Per-category metrics, standard (`tab:exp1-per-category`, `fig:exp1-per-category`) | `exp1_segmentation_standard/tables/per_category_F.csv` |
| Per-class recall (`fig:exp1-per-class-recall`) | `exp1_segmentation_standard/tables/per_class_recall_F.csv` |
| Per-stage metrics, SAM variant (`tab:exp1-sam-per-stage`) | `exp1_segmentation_sam_variant/tables/per_stage.csv` |
| Per-category comparison (`tab:exp1-sam-per-category`) | `exp1_segmentation_sam_variant/tables/per_category_F.csv` |
| Equal-height cases (`tab:exp1-equal-height`) | `exp1_equal_height/equal_height_result.json` |
| Equal-height example (`fig:exp1-equal-height`) | `exp1_equal_height/render_equal_height_figure.py` (renders from archived predictions) |
| Exp. 2 accuracy tables and figures (`sec:results:grasp-acc`) | `exp2_grasp_accuracy/primary/` (headline) and `exp2_grasp_accuracy/strict/` (strict visibility rule) |
| Exp. 3 decision quality and ROC (`sec:results:verif-eff`) | `exp3_verification/exp3_summary.json`, `exp3_per_grasp.csv`, `figures/exp3_roc.pdf` |
| Exp. 4 ablation (`sec:results:ablation`) | `exp4_ablation/exp4_summary.json`, `exp4_loo.csv`, `figures/exp4_greedy_path.pdf` |
| Exp. 5 robustness funnel (`sec:results:robustness`) | `exp5_robustness/exp5_summary.json`, `exp5_per_scene.csv` |
| Exp. 6 threshold sensitivity (`sec:results:threshold`) | `exp6_sensitivity/exp6_summary.json`, `exp6_curves.csv`, `exp6_pairwise.csv` |

## Headline numbers (spot checks)

| Thesis claim | File carrying it |
|---|---|
| Stage-F recall 0.063, precision 0.850, mIoU 0.844 (standard) | `exp1_segmentation_standard/tables/per_stage.csv`, row F |
| Stage-F recall 0.112, precision 0.801, PQ 0.171 (SAM variant) | `exp1_segmentation_sam_variant/tables/per_stage.csv`, row F |
| 1,990 equal-height pairs; separated 34 (std) / 58 (SAM); merged 63/118; false splits 18/64 of 12,952 singles | `exp1_equal_height/equal_height_result.json` |
| 1,085 matched candidates (937 rigid / 148 soft), 616 primary grasps | `exp2_grasp_accuracy/primary/exp2_summary.json` |
| Accept precision 0.55 → 0.68, FRR 21 %; Wilson CIs 265/389, 124/279, 72/337 | `exp3_verification/exp3_summary.json` |

## Reproduction

- Exp. 1 inference: `python -m experiments.exp1_seg.run_inference`
  (standard) or `... --variant sam3d` (variant); evaluation:
  `python -m experiments.exp1_seg.evaluate --run-dir <dir>`. The archived
  `preds/*.npz` (run-length-encoded masks per scene) allow re-running the
  evaluation and the equal-height analysis without any GPU inference.
- Equal-height analysis:
  `python results_archive/exp1_equal_height/equal_height_analysis.py`
  (reads the archived predictions of both Exp.-1 runs; regenerated from
  them on 2026-08-17 and identical to the counts in the thesis).
- Exp. 2–6 are offline evaluations over recorded pipeline outputs; see
  `experiments/<exp>/README.md` for the exact commands.
- The SynDePal ground truth (`Data/blender_dataset/`, ~728 scenes with
  RGB, depth, instance masks, and exact 3D annotations) is distributed
  alongside the thesis and is required for re-evaluation; it is not part
  of this repository.
