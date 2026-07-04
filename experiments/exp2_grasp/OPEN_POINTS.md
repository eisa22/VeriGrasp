# Experiment 2 — Antworten auf die OPEN POINTS A–D

## A — Persistierte Kandidaten-Felder

Der kanonische Pipeline-Lauf persistierte Centroid, OBB, Bottom-Cue und
Grasp-Normale **nur für das Primärziel** (`pipeline_result.json`,
`stage10_selected_target.json`, `stage11_suction_grasps.json`). Für die
Bewertung *aller* gematchten Kandidaten wurde die Persistenz erweitert und die
Pipeline einmal neu über alle 728 Szenen gefahren (`python main.py --test`):

- `stage_prep_context.json` (neu): gefittete Palettenebene `plane_model`
  (a,b,c,d), `z_pallet_m`, Intrinsics, Workspace-Maske (RLE).
- `stage8_candidates.json` (neu): pro Kandidat `candidate_id`, Label,
  SAM3D-Score, Maske (RLE), `centroid_3d`, `top_surface_height_m`,
  `bottom_z_m`, `bottom_method`, `bottom_confidence`, `neighbor_source`,
  `parcel_obb` (center / extents / R / corners_3d).

Das Metrik-Skript bleibt strikt offline; es liest ausschließlich diese JSONs.

Feldnamen im Code: der Bottom-Cue heißt `bottom_method` (nicht `bottom_cue`),
die Quelle des Nachbarn `neighbor_source`; die Grasp-Normale steht in
`stage11_suction_grasps.json -> primary_grasp.normal`.

## B — Quaternion-Konvention und Up-Achse

- Feldname im Datensatz: `rotation_world_quat_wxyz` — also **wxyz**. Das Feld
  existiert nur bei Objekten geneigter Szenen (378 der 728 Szenen); Baseline-
  Objekte tragen nur `yaw_world_rad`.
- Die kanonische Up-Achse ist **+Z im Asset-Frame** (konsistent mit
  `dimensions_wlh`, w×l×h).
- Die Implementierung leitet die GT-Top-Normale **direkt aus den 8 Ecken** ab
  (SVD über die 4 höchsten Ecken relativ zur Pipeline-Palettenebene), wodurch
  die Quaternion-Konvention gar nicht benötigt wird und alle 728 Szenen
  einheitlich behandelt werden. Obere/untere Ecken werden per Höhe
  identifiziert, nicht per Index-Reihenfolge.
- Orientierung: Normale zeigt zur Kamera (Flip bei n·z > 0; OpenCV-Konvention,
  Kamera blickt entlang +z). Verifiziert durch den GT-Selbsttest auf
  tilted- und angled-Szenen (alle Fehler < 1e-6).

## C — Sub-Ranges der Angled-Bänder

Die Bänder werden **nicht** über Index-Ranges bestimmt, sondern über das Feld
`scene_category` in `ground_truth.json` (Mapping in
`evaluation/scene_registry.py`, identisch zu Experiment 1). Der Scan aller
728 Szenen ergibt:

| Band | scene_category | Szenen | Index-Range |
|------|----------------|--------|-------------|
| angled_dense | `angled_view_dense` | 50 | 578–725 (nicht zusammenhängend) |
| angled_chaotic | `angled_view_chaotic` | 50 | 579–726 (nicht zusammenhängend) |
| angled_occluded | `angled_view_occluded` | 50 | 580–727 (nicht zusammenhängend) |

Die drei angled-Konfigurationen sind ab Szene 578 **verschachtelt** (nicht in
drei Blöcken), weshalb die Metadaten-basierte Zuordnung zwingend ist.

## D — Soft-Packaging-Klassen

Im Repo existiert keine offizielle soft/rigid-Liste; die folgende Zuordnung
(11 Klassen, exakte `class_name`-Strings aus `dataset_meta.json`) ist in
`experiments/exp2_grasp/eval_config.yaml` hinterlegt und **muss mit Samuel
bestätigt werden** (Vorschlag aus dem Plan; `Deformed Package` heißt im
Datensatz `Asset_Deformed_Package`):

`Burlap_Sack`, `Food_Packaging_Bag`, `Paper_Coffee_Bag`, `Paper_Shopping_Bag`,
`Small_Pouch`, `Small_Shipping_Bag`, `Space_Food_Bag`, `Sachets_Package`,
`Envelope_Stack`, `Vintage_Envelope`, `Asset_Deformed_Package`.

Soft-Klassen sind aus den Band-Zeilen der Centroid-/Top-Height-Tabelle
ausgeschlossen und werden in einer separaten `soft`-Zeile gepoolt (GT-Top-Fläche
übersteigt die sichtbare Oberfläche um bis zu 58 mm per Datensatz-Konvention).

## Zusatz: Bottom-Cue-Mapping (Thesis Tab. tab:grasp:bottom)

| Priorität | `neighbor_source` / `bottom_method` | Cue-Name im JSON |
|-----------|--------------------------------------|------------------|
| 1 | `match` | `overlap_matched_parcel` |
| 2 | `overlap` | `overlap_candidate` |
| 3 | `lateral` | `lateral_neighbor` |
| 4 | `gradient_global` | `gradient_global` |
| 5 | `gradient` | `gradient_ring` |
| 6 | `histogram` | `histogram_ring` |
| 7 | `scene_plane` | `scene_plane` |
| — | `measured` | `measured_visible` |
| — | `from_pallet` | `fallback_pallet` |
| — | `uncertain` | `fallback_uncertain` |

## Behobener Pipeline-Bug (beim Batch-Lauf entdeckt)

`perception/adapter.py::build_match_neighbors` hatte einen
Einrückungsfehler: der Schleifenkörper lag außerhalb der `for`-Schleife,
wodurch (a) Szenen ohne excluded-Matches mit `UnboundLocalError` abbrachen und
(b) die Match-Nachbarliste für die Bottom-Inferenz fast immer leer blieb. Der
Fix stellt die dokumentierte Logik wieder her; der Cue
`overlap_matched_parcel` (Priorität 1) ist damit erstmals wirksam.

## Revision 2026-07-04: zwei Eval-Korrekturen

### GT-Top-Face-Normale aus Box-Kanten statt SVD über die 4 höchsten Ecken

Die 4 höchsten Ecken bilden bei stark gekippten Boxen (Roll ≈ 45°, dünne
Pakete) keine Face — der SVD-Fit lieferte dann eine Garbage-Normale nahe 90°
zur Vertikalen. Neu (`evaluation/exp2_gt.py::_top_face_normal`): die drei
orthogonalen Kantenrichtungen werden rein geometrisch aus den 8 Ecken
rekonstruiert (kürzeste, paarweise orthogonale Differenzvektoren); die
Facenormale mit dem größten Skalarprodukt zur Pallet-Up-Richtung ist die
Top-Face-Normale. `h_top`/`h_bottom` bleiben laut Spez. (Abschnitt 4) über die
4 höchsten/tiefsten Ecken definiert — dadurch bleiben `table_centroid`,
`extent` und `visibility_strata` byte-identisch (Regressionstest bestanden).

Wirkung: 17 Szenen-Thetas fielen von ~90° auf < 3°. Verbleibender Ausreißer
θ = 81° in `scene_596` liegt am degenerierten Plane-Fit (s.u.), nicht an der
GT-Konvention.

### Degenerierter Plane-Fit: `fallback_pallet` ist dort ein Pipeline-Fehler

In 106 Szenen (v. a. `angled_*`) fällt der RANSAC-Fit auf `[0, 0, 1, 0]`
zurück — eine Ebene durch den Kameraursprung statt durch die Palette. GT und
Prediction werden von der Eval im *selben* (kaputten) Frame gerechnet, es ist
also kein Eval-Bug: `e_lat`, `e_top`, θ und Extents bleiben konsistent. Nur
der `from_pallet`-Zweig der Bottom-Inferenz snappt auf `z = 0` — im kaputten
Frame ist das die Kamerahöhe, ~2.5 m über dem Paket. Das ist ein echter
Pipeline-Fehler (Erkennung: `|d|/‖n‖ < 0.5 · z_pallet_m`). Die 8 betroffenen
Kandidaten werden als eigener Cue `fallback_pallet_degenerate_plane`
ausgewiesen (Meta-Feld `n_scenes_degenerate_plane_fit`); die reguläre
`fallback_pallet`-Zeile enthält nur noch den einen gesunden Fall
(scene_053, −210.8 mm).
