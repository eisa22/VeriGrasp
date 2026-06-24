"""Deterministic, visibility-aware 3D oriented-bounding-box verification.

Given a raw sensor point cloud and an AI-generated oriented bounding box (OBB)
for a cuboid package, judge whether the box is plausibly placed/sized. The
cloud only covers faces the sensor can see (top-down: top + maybe grazing
sides); missing points on occluded faces must NOT count as failure. Hence the
evaluation is visibility-aware and may return ``UNVERIFIABLE`` when the visible
faces lack enough data to judge responsibly.

No ML/AI, no random RANSAC: orientation uses a deterministic PCA
(``np.linalg.eigh``). Running the same inputs twice yields identical results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PASS = "PASS"
FAIL = "FAIL"
UNVERIFIABLE = "UNVERIFIABLE"

_AXIS_NAMES = ("x", "y", "z")

DEFAULT_BOX_CHECK_CFG: dict[str, Any] = {
    "inlier_min": 0.80,
    "surface_dist_median_max_m": 0.02,
    "extent_rel_dev_max": 0.15,
    "top_normal_angle_max_deg": 12.0,
    "min_points_total": 300,
    "min_coverage": 0.60,
    "eps_m": 0.01,
    "near_face_band_m": 0.03,
    "coverage_grid": 10,
    # A face counts as expected-visible only if its outward normal faces the
    # sensor by at least this cosine (generalises the strict normal.view < 0
    # rule; grazing side faces are not "expected" to yield full data top-down).
    "face_visible_min_facing": 0.2,
}


@dataclass
class FaceCoverage:
    """Per-face observability and sampling report."""

    face: str  # "+x","-x","+y","-y","+z","-z"
    expected_visible: bool
    coverage: float
    n_points: int


@dataclass
class BoxCheckResult:
    """Outcome of the deterministic OBB verification."""

    verdict: str  # PASS | FAIL | UNVERIFIABLE
    inlier_ratio: float
    surface_dist_median_m: float
    surface_dist_rms_m: float
    extent_rel_dev: list[float]  # per box axis
    extent_observed: list[bool]  # per box axis: constrained by a visible face?
    top_normal_angle_deg: float
    faces: list[FaceCoverage] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "inlier_ratio": _safe(self.inlier_ratio),
            "surface_dist_median_m": _safe(self.surface_dist_median_m),
            "surface_dist_rms_m": _safe(self.surface_dist_rms_m),
            "extent_rel_dev": [_safe(v) for v in self.extent_rel_dev],
            "extent_observed": [bool(v) for v in self.extent_observed],
            "top_normal_angle_deg": _safe(self.top_normal_angle_deg),
            "faces": [
                {
                    "face": f.face,
                    "expected_visible": bool(f.expected_visible),
                    "coverage": _safe(f.coverage),
                    "n_points": int(f.n_points),
                }
                for f in self.faces
            ],
            "reasons": list(self.reasons),
        }


def _safe(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _merge_cfg(cfg: dict | None) -> dict[str, Any]:
    merged = dict(DEFAULT_BOX_CHECK_CFG)
    if cfg:
        for k, v in cfg.items():
            if v is not None:
                merged[k] = v
    return merged


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _box_surface_distance(local: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Unsigned distance of each local point to the axis-aligned box surface."""
    q = np.abs(local) - half[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)  # <= 0 inside, 0 outside
    sdf = outside + inside
    return np.abs(sdf)


def _robust_span(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hi = float(np.percentile(values, 98.0))
    lo = float(np.percentile(values, 2.0))
    return hi - lo


def _spanned_axes(normal_axis: int) -> tuple[int, int]:
    others = [a for a in range(3) if a != normal_axis]
    return others[0], others[1]


def _face_coverage(
    local: np.ndarray,
    half: np.ndarray,
    normal_axis: int,
    sign: int,
    band: float,
    eps: float,
    grid: int,
) -> tuple[float, int]:
    """Coverage of one face: fraction of grid cells with >=1 near-face point."""
    a0, a1 = _spanned_axes(normal_axis)
    on_plane = np.abs(local[:, normal_axis] - sign * half[normal_axis]) <= band
    within = (
        (np.abs(local[:, a0]) <= half[a0] + eps)
        & (np.abs(local[:, a1]) <= half[a1] + eps)
    )
    sel = on_plane & within
    n = int(sel.sum())
    if n == 0:
        return 0.0, 0
    pts = local[sel][:, [a0, a1]]
    g = max(1, int(grid))
    occupied = np.zeros((g, g), dtype=bool)
    # Bucket into [-half, half] per spanned axis.
    ix = np.floor((pts[:, 0] + half[a0]) / (2.0 * half[a0] + 1e-12) * g).astype(int)
    iy = np.floor((pts[:, 1] + half[a1]) / (2.0 * half[a1] + 1e-12) * g).astype(int)
    ix = np.clip(ix, 0, g - 1)
    iy = np.clip(iy, 0, g - 1)
    occupied[ix, iy] = True
    coverage = float(occupied.sum()) / float(g * g)
    return coverage, n


def verify_box_placement(
    points: np.ndarray,
    center: np.ndarray,
    R: np.ndarray,
    extents: np.ndarray,
    sensor_origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    cfg: dict | None = None,
) -> BoxCheckResult:
    """Deterministically verify an OBB against a raw point cloud.

    Args:
        points: (N,3) raw sensor cloud in the same frame as the box.
        center: (3,) OBB centre.
        R: (3,3) rotation whose COLUMNS are the box axes in the sensor frame.
        extents: (3,) full edge lengths of the box.
        sensor_origin: (3,) sensor position (camera frame default = origin).
        cfg: optional threshold overrides (see ``DEFAULT_BOX_CHECK_CFG``).
    """
    c = _merge_cfg(cfg)
    pts = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    extents = np.asarray(extents, dtype=np.float64).reshape(3)
    sensor = np.asarray(sensor_origin, dtype=np.float64).reshape(3)
    half = np.maximum(extents * 0.5, 1e-6)

    band = float(c["near_face_band_m"])
    eps = float(c["eps_m"])
    grid = int(c["coverage_grid"])
    min_facing = float(c["face_visible_min_facing"])
    min_cov = float(c["min_coverage"])
    min_pts = int(c["min_points_total"])

    reasons: list[str] = []

    if pts.ndim != 2 or pts.shape[0] == 0:
        return BoxCheckResult(
            verdict=UNVERIFIABLE, inlier_ratio=0.0,
            surface_dist_median_m=float("inf"), surface_dist_rms_m=float("inf"),
            extent_rel_dev=[float("inf")] * 3, extent_observed=[False] * 3,
            top_normal_angle_deg=float("nan"), faces=[],
            reasons=["empty point cloud"],
        )

    # --- local frame: columns of R are box axes -> world_to_local = R.T ---
    local_all = (pts - center[None, :]) @ R

    # --- near-box crop (all metrics operate on this neighbourhood) ---
    near_mask = np.all(np.abs(local_all) <= (half + band)[None, :], axis=1)
    local = local_all[near_mask]
    n_near = int(local.shape[0])

    # --- expected-visible faces ---
    faces: list[FaceCoverage] = []
    face_records: list[dict[str, Any]] = []
    for axis in range(3):
        for sign in (+1, -1):
            face_center = center + R[:, axis] * (sign * half[axis])
            normal = _unit(R[:, axis] * sign)
            dir_to_sensor = _unit(sensor - face_center)
            facing = float(np.dot(normal, dir_to_sensor))
            visible = facing > min_facing
            coverage, n_face = _face_coverage(
                local, half, axis, sign, band, eps, grid
            )
            name = f"{'+' if sign > 0 else '-'}{_AXIS_NAMES[axis]}"
            faces.append(
                FaceCoverage(
                    face=name, expected_visible=visible,
                    coverage=coverage, n_points=n_face,
                )
            )
            face_records.append(
                {"axis": axis, "sign": sign, "facing": facing,
                 "visible": visible, "coverage": coverage, "normal": normal,
                 "face_center": face_center}
            )

    # --- metric a: inlier / containment ratio (over near set) ---
    if n_near > 0:
        inside = np.all(np.abs(local) <= (half + eps)[None, :], axis=1)
        inlier_ratio = float(inside.sum()) / float(n_near)
    else:
        inlier_ratio = 0.0

    # --- metric b: point-to-surface distance (near-surface points) ---
    if n_near > 0:
        surf_d = _box_surface_distance(local, half)
        near_surf = surf_d <= band
        if near_surf.any():
            d = surf_d[near_surf]
            surface_dist_median = float(np.median(d))
            surface_dist_rms = float(np.sqrt(np.mean(d * d)))
        else:
            surface_dist_median = float("inf")
            surface_dist_rms = float("inf")
    else:
        surface_dist_median = float("inf")
        surface_dist_rms = float("inf")

    # --- metric c: extent matching per axis (only observed axes fail) ---
    extent_rel_dev: list[float] = []
    extent_observed: list[bool] = []
    for axis in range(3):
        span = _robust_span(local[:, axis]) if n_near > 0 else 0.0
        edge = 2.0 * half[axis]
        rel = abs(span - edge) / edge if edge > 1e-9 else float("inf")
        extent_rel_dev.append(float(rel))
        # observed if some visible+covered face spans this axis
        observed = any(
            fr["visible"] and fr["coverage"] >= min_cov and axis in _spanned_axes(fr["axis"])
            for fr in face_records
        )
        extent_observed.append(bool(observed))
        if not observed:
            reasons.append(
                f"extent axis {_AXIS_NAMES[axis]} unconstrained "
                f"(no visible face spans it) - supported only by the prior"
            )

    # --- metric d: top-face orientation via deterministic PCA ---
    top_fr = None
    visible_frs = [fr for fr in face_records if fr["visible"]]
    if visible_frs:
        top_fr = max(visible_frs, key=lambda fr: fr["facing"])
    top_normal_angle = float("nan")
    if top_fr is not None and n_near > 0:
        axis = top_fr["axis"]
        sign = top_fr["sign"]
        a0, a1 = _spanned_axes(axis)
        on_plane = np.abs(local[:, axis] - sign * half[axis]) <= band
        within = (
            (np.abs(local[:, a0]) <= half[a0] + eps)
            & (np.abs(local[:, a1]) <= half[a1] + eps)
        )
        face_pts_local = local[on_plane & within]
        if face_pts_local.shape[0] >= 3:
            face_pts_world = face_pts_local @ R.T  # back to sensor frame
            mean = face_pts_world.mean(axis=0)
            cov = np.cov((face_pts_world - mean).T)
            evals, evecs = np.linalg.eigh(cov)  # ascending, deterministic
            surf_normal = _unit(evecs[:, 0])
            box_normal = top_fr["normal"]
            cos_a = abs(float(np.dot(surf_normal, box_normal)))
            cos_a = max(0.0, min(1.0, cos_a))
            top_normal_angle = float(np.degrees(np.arccos(cos_a)))

    # --- verdict logic ---
    expected_visible_faces = [f for f in faces if f.expected_visible]
    gappy = [f for f in expected_visible_faces if f.coverage < min_cov]

    inlier_min = float(c["inlier_min"])
    sdist_max = float(c["surface_dist_median_max_m"])
    extent_max = float(c["extent_rel_dev_max"])
    angle_max = float(c["top_normal_angle_max_deg"])

    verdict = PASS
    if n_near < min_pts:
        verdict = UNVERIFIABLE
        reasons.append(f"too few near-box points ({n_near} < {min_pts})")
    elif not expected_visible_faces:
        verdict = UNVERIFIABLE
        reasons.append("no expected-visible face (cannot judge)")
    elif gappy:
        verdict = UNVERIFIABLE
        reasons.append(
            "expected-visible face(s) below min_coverage: "
            + ", ".join(f"{f.face}={f.coverage:.2f}" for f in gappy)
        )

    if verdict != UNVERIFIABLE:
        if inlier_ratio < inlier_min:
            verdict = FAIL
            reasons.append(f"inlier_ratio {inlier_ratio:.2f} < {inlier_min:.2f}")
        if surface_dist_median > sdist_max:
            verdict = FAIL
            reasons.append(
                f"surface_dist_median {surface_dist_median:.3f}m > {sdist_max:.3f}m"
            )
        for axis in range(3):
            if extent_observed[axis] and extent_rel_dev[axis] > extent_max:
                verdict = FAIL
                reasons.append(
                    f"extent axis {_AXIS_NAMES[axis]} rel_dev "
                    f"{extent_rel_dev[axis]:.2f} > {extent_max:.2f}"
                )
        if np.isfinite(top_normal_angle) and top_normal_angle > angle_max:
            verdict = FAIL
            reasons.append(
                f"top_normal_angle {top_normal_angle:.1f}deg > {angle_max:.1f}deg"
            )
        if verdict == PASS:
            reasons.append("all observed metrics within bounds")

    return BoxCheckResult(
        verdict=verdict,
        inlier_ratio=inlier_ratio,
        surface_dist_median_m=surface_dist_median,
        surface_dist_rms_m=surface_dist_rms,
        extent_rel_dev=extent_rel_dev,
        extent_observed=extent_observed,
        top_normal_angle_deg=top_normal_angle,
        faces=faces,
        reasons=reasons,
        detail={
            "n_near": n_near,
            "n_total": int(pts.shape[0]),
            "half_extents_m": half.tolist(),
        },
    )
