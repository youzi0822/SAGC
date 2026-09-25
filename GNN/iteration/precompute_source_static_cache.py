

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_GOAL_DIR = _THIS_DIR.parent / "goal"
if str(_GOAL_DIR) not in sys.path:
    sys.path.insert(0, str(_GOAL_DIR))

import cache_common as cc
from boundary_geometry import (
    RECT_FALLBACK_AREA_RATIO_MIN,
    extract_boundary_segments,
    floor_area_from_faces,
    rectangle_boundary,
)
from door_window_utils import extract_door_window_nodes
from supervised_dataset2 import load_boxes_npz, load_door_win_json


def list_source_uids(split: str) -> List[str]:

    if split == "scan":
        seen = {}
        for d in sorted(cc.PERTURBED_DATA_DIR.iterdir()):
            if d.is_dir():
                seen[cc.sample_uid_to_source_uid(d.name)] = True
        return list(seen.keys())

    splits = cc.load_splits()
    if split == "all":
        out: List[str] = []
        seen = set()
        for key in ("train_source_uids", "val_source_uids", "test_source_uids"):
            for uid in splits.get(key, []):
                if uid not in seen:
                    seen.add(uid)
                    out.append(uid)
        return out
    key = f"{split}_source_uids"
    if key not in splits:
        raise KeyError(f"splits.json 缺少 {key}")
    return list(splits[key])


def find_sibling_sample_dirs(source_uid: str) -> List[Path]:
  
    dirs = []
    exact = cc.PERTURBED_DATA_DIR / source_uid
    if exact.is_dir():
        dirs.append(exact)
    dirs.extend(sorted(p for p in cc.PERTURBED_DATA_DIR.glob(f"{source_uid}__*") if p.is_dir()))
    return dirs


def _room_shell_dims(door_win_data: Optional[Dict]) -> Tuple[float, float, float, List[str]]:
  
    warnings: List[str] = []
    if door_win_data is None or "sample" not in door_win_data:
        return 0.0, 0.0, 0.0, ["no_door_win_json"]
    sample = door_win_data["sample"]
    room_shell = sample.get("room_shell", {}) or {}
    width_x = float(room_shell.get("width_x", sample.get("width_x", 0.0)))
    length_z = float(room_shell.get("length_z", sample.get("length_z", 0.0)))
    if "height_y" in room_shell:
        height_y = float(room_shell["height_y"])
    elif "height_y" in sample:
        height_y = float(sample["height_y"])
    else:
        height_y = 0.0
        warnings.append("missing_room_height_y")
    if width_x <= 0.0 or length_z <= 0.0:
        warnings.append("nonpositive_room_dims")
    return width_x, length_z, height_y, warnings


def _static_signature(boxes_data: Dict, door_win_data: Optional[Dict]) -> str:
   
    parts = []
    for key in ("floor_plan_vertices", "floor_plan_faces", "floor_plan_centroid"):
        arr = boxes_data.get(key, None)
        parts.append(np.asarray(arr).tobytes().hex() if arr is not None else "None")
    if door_win_data is not None and "sample" in door_win_data:
        sample = door_win_data["sample"]
        parts.append(json.dumps(sample.get("room_shell", {}), sort_keys=True))
        parts.append(json.dumps(sample.get("openings", []), sort_keys=True))
    else:
        parts.append("no_door_win")
    return "|".join(parts)


PERIMETER_MISMATCH_TOL = 0.05

RING_AREA_TOL = 0.01


def _boundary_is_broken(
    boundary: Dict,
    width_x: float,
    length_z: float,
    mesh_floor_area: float,
) -> Optional[str]:
   
    for w in boundary["warnings"]:
        if w.startswith("multi_ring"):
            return w
        if w.startswith("no_boundary_edges") or w.startswith("no_valid_ring"):
            return w

    segs = np.asarray(boundary["segments_xz"], dtype=np.float64)
    if segs.shape[0] < 3:
        return "too_few_segments"
    if float(boundary["segment_length"].sum()) <= 0.0:
        return "zero_perimeter"

   
    ring = segs[:, 0, :]
    x, z = ring[:, 0], ring[:, 1]
    ring_area = 0.5 * abs(
        float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))
    )
    if mesh_floor_area > 0.0:
        rel = abs(ring_area - mesh_floor_area) / mesh_floor_area
        if rel > RING_AREA_TOL:
            return f"ring_area_mismatch_rel_{rel:.4f}"
    return None


def _resolve_boundary(
    fv: Optional[np.ndarray],
    ff: Optional[np.ndarray],
    floor_plan_centroid: np.ndarray,
    width_x: float,
    length_z: float,
) -> Tuple[Dict, str, bool, float, List[str]]:
    
    warnings: List[str] = []

    if fv is None or ff is None:
        warnings.append("missing_floor_plan_mesh")
        if width_x > 0.0 and length_z > 0.0:
            boundary = rectangle_boundary(width_x, length_z)
            warnings.extend(boundary["warnings"])
          
            return boundary, "rect_unverified", False, float("nan"), warnings
        boundary = {
            "vertices_centered_xz": np.zeros((0, 2), dtype=np.float32),
            "segments_xz": np.zeros((0, 2, 2), dtype=np.float32),
            "tangent_xz": np.zeros((0, 2), dtype=np.float32),
            "inward_normal_xz": np.zeros((0, 2), dtype=np.float32),
            "segment_length": np.zeros((0,), dtype=np.float32),
            "n_rings": 0,
            "warnings": ["missing_floor_plan_mesh"],
        }
        return boundary, "none", False, float("nan"), warnings

    boundary = extract_boundary_segments(fv, ff, floor_plan_centroid)
    warnings.extend(boundary["warnings"])

    rect_area = width_x * length_z
    floor_area = floor_area_from_faces(fv, ff)
    area_ratio = float(floor_area / rect_area) if rect_area > 0.0 else float("nan")

  
    perimeter = float(boundary["segment_length"].sum())
    shell_perimeter = 2.0 * (width_x + length_z)
    if shell_perimeter > 0.0 and perimeter > 0.0:
        rel = abs(perimeter - shell_perimeter) / shell_perimeter
        if rel > PERIMETER_MISMATCH_TOL:
            warnings.append(f"perimeter_gt_shell_rel_{rel:.3f}")

    broken = _boundary_is_broken(boundary, width_x, length_z, floor_area)
    if broken is None:
        return boundary, "mesh", True, area_ratio, warnings

    warnings.append(f"boundary_broken({broken})")
    if rect_area > 0.0 and area_ratio >= RECT_FALLBACK_AREA_RATIO_MIN:
        rect = rectangle_boundary(width_x, length_z)
        warnings.append(f"rect_fallback_area_ratio_{area_ratio:.4f}")
        return rect, "rect_exact", True, area_ratio, warnings

    warnings.append(f"boundary_unusable_area_ratio_{area_ratio:.4f}")
    return boundary, "mesh_broken", False, area_ratio, warnings


def build_source_static_entry(
    source_uid: str,
    room_type: str = cc.DEFAULT_ROOM_TYPE,
    verify_siblings: int = 0,
) -> Dict:
    
    sibling_dirs = find_sibling_sample_dirs(source_uid)
    if not sibling_dirs:
        raise RuntimeError(f"{source_uid}: correct_data ")

    rep_dir = None
    boxes_data = None
    door_win_data = None
    for cand in sibling_dirs:
        bd = load_boxes_npz(cand / "boxes.npz")
        if bd is None:
            continue
        rep_dir = cand
        boxes_data = bd
        door_win_data = load_door_win_json(cand / "door_win.json")
        break

    if rep_dir is None or boxes_data is None:
        raise RuntimeError(f"{source_uid}")

    warnings: List[str] = []
    width_x, length_z, height_y, dim_warnings = _room_shell_dims(door_win_data)
    warnings.extend(dim_warnings)

    
    room_diag_D = float(np.sqrt(width_x ** 2 + length_z ** 2))

    floor_plan_centroid = np.asarray(
        boxes_data.get("floor_plan_centroid", np.zeros(3)), dtype=np.float32
    ).reshape(-1)

    fv = boxes_data.get("floor_plan_vertices", None)
    ff = boxes_data.get("floor_plan_faces", None)
    boundary, boundary_source, boundary_usable, floor_area_ratio, b_warnings = (
        _resolve_boundary(fv, ff, floor_plan_centroid, width_x, length_z)
    )
    warnings.extend(b_warnings)

    
    dw_categories, dw_positions, dw_sizes, dw_thetas = extract_door_window_nodes(
        door_win_data, floor_plan_centroid, room_type=room_type
    )

    sibling_mismatch = 0
    n_verified = 0
    if verify_siblings > 0 and len(sibling_dirs) > 1:
        ref_sig = _static_signature(boxes_data, door_win_data)
        step = max(1, len(sibling_dirs) // max(1, verify_siblings))
        for cand in sibling_dirs[::step][:verify_siblings]:
            if cand == rep_dir:
                continue
            bd = load_boxes_npz(cand / "boxes.npz")
            if bd is None:
                continue
            dw = load_door_win_json(cand / "door_win.json")
            n_verified += 1
            if _static_signature(bd, dw) != ref_sig:
                sibling_mismatch += 1
        if sibling_mismatch > 0:
            warnings.append(f"sibling_static_mismatch_{sibling_mismatch}")

    return {
        "source_uid": source_uid,
        "room_width_x": np.float32(width_x),
        "room_length_z": np.float32(length_z),
        "room_height_y": np.float32(height_y),
        "room_diag_D": np.float32(room_diag_D),
        "floor_plan_centroid": floor_plan_centroid.astype(np.float32),
        "boundary_vertices_centered_xz": boundary["vertices_centered_xz"],
        "boundary_segments_xz": boundary["segments_xz"],
        "boundary_tangent_xz": boundary["tangent_xz"],
        "boundary_inward_normal_xz": boundary["inward_normal_xz"],
        "boundary_segment_length": boundary["segment_length"],
        "boundary_usable": bool(boundary_usable),
        "boundary_source": boundary_source,
        "floor_area_ratio": np.float32(floor_area_ratio),
        "door_window_categories": list(dw_categories),
        "door_window_positions": np.asarray(dw_positions, dtype=np.float32).reshape(-1, 3),
        "door_window_sizes_half_wdh": np.asarray(dw_sizes, dtype=np.float32).reshape(-1, 3),
        "door_window_thetas": np.asarray(dw_thetas, dtype=np.float32).reshape(-1, 2),
        "coordinate_contract_version": cc.COORDINATE_CONTRACT_VERSION,
        "boundary_contract_version": cc.BOUNDARY_CONTRACT_VERSION,
        "representative_sample_uid": rep_dir.name,
        "num_sibling_samples": len(sibling_dirs),
        "num_boundary_segments": int(boundary["segments_xz"].shape[0]),
        "n_boundary_rings": int(boundary["n_rings"]),
        "num_door_window": int(len(dw_categories)),
        "room_type": room_type,
        "siblings_verified": n_verified,
        "warnings": warnings,
    }


def _worker(task: Tuple[str, str, int, bool]) -> Dict:
    source_uid, room_type, verify_siblings, overwrite = task
    out_path = cc.source_static_path(source_uid)
    try:
        if out_path.exists() and not overwrite:
            return {"source_uid": source_uid, "status": "skipped", "warnings": []}
        entry = build_source_static_entry(
            source_uid, room_type=room_type, verify_siblings=verify_siblings
        )
        cc.atomic_torch_save(entry, out_path)
        return {
            "source_uid": source_uid,
            "status": "ok",
            "warnings": entry["warnings"],
            "num_boundary_segments": entry["num_boundary_segments"],
            "num_door_window": entry["num_door_window"],
            "boundary_usable": entry["boundary_usable"],
            "boundary_source": entry["boundary_source"],
        }
    except Exception as e:
        return {
            "source_uid": source_uid,
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=5),
            "warnings": [],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=" source ")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "val", "test", "scan"],
        help="source_uid ；scan orrect_data",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help=" source")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--room-type", default=cc.DEFAULT_ROOM_TYPE)
    parser.add_argument(
        "--verify-siblings",
        type=int,
        default=0,
        help=" source ",
    )
    args = parser.parse_args()

    cc.ensure_cache_dirs()
    source_uids = list_source_uids(args.split)
    if args.limit > 0:
        source_uids = source_uids[: args.limit]

    print(f"[source_static] split={args.split} sources={len(source_uids)} workers={args.workers}")
    t0 = time.time()

    tasks = [(uid, args.room_type, args.verify_siblings, args.overwrite) for uid in source_uids]
    results: List[Dict] = []

    if args.workers <= 1:
        for i, task in enumerate(tasks):
            results.append(_worker(task))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)} ...", flush=True)
    else:
        import multiprocessing as mp

        with mp.Pool(processes=args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, tasks, chunksize=8)):
                results.append(res)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(tasks)} ...", flush=True)

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = sum(1 for r in results if r["status"] == "failed")

    warn_counts: Dict[str, int] = {}
    for r in results:
        for w in r.get("warnings", []):
            key = w.split("_rel_")[0].rstrip("0123456789_")
            warn_counts[key] = warn_counts.get(key, 0) + 1

    print(
        f"[source_static] ok={n_ok} skipped={n_skip} failed={n_fail} "
        f"用时={elapsed:.1f}s"
    )
    if warn_counts:
        print(f"[source_static] warning : {warn_counts}")

    bsrc_counts: Dict[str, int] = {}
    n_unusable = 0
    for r in results:
        if r["status"] != "ok":
            continue
        key = r.get("boundary_source", "unknown")
        bsrc_counts[key] = bsrc_counts.get(key, 0) + 1
        if not r.get("boundary_usable", True):
            n_unusable += 1
    if bsrc_counts:
        print(f"[source_static] : {bsrc_counts}")
        print(
            f"[source_static]  source: {n_unusable}/{n_ok} "
            f"( + mesh； manifest  "
            f"boundary_usable_sample_uids )"
        )

    log = {
        "split": args.split,
        "num_sources": len(source_uids),
        "ok": n_ok,
        "skipped": n_skip,
        "failed": n_fail,
        "elapsed_sec": round(elapsed, 2),
        "room_type": args.room_type,
        "verify_siblings": args.verify_siblings,
        "warning_counts": warn_counts,
        "boundary_source_counts": bsrc_counts,
        "num_boundary_unusable": n_unusable,
        "rect_fallback_area_ratio_min": RECT_FALLBACK_AREA_RATIO_MIN,
        "coordinate_contract_version": cc.COORDINATE_CONTRACT_VERSION,
        "boundary_contract_version": cc.BOUNDARY_CONTRACT_VERSION,
        "failures": [r for r in results if r["status"] == "failed"][:50],
    }
    log_path = cc.LOGS_DIR / f"source_static_{args.split}.json"
    cc.atomic_json_dump(log, log_path)
    print(f"[source_static] : {log_path}")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
