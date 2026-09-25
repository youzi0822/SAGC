

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── 固定输入路径 ────────────────────────────────────────────────
DATA_ROOT = Path(
    os.environ.get("GNN_DATA_ROOT", "SAGC/GNN/goal/collect_data/more_correct")
)
PERTURBED_DATA_DIR = DATA_ROOT / "correct_data"
LABELS_DIR = DATA_ROOT / "correct_labels"
SPLITS_JSON = DATA_ROOT / "splits.json"

ORIGINAL_DIR = Path(
    os.environ.get("GNN_ORIGINAL_ROOT", "SAGC/data/"
    "3d_front_processed/bedrooms_objfeats_32_64")
)

CACHE_ROOT = Path(os.environ.get("GNN_CACHE_ROOT", str(DATA_ROOT / "cache_geom_v1")))
SOURCE_STATIC_DIR = CACHE_ROOT / "source_static_pt"
SAMPLE_GRAPH_DIR = CACHE_ROOT / "sample_graph_pt"
SAMPLE_GEOMETRY_DIR = CACHE_ROOT / "sample_geometry_pt"
MANIFESTS_DIR = CACHE_ROOT / "manifests"
LOGS_DIR = CACHE_ROOT / "logs"
BENCHMARKS_DIR = CACHE_ROOT / "benchmarks"

ALL_CACHE_DIRS = (
    SOURCE_STATIC_DIR,
    SAMPLE_GRAPH_DIR,
    SAMPLE_GEOMETRY_DIR,
    MANIFESTS_DIR,
    LOGS_DIR,
    BENCHMARKS_DIR,
)

# translations d）
COORDINATE_CONTRACT_VERSION = "coord_room_center_v1"


BOUNDARY_CONTRACT_VERSION = "boundary_ring_ccw_inward_v2_rectfallback"

GRAPH_CONTRACT_VERSION = "graph_builder_v1"

DATASET_CONTRACT_VERSION = "supervised_scene_graph_v2_fixed"

GEOMETRY_CONTRACT_VERSION = "geom_scalar_vector_v1"


DEFAULT_RADIUS = 1.6
DEFAULT_DW_RADIUS = 3.0
DEFAULT_ROOM_TYPE = os.environ.get("GNN_ROOM_TYPE", "bedroom")


GRAPH_STANDARD_FIELDS: Tuple[str, ...] = (
    "sample_uid",
    "source_uid",
    "combo_name",
    "num_objects",
    "class_labels",
    "translations",
    "sizes_half",
    "angles",
    "node_layout",
    "graph_node_feats",
    "graph_edge_index",
    "graph_edge_feats",
    "graph_edge_types",
    "edge_bert",
    "furniture_mask",
    "vnode_idx",
    "num_furniture",
    "num_door_win",
    "head_gt",
    "policy_target",
    "g_gt_raw",
    "g_gt_step",
    "g_gt_normalized",
    "changed_mask",
    "policy_valid",
    "actor_valid",
    "warnings",
)

GRAPH_VERSION_FIELDS: Tuple[str, ...] = (
    "coordinate_contract_version",
    "graph_contract_version",
    "dataset_contract_version",
)


GRAPH_REQUIRED_FIELDS: Tuple[str, ...] = tuple(
    f for f in GRAPH_STANDARD_FIELDS if f != "edge_bert"
)

GEOMETRY_STANDARD_FIELDS: Tuple[str, ...] = (
    "sample_uid",
    "source_uid",
    "geo_node_position",
    "geo_node_yaw",
    "geo_node_full_wdh",
    "geo_node_type",
    "geo_to_old_index",
    "geo_furniture_index",
    "geo_edge_index",
    "geo_edge_scalar",
    "geo_edge_vector_basis",
    "boundary_scalar",
    "boundary_vector_basis",
    "boundary_target_index",
    "boundary_scene_index",
    "boundary_usable",
    "boundary_source",
    "floor_area_ratio",
    "room_diag_D",
    "room_height_y",
    "geometry_contract_version",
)

SOURCE_STATIC_FIELDS: Tuple[str, ...] = (
    "source_uid",
    "room_width_x",
    "room_length_z",
    "room_height_y",
    "room_diag_D",
    "floor_plan_centroid",
    "boundary_vertices_centered_xz",
    "boundary_segments_xz",
    "boundary_tangent_xz",
    "boundary_inward_normal_xz",
    "boundary_segment_length",
    "boundary_usable",
    "boundary_source",
    "floor_area_ratio",
    "door_window_categories",
    "door_window_positions",
    "door_window_sizes_half_wdh",
    "door_window_thetas",
    "coordinate_contract_version",
    "boundary_contract_version",
)


# =============================================================================
#  uid
# =============================================================================

def ensure_cache_dirs(cache_root: Path = CACHE_ROOT) -> None:
    """创建固定的缓存目录结构（幂等）。"""
    cache_root = Path(cache_root)
    for name in (
        "source_static_pt",
        "sample_graph_pt",
        "sample_geometry_pt",
        "manifests",
        "logs",
        "benchmarks",
    ):
        (cache_root / name).mkdir(parents=True, exist_ok=True)


def sample_uid_to_source_uid(sample_uid: str) -> str:
    """与 SupervisedDataset2Fixed ：sample_uid.split('__')[0]。"""
    return sample_uid.split("__")[0]


def graph_cache_path(sample_uid: str, graph_cache_dir: Path = SAMPLE_GRAPH_DIR) -> Path:
    return Path(graph_cache_dir) / f"{sample_uid}.pt"


def geometry_cache_path(
    sample_uid: str, geometry_cache_dir: Path = SAMPLE_GEOMETRY_DIR
) -> Path:
    return Path(geometry_cache_dir) / f"{sample_uid}.pt"


def source_static_path(
    source_uid: str, source_static_dir: Path = SOURCE_STATIC_DIR
) -> Path:
    return Path(source_static_dir) / f"{source_uid}.pt"


def load_splits(splits_json: Path = SPLITS_JSON) -> Dict:
    with open(splits_json, "r") as f:
        return json.load(f)


def get_split_sample_uids(split: str, splits_json: Path = SPLITS_JSON) -> List[str]:
   
    splits = load_splits(splits_json)
    if split == "all":
        out: List[str] = []
        for key in ("train_sample_uids", "val_sample_uids", "test_sample_uids"):
            out.extend(splits.get(key, []))
        return out
    key = f"{split}_sample_uids"
    if key not in splits:
        raise KeyError(f"{key}；{list(splits.keys())}")
    return list(splits[key])


# =============================================================================
# 
# =============================================================================

def atomic_torch_save(obj, path: Path) -> None:
   
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def load_cache_file(path: Path) -> Dict:
   
    return torch.load(str(path), map_location="cpu", weights_only=False)


def atomic_json_dump(obj, path: Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise




def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compare_field(name: str, a, b, rtol: float = 0.0, atol: float = 0.0) -> Optional[str]:
   
    if a is None and b is None:
        return None
    if (a is None) != (b is None):
        return f"{name}: 一侧为 None (cached={a is None and 'None' or 'value'}, live={b is None and 'None' or 'value'})"

   
    if isinstance(b, str) or isinstance(a, str):
        return None if a == b else f"{name}: str 不同 cached={a!r} live={b!r}"
    if isinstance(b, (bool, np.bool_)) or isinstance(a, (bool, np.bool_)):
        return None if bool(a) == bool(b) else f"{name}: bool 不同 cached={a} live={b}"
    if isinstance(b, list) or isinstance(a, list):
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return f"{name}: list cached={len(a)} live={len(b)}"
            for i, (x, y) in enumerate(zip(a, b)):
                sub = compare_field(f"{name}[{i}]", x, y, rtol=rtol, atol=atol)
                if sub is not None:
                    return sub
            return None
        return None if a == b else f"{name}: list/"
    if isinstance(b, (int, float, np.integer, np.floating)) and not isinstance(
        b, np.ndarray
    ):
        if isinstance(a, (int, float, np.integer, np.floating)):
            if float(a) == float(b):
                return None
            if atol > 0.0 and abs(float(a) - float(b)) <= atol:
                return None
            return f"{name}:  cached={a} live={b}"

    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)
    if arr_a.shape != arr_b.shape:
        return f"{name}: shape  cached={arr_a.shape} live={arr_b.shape}"
    if arr_a.dtype != arr_b.dtype:
        return f"{name}: dtype  cached={arr_a.dtype} live={arr_b.dtype}"
    if arr_a.size == 0:
        return None
    if arr_a.dtype.kind in "OUS":
        return None if np.array_equal(arr_a, arr_b) else f"{name}"
    if arr_a.dtype.kind == "b":
        return None if np.array_equal(arr_a, arr_b) else f"{name}"
    if arr_a.dtype.kind in "iu":
        if np.array_equal(arr_a, arr_b):
            return None
        n_diff = int((arr_a != arr_b).sum())
        return f"{name}: （{n_diff}/{arr_a.size}）"

    if np.array_equal(arr_a, arr_b):
        return None
    if rtol > 0.0 or atol > 0.0:
        if np.allclose(arr_a, arr_b, rtol=rtol, atol=atol, equal_nan=True):
            return None
    max_abs = float(np.nanmax(np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))))
    return f"{name}: （max_abs_diff={max_abs:.6g}）"


def compare_samples(
    cached: Dict,
    live: Dict,
    fields: Tuple[str, ...] = GRAPH_STANDARD_FIELDS,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> List[str]:
    
    diffs: List[str] = []
    for name in fields:
        if name not in cached:
            diffs.append(f"{name}")
            continue
        if name not in live:
            diffs.append(f"{name}")
            continue
        msg = compare_field(name, cached[name], live[name], rtol=rtol, atol=atol)
        if msg is not None:
            diffs.append(msg)
    return diffs


def build_supervised_dataset_fixed(
    sample_uids: Optional[List[str]] = None,
    perturbed_data_dir: Path = PERTURBED_DATA_DIR,
    labels_dir: Path = LABELS_DIR,
    original_dir: Path = ORIGINAL_DIR,
    room_type: str = DEFAULT_ROOM_TYPE,
    radius: float = DEFAULT_RADIUS,
    dw_radius: float = DEFAULT_DW_RADIUS,
    check_consistency: bool = True,
    filter_invalid: bool = True,
):
    
    import sys

    goal_dir = Path(__file__).resolve().parent.parent / "goal"
    if str(goal_dir) not in sys.path:
        sys.path.insert(0, str(goal_dir))
    from supervised_dataset2 import SupervisedDataset2Fixed

    return SupervisedDataset2Fixed(
        perturbed_data_dir=Path(perturbed_data_dir),
        labels_dir=Path(labels_dir),
        original_dir=Path(original_dir),
        room_type=room_type,
        radius=radius,
        dw_radius=dw_radius,
        check_consistency=check_consistency,
        filter_invalid=filter_invalid,
        sample_uids_filter=sample_uids,
    )

