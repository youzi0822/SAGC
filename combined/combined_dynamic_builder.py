

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_GNN_ROOT = _HERE.parent / "GNN"
_GOAL_DIR = _GNN_ROOT / "goal"
_ITER_DIR = _GNN_ROOT / "iteration"
_KNN_DIR = _GNN_ROOT / "k-NN"

for p in [_GNN_ROOT, _GOAL_DIR, _ITER_DIR, _KNN_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from supervised_dataset2 import (
    BEDROOM_CATEGORIES,
    CAT_TO_IDX,
    load_boxes_npz,
    load_door_win_json,
)
from door_window_utils import (
    extract_door_window_nodes,
    combine_furniture_and_door_window,
)


_graph_builder = None

def _import_graph_builder():
    global _graph_builder
    if _graph_builder is None:
        from graph_builder import build_scene_graph
        _graph_builder = build_scene_graph
    return _graph_builder


from boundary_geometry import build_edge_features, build_boundary_relations
from adapters.vocab import CLASSES_BY_ROOM
from evaluation.layout_schema import DEFAULT_LAYOUT_SPEC



TRANSLATION_DIM = 3
SIZE_DIM = 3
ANGLE_DIM = 2  
CLASS_DIM = 22  
OBJFEAT_DIM = 32

EMPTY_CLASS_INDEX = CLASS_DIM - 1


class AllSlotsEmpty(ValueError):
  


  FURNITURE_RADIUS = 1.6

DW_RADIUS = 3.0


NUM_BEDROOM_CATEGORIES = 23


NODE_TYPE_FURNITURE = 0
NODE_TYPE_DOOR = 1
NODE_TYPE_WINDOW = 2


class SceneContextLoader:
  

    def __init__(
        self,
        dataset_dir: str | Path = Path(__file__).resolve().parents[1] / "dataset" / "3d_front_processed" / "bedrooms_objfeats_32_64",
        room_type: str = "bedroom",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.room_type = room_type
        self._cache: Dict[str, Dict] = {}

    def load(self, source_uid: str) -> Dict[str, Any]:
   
        if source_uid in self._cache:
            return self._cache[source_uid]

        scene_dir = self.dataset_dir / source_uid
        boxes_path = scene_dir / "boxes.npz"
        door_win_path = scene_dir / "door_win.json"
      
        if not boxes_path.exists():
            matches = sorted(self.dataset_dir.glob(f"*_{source_uid}"))
            if len(matches) == 1:
                scene_dir = matches[0]
                boxes_path = scene_dir / "boxes.npz"
                door_win_path = scene_dir / "door_win.json"

        if not boxes_path.exists():
            raise FileNotFoundError(f"boxes.npz 不存在: {boxes_path}")

        boxes_data = load_boxes_npz(boxes_path)
        if boxes_data is None:
            raise RuntimeError(f"无法加载 boxes.npz: {boxes_path}")

        door_win_data = load_door_win_json(door_win_path)
        if self.room_type != 'bedroom':
            if door_win_data is None:
                raise FileNotFoundError(f"Missing room openings: {door_win_path}")
            from precompute_source_static_cache import _room_shell_dims, _resolve_boundary
            width, length, height, _ = _room_shell_dims(door_win_data)
            centroid = np.asarray(boxes_data['floor_plan_centroid'], dtype=np.float32).reshape(3)
            boundary, source, usable, _, _ = _resolve_boundary(
                boxes_data.get('floor_plan_vertices'), boxes_data.get('floor_plan_faces'), centroid, width, length)
            if not usable:
                raise ValueError(f"Unusable room boundary: {source_uid} ({source})")
            cats, pos, sizes, thetas = extract_door_window_nodes(door_win_data, centroid, room_type=self.room_type)
            context = dict(source_uid=source_uid, room_type=self.room_type, room_width_x=width,
                           room_length_z=length, room_height_y=height, room_diag_D=float(np.hypot(width, length)),
                           floor_plan_centroid=centroid, dw_categories=cats, dw_positions=pos,
                           dw_sizes_half=sizes, dw_thetas=thetas,
                           boundary_segments_xz=boundary['segments_xz'], boundary_tangent_xz=boundary['tangent_xz'],
                           boundary_inward_normal_xz=boundary['inward_normal_xz'], boundary_segment_length=boundary['segment_length'])
            self._cache[source_uid] = context
            return context

    
        if door_win_data and "sample" in door_win_data:
            sample = door_win_data["sample"]
            room_shell = sample.get("room_shell", {}) or {}
            width_x = float(room_shell.get("width_x", sample.get("width_x", 0.0)))
            length_z = float(room_shell.get("length_z", sample.get("length_z", 0.0)))
            height_y = float(room_shell.get("height_y", sample.get("height_y", 2.5)))
        else:
            width_x = length_z = height_y = 0.0

        room_diag_D = float(np.sqrt(width_x ** 2 + length_z ** 2))

        floor_plan_centroid = boxes_data.get("floor_plan_centroid", np.zeros(3, dtype=np.float32))
        floor_plan_centroid = np.asarray(floor_plan_centroid, dtype=np.float32).reshape(3)

        dw_categories, dw_positions, dw_sizes_half, dw_thetas = extract_door_window_nodes(
            door_win_data,
            floor_plan_centroid,
            room_type=self.room_type
        )

  
        from boundary_geometry import extract_boundary_segments, rectangle_boundary
        fv = boxes_data.get("floor_plan_vertices")
        ff = boxes_data.get("floor_plan_faces")

        if fv is not None and ff is not None:
            boundary = extract_boundary_segments(fv, ff, floor_plan_centroid)
        else:
           
            boundary = rectangle_boundary(width_x, length_z)

        context = {
            "source_uid": source_uid,
            "room_width_x": width_x,
            "room_length_z": length_z,
            "room_height_y": height_y,
            "room_diag_D": room_diag_D,
            "floor_plan_centroid": floor_plan_centroid,
            "dw_categories": dw_categories,
            "dw_positions": dw_positions,
            "dw_sizes_half": dw_sizes_half,
            "dw_thetas": dw_thetas,
            "boundary_segments_xz": boundary["segments_xz"],
            "boundary_tangent_xz": boundary["tangent_xz"],
            "boundary_inward_normal_xz": boundary["inward_normal_xz"],
            "boundary_segment_length": boundary["segment_length"],
        }

        self._cache[source_uid] = context
        return context




def decode_x0_from_ddpm(
    x0: torch.Tensor,
    dataset,
    spec=DEFAULT_LAYOUT_SPEC,
) -> Dict[str, np.ndarray]:

    assert x0.ndim == 3 and x0.shape[0] == 1, f"x0 应为 [1, max_objects, 62]，got {x0.shape}"

    x0 = x0[0]  


    offset = 0
    translations_norm = x0[:, offset:offset+TRANSLATION_DIM]  
    offset += TRANSLATION_DIM

    sizes_norm = x0[:, offset:offset+SIZE_DIM]  
    offset += SIZE_DIM

    angle_cossin = x0[:, offset:offset+ANGLE_DIM]  
    offset += ANGLE_DIM

    class_logits = x0[:, offset:offset+spec.class_dim]
    offset += spec.class_dim

    objfeats = x0[:, offset:offset+spec.objfeat_dim]
    if offset + spec.objfeat_dim != x0.shape[-1]:
        raise ValueError(
            f"x0：offset={offset}+{OBJFEAT_DIM} != {x0.shape[-1]}"
        )


    valid_mask = (class_logits[:, spec.class_dim - 1] < 0).cpu().numpy()
    N = int(valid_mask.sum())

    if N == 0:
        raise AllSlotsEmpty(
            f" {class_logits.shape[0]}  "
            f"（ch{TRANSLATION_DIM + SIZE_DIM + ANGLE_DIM + EMPTY_CLASS_INDEX}）"
            f"all >= 0"
        )

    samples_dict = {
        "translations": translations_norm[valid_mask].unsqueeze(0).cpu(),  
        "sizes": sizes_norm[valid_mask].unsqueeze(0).cpu(),                
        "angles": angle_cossin[valid_mask].unsqueeze(0).cpu(),             
        "class_labels": class_logits[valid_mask].unsqueeze(0).cpu(),       
    }

    
    boxes = dataset.post_process(samples_dict)

  
    boxes = {k: v.squeeze(0) if v.dim() == 3 else v for k, v in boxes.items()}

   
    translations = boxes["translations"].numpy()  
    sizes_half = boxes["sizes"].numpy()          
    angles_data = boxes["angles"].numpy()         

    
    if angles_data.ndim == 1:
      
        angles_data = angles_data.reshape(-1, 1)

    if angles_data.shape[1] == 1:
    
        angles = angles_data
    elif angles_data.shape[1] == 2:

        angles = np.arctan2(angles_data[:, 1:2], angles_data[:, 0:1]) 
    else:
        raise ValueError(f"Unexpected angles_data shape: {angles_data.shape}")

  
    class_labels = boxes["class_labels"].numpy()  # [N, 22]

    slot_indices = np.flatnonzero(valid_mask).astype(np.int64)

    return {
        "class_labels": class_labels,
        "translations": translations,
        "sizes_half": sizes_half,
        "angles": angles,
        "num_objects": N,
        "slot_indices": slot_indices,
    }




def build_batch(
    x0: torch.Tensor,
    scene_context: Dict[str, Any],
    ddpm_interface=None,
) -> Dict[str, Any]:

    
    scene_id = scene_context["scene_id"]
    raw_scene = scene_context["raw_scene"]

    
    dataset = scene_context.get("_dataset")
    if dataset is None and ddpm_interface is not None:
        dataset = ddpm_interface.dataset
    if dataset is None:
        raise ValueError(
            "error"  
        )

    spec = scene_context.get('layout_spec', getattr(ddpm_interface, 'layout_spec', DEFAULT_LAYOUT_SPEC))
    room_type = scene_context.get('room_type', getattr(ddpm_interface, 'room_type', 'bedroom'))
    furniture_state = decode_x0_from_ddpm(x0, dataset, spec)
   
    names = scene_context.get('class_names', getattr(dataset, 'class_labels', None))
    if names is None and room_type != 'bedroom':
        raise ValueError('Room-specific diffusion class names are required')
    furniture_state['class_names'] = list(names[:spec.furniture_class_dim]) if names is not None else BEDROOM_CATEGORIES[:21]
    furniture_state['room_type'] = room_type
    N = furniture_state["num_objects"]

   
    disk_scene_id = scene_context.get("full_scene_id", scene_id)
    dataset_dir = scene_context.get("static_dataset_directory") or scene_context.get("dataset_directory")
    if dataset_dir is None and ddpm_interface is not None:
        dataset_dir = ddpm_interface.config["data"]["dataset_directory"]
    loader = SceneContextLoader(dataset_dir, room_type=room_type) if dataset_dir is not None else SceneContextLoader(room_type=room_type)
    static_context = loader.load(disk_scene_id)

   
    scene_context["room_diag"] = static_context["room_diag_D"]
    scene_context["room_height"] = static_context["room_height_y"]


    semantic_graph = _build_semantic_graph(furniture_state, static_context)


    geometry_graph = _build_geometry_graph(furniture_state, semantic_graph, static_context)


    boundary_data = _build_boundary_relations(furniture_state, static_context)


    batch = {
        "has_geometry": True,
        "ddpm_slot_indices": torch.from_numpy(furniture_state["slot_indices"]).long(),
        **semantic_graph,
        **geometry_graph,
        **boundary_data,
    }

    return batch


def _build_semantic_graph(
    furniture_state: Dict,
    static_context: Dict,
) -> Dict[str, torch.Tensor]:
   
    N = furniture_state["num_objects"]


    class_labels = furniture_state["class_labels"]  
    translations = furniture_state["translations"]  
    sizes_half = furniture_state["sizes_half"]      
    angles = furniture_state["angles"]              

    
    room_type = furniture_state.get('room_type', 'bedroom')
    class_names = furniture_state.get('class_names', BEDROOM_CATEGORIES[:21])
    cat_to_idx = {name: i for i, name in enumerate(CLASSES_BY_ROOM[room_type])}
    categories = []
    for i in range(N):
        idx = int(np.argmax(class_labels[i, :len(class_names)]))
        categories.append(class_names[idx])


    yaw = angles[:, 0]  # [N]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    thetas = np.stack([cos_yaw, sin_yaw], axis=1)  

  
    sizes_for_graph = np.stack([
        sizes_half[:, 0],  
        sizes_half[:, 2],  
        sizes_half[:, 1],  
    ], axis=1)  # [N, 3]

    
    dw_categories = static_context["dw_categories"]
    dw_positions = static_context["dw_positions"]
    dw_sizes_half = static_context["dw_sizes_half"]
    dw_thetas = static_context["dw_thetas"]

    all_categories, all_positions, all_sizes, all_thetas, num_furniture = \
        combine_furniture_and_door_window(
            categories, translations, sizes_for_graph, thetas,
            dw_categories, dw_positions, dw_sizes_half, dw_thetas
        )

    num_door_window = len(dw_categories)

   
    build_scene_graph = _import_graph_builder()
    graph_node_feats, graph_edge_index, graph_edge_feats, graph_edge_types = \
        build_scene_graph(
            categories=all_categories,
            cat_to_idx=cat_to_idx,
            positions=all_positions,
            sizes=all_sizes,
            thetas=all_thetas,
            radius=FURNITURE_RADIUS,
            num_furniture=num_furniture,
            dw_radius=DW_RADIUS,
        )

    vnode_idx = num_furniture + num_door_window

  
    return {
        "graph_node_feats": torch.from_numpy(graph_node_feats).float(),
        "graph_edge_index": torch.from_numpy(graph_edge_index).long(),
        "graph_edge_feats": torch.from_numpy(graph_edge_feats).float(),
        "graph_edge_types": torch.from_numpy(graph_edge_types).long(),
        "vnode_indices": torch.tensor([vnode_idx], dtype=torch.long),
        "furniture_node_ranges": torch.tensor([[0, num_furniture]], dtype=torch.long),
        "scene_furniture_counts": torch.tensor([num_furniture], dtype=torch.long),
    }


def _build_geometry_graph(
    furniture_state: Dict,
    semantic_graph: Dict,
    static_context: Dict,
) -> Dict[str, torch.Tensor]:
   
    N = furniture_state["num_objects"]
    num_door_window = len(static_context["dw_categories"])
    V_geo = N + num_door_window

    graph_node_feats = semantic_graph["graph_node_feats"].numpy()
    graph_edge_index = semantic_graph["graph_edge_index"].numpy()
    graph_edge_types = semantic_graph["graph_edge_types"].numpy()

  
    cat_to_idx = {name: i for i, name in enumerate(CLASSES_BY_ROOM[furniture_state.get('room_type', 'bedroom')])}
    num_categories = len(cat_to_idx)
    layout_tail = graph_node_feats[:V_geo, num_categories:]  

    geo_node_position = layout_tail[:, 0:3]  
    cos_yaw = layout_tail[:, 3]
    sin_yaw = layout_tail[:, 4]
    geo_node_yaw = np.arctan2(sin_yaw, cos_yaw)  
    half_wdh = layout_tail[:, 5:8]  
    geo_node_full_wdh = 2.0 * half_wdh  

   
    geo_node_type = np.full(V_geo, NODE_TYPE_FURNITURE, dtype=np.int64)
    if num_door_window > 0:
        cat_idx = np.argmax(graph_node_feats[:V_geo, :num_categories], axis=1)
        door_i = cat_to_idx["door"]
        window_i = cat_to_idx["window"]
        dw_slice = slice(N, V_geo)
        dw_cat = cat_idx[dw_slice]
        geo_node_type[dw_slice] = np.where(
            dw_cat == door_i,
            NODE_TYPE_DOOR,
            np.where(dw_cat == window_i, NODE_TYPE_WINDOW, NODE_TYPE_FURNITURE),
        )

   
    geo_to_old_index = np.arange(V_geo, dtype=np.int64)
    geo_node_scene_index = np.zeros(V_geo, dtype=np.int64)
    geo_furniture_index = np.arange(N, dtype=np.int64)

    
    keep = (graph_edge_index[0] < V_geo) & (graph_edge_index[1] < V_geo)
    geo_edge_index = graph_edge_index[:, keep]
    geo_edge_type = graph_edge_types[keep]

  
    room_diag_D = static_context["room_diag_D"]
    room_height_y = static_context["room_height_y"]

    geo_edge_scalar, geo_edge_vector_basis = build_edge_features(
        edge_index=geo_edge_index,
        positions=geo_node_position,
        yaw=geo_node_yaw,
        full_wdh=geo_node_full_wdh,
        room_diag_D=room_diag_D,
        room_height_y=room_height_y,
    )

    return {
        "geo_node_position": torch.from_numpy(geo_node_position).float(),
        "geo_node_yaw": torch.from_numpy(geo_node_yaw).float(),
        "geo_node_full_wdh": torch.from_numpy(geo_node_full_wdh).float(),
        "geo_node_type": torch.from_numpy(geo_node_type).long(),
        "geo_to_old_index": torch.from_numpy(geo_to_old_index).long(),
        "geo_node_scene_index": torch.from_numpy(geo_node_scene_index).long(),
        "geo_furniture_index": torch.from_numpy(geo_furniture_index).long(),
        "geo_edge_index": torch.from_numpy(geo_edge_index).long(),
        "geo_edge_type": torch.from_numpy(geo_edge_type).long(),
        "geo_edge_scalar": torch.from_numpy(geo_edge_scalar).float(),
        "geo_edge_vector_basis": torch.from_numpy(geo_edge_vector_basis).float(),
        "room_diag_D": torch.tensor([room_diag_D], dtype=torch.float32),
        "room_height_y": torch.tensor([room_height_y], dtype=torch.float32),
    }


def _build_boundary_relations(
    furniture_state: Dict,
    static_context: Dict,
) -> Dict[str, torch.Tensor]:
    
    N = furniture_state["num_objects"]
    translations = furniture_state["translations"]
    sizes_half = furniture_state["sizes_half"]
    angles = furniture_state["angles"]

    yaw = angles[:, 0]

  
    furniture_full_wdh = np.stack([
        2.0 * sizes_half[:, 0],  # full_w
        2.0 * sizes_half[:, 2],  # full_d
        2.0 * sizes_half[:, 1],  # full_h
    ], axis=1)

    furniture_geo_index = np.arange(N, dtype=np.int64)

   
    boundary = build_boundary_relations(
        furniture_positions=translations,
        furniture_yaw=yaw,
        furniture_full_wdh=furniture_full_wdh,
        furniture_geo_index=furniture_geo_index,
        segments_xz=static_context["boundary_segments_xz"],
        tangent_xz=static_context["boundary_tangent_xz"],
        inward_normal_xz=static_context["boundary_inward_normal_xz"],
        segment_length=static_context["boundary_segment_length"],
        room_diag_D=static_context["room_diag_D"],
        room_height_y=static_context["room_height_y"],
    )

    R = boundary["boundary_scalar"].shape[0]
    boundary_scene_index = np.zeros(R, dtype=np.int64)

    return {
        "boundary_scalar": torch.from_numpy(boundary["boundary_scalar"]).float(),
        "boundary_vector_basis": torch.from_numpy(boundary["boundary_vector_basis"]).float(),
        "boundary_target_index": torch.from_numpy(boundary["boundary_target_index"]).long(),
        "boundary_scene_index": torch.from_numpy(boundary_scene_index).long(),
        "boundary_length": torch.from_numpy(boundary['boundary_length']).float(),
    }





_GLOBAL_DDPM = None

def set_ddpm_interface(ddpm):
   
    global _GLOBAL_DDPM
    _GLOBAL_DDPM = ddpm


def create_builder(dataset_dir: str | Path = None) -> callable:
   
    def builder(x0: torch.Tensor, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        return build_batch(x0, scene_context, ddpm_interface=_GLOBAL_DDPM)

    return builder


def build(x0: torch.Tensor, scene_context: Dict[str, Any]) -> Dict[str, Any]:
    
    return build_batch(x0, scene_context, ddpm_interface=_GLOBAL_DDPM)


if __name__ == "__main__":
   
    print("combined_dynamic_builder ")
    print(f"BEDROOM_CATEGORIES : {len(BEDROOM_CATEGORIES)}")
    print(f"furniture={FURNITURE_RADIUS}m, door/window={DW_RADIUS}m")
    print("\n use：")
    print("  python run_combined.py --combined-config combined_inference.yaml \\")
    print("    --batch-builder combined_dynamic_builder:create_builder")
