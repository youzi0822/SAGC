
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


GNN_ROOT = Path(__file__).parent.parent
KNN_DIR = GNN_ROOT / "k-NN"
if str(KNN_DIR) not in sys.path:
    sys.path.insert(0, str(KNN_DIR))

def _import_graph_builder():
   
    import sys
    from pathlib import Path


    current_file = Path(__file__).resolve()
    KNN_DIR = current_file.parent.parent / "k-NN"
    KNN_DIR = KNN_DIR.resolve()

    if str(KNN_DIR) not in sys.path:
        sys.path.insert(0, str(KNN_DIR))

    try:
        from graph_builder import build_scene_graph
        return build_scene_graph
    except ImportError as e:
        raise ImportError(f"·graph_builder: {e}\nKNN_DIR: {KNN_DIR}\nsys.path: {sys.path[:3]}")

build_scene_graph = None  # 

# ── 导入门窗工具 ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from door_window_utils import extract_door_window_nodes, combine_furniture_and_door_window
from room_data_config import load_class_names, original_root

# ── 版本信息 ────────────────────────────────────────────────────
DATASET_VERSION = "supervised_scene_graph_v2"
LABEL_VERSION = "v1_supervised_correction"
GRAPH_VERSION = "graph_builder_v1"
FEATURE_VERSION = "layout_C+8_v1"


ACTOR_SCALE = np.array([
    0.20,   # gx  (m)
    0.05,   # gy  (m)
    0.20,   # gz  (m)
    0.10,   # gw  (m)
    0.10,   # gd  (m)
    0.10,   # gh  (m)
    0.26    # g_theta (rad)
], dtype=np.float32)


HEAD_MASKS = {
    0: [1, 1, 1, 0, 0, 0, 1],  # position + rotation
    1: [0, 0, 0, 1, 1, 1, 1],  # size + rotation
    2: [1, 1, 1, 1, 1, 1, 0],  # position + size
    3: [1, 1, 1, 1, 1, 1, 1],  # position + size + rotation
}

BEDROOM_CATEGORIES = [
    "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair",
    "children_cabinet", "coffee_table", "desk", "double_bed",
    "dressing_chair", "dressing_table", "kids_bed", "nightstand",
    "pendant_lamp", "shelf", "single_bed", "sofa", "stool",
    "table", "tv_stand", "wardrobe",
    "door", "window"  
]


CAT_TO_IDX = {cat: i for i, cat in enumerate(BEDROOM_CATEGORIES)}




def load_boxes_npz(npz_path: Path) -> Optional[Dict]:
    
    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)

        # 提取核心字段
        class_labels = data["class_labels"]  # [N, C]
        translations = data["translations"]  # [N, 3]
        sizes = data["sizes"]               # [N, 3]
        angles = data["angles"]             # [N, 1]

        N = class_labels.shape[0]
        if N == 0:
            return None

      
        result = {
            "class_labels": class_labels,
            "translations": translations,
            "sizes": sizes,
            "angles": angles,
            "N": N,
        }

     
        for key in ["uids", "jids", "scene_uid", "scene_id", "scene_type",
                    "floor_plan_centroid", "floor_plan_vertices", "floor_plan_faces"]:
            if key in data:
                result[key] = data[key]

        return result

    except Exception as e:
        print(f"[ERROR] : {npz_path}, {e}")
        return None


def load_door_win_json(json_path: Path) -> Optional[Dict]:
   
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] 加载 door_win.json 失败: {json_path}, {e}")
        return None


def load_supervised_labels(json_path: Path) -> Optional[Dict]:
    
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] : {json_path}, {e}")
        return None


def extract_furniture_categories(class_labels: np.ndarray, class_names=None) -> List[str]:
 
    indices = np.argmax(class_labels, axis=1)
    names = BEDROOM_CATEGORIES if class_names is None else class_names
    if class_labels.shape[1] != len(names):
        raise ValueError("Furniture category width does not match the room vocabulary")
    categories = [names[i] for i in indices]
    if any(name in ("start", "end") for name in categories):
        raise ValueError("Sentinel tokens must be removed before supervised graph construction")
    return categories


def check_supervision_consistency(boxes_data: Dict, sup_labels: Dict) -> Tuple[bool, str]:
   
    N = boxes_data["N"]
    train_targets = sup_labels.get("train_targets", {})

    for key in ["g_gt_raw", "g_gt_step", "g_gt_normalized"]:
        if key in train_targets:
            g_gt = np.array(train_targets[key])
            if g_gt.shape != (N, 7):
                return False, f"{key} shape  ({N}, 7),  {g_gt.shape}"

    if "changed_mask" in train_targets:
        changed_mask = np.array(train_targets["changed_mask"])
        if changed_mask.shape != (N,):
            return False, f"changed_mask shape  {changed_mask.shape}"

    head_gt = train_targets.get("head_gt", -1)
    if head_gt not in {0, 1, 2, 3, -1}:
        return False, f"head_gt: {head_gt}"

    if "g_gt_step" in train_targets:
        g_gt_step = np.array(train_targets["g_gt_step"], dtype=np.float32)
        max_vals = np.abs(g_gt_step).max(axis=0)
        if np.any(max_vals > ACTOR_SCALE + 1e-3):
            return False, f"g_gt_step ACTOR_SCALE: max={max_vals}, scale={ACTOR_SCALE}"

    if "g_gt_normalized" in train_targets:
        g_gt_norm = np.array(train_targets["g_gt_normalized"], dtype=np.float32)
        if np.any(np.abs(g_gt_norm) > 1.0 + 1e-3):
            return False, f"g_gt_normalized [-1,1]: max={np.abs(g_gt_norm).max()}"

  
    if head_gt >= 0 and "g_gt_step" in train_targets:
        g_gt_step = np.array(train_targets["g_gt_step"], dtype=np.float32)
        head_mask = np.array(HEAD_MASKS[head_gt])

        for i in range(N):
            nonzero_dims = np.abs(g_gt_step[i]) > 1e-6
            uncovered = nonzero_dims & (~head_mask.astype(bool))
            if uncovered.any():
                return False, f" {i}: head_gt={head_gt}  {np.where(uncovered)[0]}"

    return True, ""


#

class SupervisedDataset2(Dataset):
    

    def __init__(
        self,
        perturbed_data_dir: Path,
        labels_dir: Path,
        original_dir: Path,
        room_type: str = "bedroom",
        radius: float = 1.6,
        dw_radius: float = 3.0,
        check_consistency: bool = True,
        filter_invalid: bool = True,
        sample_uids_filter: Optional[List[str]] = None,  
    ):
        super().__init__()
        self.perturbed_data_dir = Path(perturbed_data_dir)
        self.labels_dir = Path(labels_dir)
        self.original_dir = Path(original_dir)
        self.room_type = room_type
        stats_root = self.original_dir if (self.original_dir / "dataset_stats.txt").exists() else original_root(room_type)
        self.class_names = load_class_names(stats_root)
        graph_classes = sorted(name for name in self.class_names if name not in ("start", "end")) + ["door", "window"]
        self.cat_to_idx = {name: index for index, name in enumerate(graph_classes)}
        self.radius = radius
        self.dw_radius = dw_radius
        self.check_consistency = check_consistency
        self.filter_invalid = filter_invalid

        
        if not self.labels_dir.exists():
            raise ValueError(f"标签目录不存在: {self.labels_dir}")

        if sample_uids_filter is not None:
            all_label_dirs = [self.labels_dir / uid for uid in sample_uids_filter if (self.labels_dir / uid).is_dir()]
            print(f"[SupervisedDataset2] 使用过滤列表: {len(sample_uids_filter)} ")
        else:
            all_label_dirs = sorted([d for d in self.labels_dir.iterdir() if d.is_dir()])

        
        self.valid_samples = []
        self.invalid_samples = []

        for label_dir in all_label_dirs:
            sample_uid = label_dir.name
            perturbed_sample_dir = self.perturbed_data_dir / sample_uid

         
            source_uid = sample_uid.split("__")[0]
            original_sample_dir = self.original_dir / source_uid

            if self._is_valid_sample(sample_uid, perturbed_sample_dir, label_dir, original_sample_dir):
                self.valid_samples.append(sample_uid)
            else:
                self.invalid_samples.append(sample_uid)

        print(f"[SupervisedDataset2] : {len(all_label_dirs)}, "
              f"{len(self.valid_samples)}, "
              f" {len(self.invalid_samples)}")

    def _is_valid_sample(self, sample_uid: str, perturbed_dir: Path, label_dir: Path, original_dir: Path) -> bool:
        
        if not self.filter_invalid:
            return True

        
        perturbed_boxes = perturbed_dir / "boxes.npz"
        sup_path = label_dir / "perturbation_meta_supervised.json"

        
        original_boxes = original_dir / "boxes.npz"

        if not perturbed_boxes.exists():
            return False
        if not sup_path.exists():
            return False
        if not original_boxes.exists():
            return False

        perturbed_data = load_boxes_npz(perturbed_boxes)
        if perturbed_data is None:
            return False

        original_data = load_boxes_npz(original_boxes)
        if original_data is None:
            return False

        sup_labels = load_supervised_labels(sup_path)
        if sup_labels is None:
            return False

        
        train_targets = sup_labels.get("train_targets", {})
        if not train_targets.get("use_for_training", False):
            return False

        return True

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_uid = self.valid_samples[idx]

        source_uid = sample_uid.split("__")[0]

       
        perturbed_sample_dir = self.perturbed_data_dir / sample_uid
        label_sample_dir = self.labels_dir / sample_uid
        original_sample_dir = self.original_dir / source_uid

       
        perturbed_boxes_path = perturbed_sample_dir / "boxes.npz"
        perturbed_boxes_data = load_boxes_npz(perturbed_boxes_path)

        
        original_boxes_path = original_sample_dir / "boxes.npz"
        original_door_win_path = original_sample_dir / "door_win.json"

        original_boxes_data = load_boxes_npz(original_boxes_path)
        door_win_data = load_door_win_json(original_door_win_path)

      
        sup_path = label_sample_dir / "perturbation_meta_supervised.json"
        sup_labels = load_supervised_labels(sup_path)

        if perturbed_boxes_data is None:
            raise RuntimeError(f"{perturbed_boxes_path}")
        if original_boxes_data is None:
            raise RuntimeError(f"{original_boxes_path}")
        if sup_labels is None:
            raise RuntimeError(f" {sup_path}")

        
        if self.check_consistency:
            is_valid, error_msg = check_supervision_consistency(perturbed_boxes_data, sup_labels)
            if not is_valid:
                raise RuntimeError(f"[{sample_uid}]  {error_msg}")

    
        class_labels = original_boxes_data["class_labels"]  # [N, C]
        translations = original_boxes_data["translations"]  # [N, 3]
        sizes_half = original_boxes_data["sizes"]           # [N, 3] half-size
        angles = original_boxes_data["angles"]              # [N, 1]

        N = original_boxes_data["N"]

        # [class_onehot, x,y,z, cos,sin, w,d,h] ──
        yaw = angles[:, 0]  # [N]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        orientation = np.stack([cos_yaw, sin_yaw], axis=1)  # [N, 2]

        #  full-size: (w,d,h) = (2*half_X, 2*half_Z, 2*half_Y)
        # w=X, d=Z, h=Y
        full_size = np.stack([
            2 * sizes_half[:, 0],  # w = 2*half_X
            2 * sizes_half[:, 2],  # d = 2*half_Z
            2 * sizes_half[:, 1],  # h = 2*half_Y
        ], axis=1)  # [N, 3]

        node_layout = np.concatenate([
            class_labels,   # [N, C]
            translations,   # [N, 3]
            orientation,    # [N, 2]
            full_size,      # [N, 3]
        ], axis=1)  # [N, C+8]

     
        categories = extract_furniture_categories(class_labels, self.class_names)

   
        positions = translations  # [N, 3]
        sizes_for_graph = full_size  # [N, 3] 使用 full-size
        thetas = orientation  # [N, 2]

        floor_plan_centroid = original_boxes_data.get("floor_plan_centroid", np.array([0, 0, 0], dtype=np.float32))

        dw_categories, dw_positions, dw_sizes, dw_thetas = extract_door_window_nodes(
            door_win_data,
            floor_plan_centroid,
            room_type=self.room_type
        )

      
        all_categories, all_positions, all_sizes, all_thetas, num_furniture = \
            combine_furniture_and_door_window(
                categories, positions, sizes_for_graph, thetas,
                dw_categories, dw_positions, dw_sizes, dw_thetas
            )

        num_door_window = len(dw_categories)

        try:
            global build_scene_graph
            if build_scene_graph is None:
                build_scene_graph = _import_graph_builder()

            graph_node_feats, graph_edge_index, graph_edge_feats, graph_edge_types = \
                build_scene_graph(
                    categories=all_categories,  # 
                    cat_to_idx=self.cat_to_idx,
                    positions=all_positions,
                    sizes=all_sizes,
                    thetas=all_thetas,
                    radius=self.radius,
                    num_furniture=num_furniture,  # 
                    dw_radius=self.dw_radius,
                )
        except Exception as e:
            raise RuntimeError(f"[{sample_uid}] : {e}")

  
        num_graph_nodes = graph_node_feats.shape[0]
        furniture_mask = np.zeros(num_graph_nodes, dtype=bool)
        furniture_mask[:num_furniture] = True  # 

        vnode_idx = num_furniture + num_door_window  # 

        train_targets = sup_labels["train_targets"]

        head_gt = train_targets["head_gt"]
        g_gt_raw = np.array(train_targets["g_gt_raw"], dtype=np.float32)
        g_gt_step = np.array(train_targets["g_gt_step"], dtype=np.float32)
        g_gt_normalized = np.array(train_targets["g_gt_normalized"], dtype=np.float32)
        changed_mask = np.array(train_targets["changed_mask"], dtype=np.int64)

        policy_valid = (
            sup_labels.get("valid", False)
            and train_targets.get("use_for_training", False)
            and head_gt in {0, 1, 2, 3}
        )

        actor_valid = (
            policy_valid
            and g_gt_step.shape == (N, 7)
            and changed_mask.shape == (N,)
        )

        warnings = train_targets.get("warnings", [])

        edge_bert_path = perturbed_sample_dir / "edge_bert.pt"
        if not edge_bert_path.exists():
            edge_bert_path = label_sample_dir / "edge_bert.pt"

        edge_bert = None
        if edge_bert_path.exists():
            try:
                edge_bert = torch.load(edge_bert_path, map_location="cpu")
             
                E = graph_edge_index.shape[1]
                if edge_bert.shape[0] != E:
                    print(f"[WARNING] {sample_uid}: edge_bert.shape[0]={edge_bert.shape[0]} != E={E}，")
                    edge_bert = None
            except Exception as e:
                print(f"[WARNING] {sample_uid}:  edge_bert.pt : {e}")
                edge_bert = None

        return {
            # Identification
            "sample_uid": sample_uid,
            "source_uid": source_uid,
            "combo_name": sup_labels.get("combo_name", ""),
            "num_objects": N,

            # Raw furniture state (from original unperturbed scene)
            "class_labels": class_labels,
            "translations": translations,
            "sizes_half": sizes_half,
            "angles": angles,
            "node_layout": node_layout,

            # Graph (built from original unperturbed scene)
            "graph_node_feats": graph_node_feats,
            "graph_edge_index": graph_edge_index,
            "graph_edge_feats": graph_edge_feats,
            "graph_edge_types": graph_edge_types,
            "edge_bert": edge_bert, 
            "furniture_mask": furniture_mask,
            "vnode_idx": vnode_idx,
            "num_furniture": N,
            "num_door_win": num_door_window,

            # Policy supervision (from labels)
            "head_gt": head_gt,
            "policy_target": head_gt,

            # Actor supervision (from labels, relative to perturbed scene)
            "g_gt_raw": g_gt_raw,
            "g_gt_step": g_gt_step,
            "g_gt_normalized": g_gt_normalized,
            "changed_mask": changed_mask,

            # Validity flags
            "policy_valid": policy_valid,
            "actor_valid": actor_valid,
            "warnings": warnings,
        }

    def get_invalid_samples(self):
       
        return self.invalid_samples


class SupervisedDataset2Fixed(Dataset):
  
    def __init__(
        self,
        perturbed_data_dir: Path,
        labels_dir: Path,
        original_dir: Path,
        room_type: str = "bedroom",
        radius: float = 1.6,
        dw_radius: float = 3.0,
        check_consistency: bool = True,
        filter_invalid: bool = True,
        sample_uids_filter: Optional[List[str]] = None,
    ):
        super().__init__()
        self.perturbed_data_dir = Path(perturbed_data_dir)
        self.labels_dir = Path(labels_dir)
        self.original_dir = Path(original_dir)
        self.room_type = room_type
        stats_root = self.original_dir if (self.original_dir / "dataset_stats.txt").exists() else original_root(room_type)
        self.class_names = load_class_names(stats_root)
        graph_classes = sorted(name for name in self.class_names if name not in ("start", "end")) + ["door", "window"]
        self.cat_to_idx = {name: index for index, name in enumerate(graph_classes)}
        self.radius = radius
        self.dw_radius = dw_radius
        self.check_consistency = check_consistency
        self.filter_invalid = filter_invalid

        if not self.labels_dir.exists():
            raise ValueError(f"{self.labels_dir}")

        if sample_uids_filter is not None:
            all_label_dirs = [
                self.labels_dir / uid
                for uid in sample_uids_filter
                if (self.labels_dir / uid).is_dir()
            ]
            print(f"[SupervisedDataset2Fixed] : {len(sample_uids_filter)} ")
        else:
            all_label_dirs = sorted([d for d in self.labels_dir.iterdir() if d.is_dir()])

        self.valid_samples = []
        self.invalid_samples = []

        for label_dir in all_label_dirs:
            sample_uid = label_dir.name
            perturbed_sample_dir = self.perturbed_data_dir / sample_uid
            source_uid = sample_uid.split("__")[0]
            original_sample_dir = self.original_dir / source_uid

            if self._is_valid_sample(sample_uid, perturbed_sample_dir, label_dir, original_sample_dir):
                self.valid_samples.append(sample_uid)
            else:
                self.invalid_samples.append(sample_uid)

        print(
            f"[SupervisedDataset2Fixed] : {len(all_label_dirs)}, "
            f"{len(self.valid_samples)}, "
            f"{len(self.invalid_samples)}"
        )

    def _is_valid_sample(
        self,
        sample_uid: str,
        perturbed_dir: Path,
        label_dir: Path,
        original_dir: Path,
    ) -> bool:
       
        if not self.filter_invalid:
            return True

        perturbed_boxes = perturbed_dir / "boxes.npz"
        sup_path = label_dir / "perturbation_meta_supervised.json"

        if not perturbed_boxes.exists():
            return False
        if not sup_path.exists():
            return False

        perturbed_data = load_boxes_npz(perturbed_boxes)
        if perturbed_data is None:
            return False

        sup_labels = load_supervised_labels(sup_path)
        if sup_labels is None:
            return False

        train_targets = sup_labels.get("train_targets", {})
        if not train_targets.get("use_for_training", False):
            return False

        return True

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_uid = self.valid_samples[idx]
        source_uid = sample_uid.split("__")[0]

        perturbed_sample_dir = self.perturbed_data_dir / sample_uid
        label_sample_dir = self.labels_dir / sample_uid
        original_sample_dir = self.original_dir / source_uid

        perturbed_boxes_path = perturbed_sample_dir / "boxes.npz"
        perturbed_door_win_path = perturbed_sample_dir / "door_win.json"
        original_door_win_path = original_sample_dir / "door_win.json"
        sup_path = label_sample_dir / "perturbation_meta_supervised.json"

        perturbed_boxes_data = load_boxes_npz(perturbed_boxes_path)
        sup_labels = load_supervised_labels(sup_path)

        if perturbed_boxes_data is None:
            raise RuntimeError(f"{perturbed_boxes_path}")
        if sup_labels is None:
            raise RuntimeError(f"{sup_path}")

        if self.check_consistency:
            is_valid, error_msg = check_supervision_consistency(perturbed_boxes_data, sup_labels)
            if not is_valid:
                raise RuntimeError(f"[{sample_uid}]  {error_msg}")

        class_labels = perturbed_boxes_data["class_labels"]
        translations = perturbed_boxes_data["translations"]
        sizes_half = perturbed_boxes_data["sizes"]
        angles = perturbed_boxes_data["angles"]
        N = perturbed_boxes_data["N"]

        yaw = angles[:, 0]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        orientation = np.stack([cos_yaw, sin_yaw], axis=1)

        # full_size 
        full_size = np.stack([
            2 * sizes_half[:, 0],
            2 * sizes_half[:, 2],
            2 * sizes_half[:, 1],
        ], axis=1)

        node_layout = np.concatenate([
            class_labels,
            translations,
            orientation,
            full_size,
        ], axis=1)

      
        categories = extract_furniture_categories(class_labels, self.class_names)
        positions = translations
        sizes_for_graph = np.stack([
            sizes_half[:, 0],  # half_X → w
            sizes_half[:, 2],  # half_Z → d
            sizes_half[:, 1],  # half_Y → h
        ], axis=1)
        thetas = orientation

        door_win_data = load_door_win_json(perturbed_door_win_path)
        if door_win_data is None:
            door_win_data = load_door_win_json(original_door_win_path)

        floor_plan_centroid = perturbed_boxes_data.get(
            "floor_plan_centroid",
            np.array([0, 0, 0], dtype=np.float32),
        )

        dw_categories, dw_positions, dw_sizes, dw_thetas = extract_door_window_nodes(
            door_win_data,
            floor_plan_centroid,
            room_type=self.room_type
        )

        all_categories, all_positions, all_sizes, all_thetas, num_furniture = \
            combine_furniture_and_door_window(
                categories, positions, sizes_for_graph, thetas,
                dw_categories, dw_positions, dw_sizes, dw_thetas
            )

        num_door_window = len(dw_categories)

        try:
            global build_scene_graph
            if build_scene_graph is None:
                build_scene_graph = _import_graph_builder()

            graph_node_feats, graph_edge_index, graph_edge_feats, graph_edge_types = \
                build_scene_graph(
                    categories=all_categories,
                    cat_to_idx=self.cat_to_idx,
                    positions=all_positions,
                    sizes=all_sizes,
                    thetas=all_thetas,
                    radius=self.radius,
                    num_furniture=num_furniture,
                    dw_radius=self.dw_radius,
                )
        except Exception as e:
            raise RuntimeError(f"[{sample_uid}]  {e}")

        num_graph_nodes = graph_node_feats.shape[0]
        furniture_mask = np.zeros(num_graph_nodes, dtype=bool)
        furniture_mask[:num_furniture] = True
        vnode_idx = num_furniture + num_door_window

        train_targets = sup_labels["train_targets"]
        head_gt = train_targets["head_gt"]
        g_gt_raw = np.array(train_targets["g_gt_raw"], dtype=np.float32)
        g_gt_step = np.array(train_targets["g_gt_step"], dtype=np.float32)
        g_gt_normalized = np.array(train_targets["g_gt_normalized"], dtype=np.float32)
        changed_mask = np.array(train_targets["changed_mask"], dtype=np.int64)

        policy_valid = (
            sup_labels.get("valid", False)
            and train_targets.get("use_for_training", False)
            and head_gt in {0, 1, 2, 3}
        )
        actor_valid = (
            policy_valid
            and g_gt_step.shape == (N, 7)
            and changed_mask.shape == (N,)
        )
        warnings = train_targets.get("warnings", [])

        edge_bert_path = perturbed_sample_dir / "edge_bert.pt"
        if not edge_bert_path.exists():
            edge_bert_path = label_sample_dir / "edge_bert.pt"

        edge_bert = None
        if edge_bert_path.exists():
            try:
                edge_bert = torch.load(edge_bert_path, map_location="cpu")
                E = graph_edge_index.shape[1]
                if edge_bert.shape[0] != E:
                    print(
                        f"[WARNING] {sample_uid}: edge_bert.shape[0]={edge_bert.shape[0]} != E={E}"
                    )
                    edge_bert = None
            except Exception as e:
                print(f"[WARNING] {sample_uid}: {e}")
                edge_bert = None

        return {
            "sample_uid": sample_uid,
            "source_uid": source_uid,
            "combo_name": sup_labels.get("combo_name", ""),
            "num_objects": N,
            "class_labels": class_labels,
            "translations": translations,
            "sizes_half": sizes_half,
            "angles": angles,
            "node_layout": node_layout,
            "graph_node_feats": graph_node_feats,
            "graph_edge_index": graph_edge_index,
            "graph_edge_feats": graph_edge_feats,
            "graph_edge_types": graph_edge_types,
            "edge_bert": edge_bert,
            "furniture_mask": furniture_mask,
            "vnode_idx": vnode_idx,
            "num_furniture": N,
            "num_door_win": num_door_window,
            "head_gt": head_gt,
            "policy_target": head_gt,
            "g_gt_raw": g_gt_raw,
            "g_gt_step": g_gt_step,
            "g_gt_normalized": g_gt_normalized,
            "changed_mask": changed_mask,
            "policy_valid": policy_valid,
            "actor_valid": actor_valid,
            "warnings": warnings,
        }

    def get_invalid_samples(self):
       
        return self.invalid_samples


if __name__ == "__main__":

    PERTURBED_DATA_DIR = Path("SAGC/GNN/goal/collect_data/more_correct/correct_data")
    LABELS_DIR = Path("SAGC/GNN/goal/collect_data/more_correct/correct_labels")
    ORIGINAL_DIR = Path("SAGC/data/3d_front_processed/bedrooms_objfeats_32_64")

    print(f"SupervisedDataset2")
    print(f"  perturbed_data_dir: {PERTURBED_DATA_DIR}")
    print(f"  labels_dir:         {LABELS_DIR}")
    print(f"  original_dir:       {ORIGINAL_DIR}")

    dataset = SupervisedDataset2(PERTURBED_DATA_DIR, LABELS_DIR, ORIGINAL_DIR)

    print(f"\n{len(dataset)}")
    print(f"{len(dataset.get_invalid_samples())}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n:")
        print(f"  sample_uid     = {sample['sample_uid']}")
        print(f"  source_uid     = {sample['source_uid']}")
        print(f"  combo_name     = {sample['combo_name']}")
        print(f"  num_objects    = {sample['num_objects']}")
        print(f"  node_layout    = {sample['node_layout'].shape}")
        print(f"  graph_node_feats = {sample['graph_node_feats'].shape}")
        print(f"  graph_edge_index = {sample['graph_edge_index'].shape}")
        print(f"  head_gt        = {sample['head_gt']}")
        print(f"  g_gt_step      = {sample['g_gt_step'].shape}")
        print(f"  changed_mask   = {sample['changed_mask'].shape}")
        print(f"  policy_valid   = {sample['policy_valid']}")
        print(f"  actor_valid    = {sample['actor_valid']}")
