

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import yaml

_HERE = Path(__file__).resolve()
_COND = _HERE.parents[1] / "GNN" / "iteration" / "factor_aware" / "conditional_reasoning"
_ITER = _COND.parents[1]
for path in (_ITER, _COND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collate_cached_geometry import collate_cached_geometry_batch, move_batch_to_device  
from correction_losses import SCALE, to_physical_step  
from model import ConditionalReasoningModel  
DEFAULT_BERT_PATH = _COND.parents[3] / "hf_models" / "bert"
DEFAULT_NODE_CACHE_DIR = _COND.parents[3] / "GNN" / "adapters" / "cache"


INFERENCE_BATCH_KEYS = (
    "has_geometry",
    "graph_node_feats", "graph_edge_index", "graph_edge_feats", "graph_edge_types",
    "vnode_indices", "furniture_node_ranges", "scene_furniture_counts",
    "geo_node_position", "geo_node_yaw", "geo_node_full_wdh", "geo_node_type",
    "geo_to_old_index", "geo_node_scene_index", "geo_furniture_index",
    "geo_edge_index", "geo_edge_scalar", "geo_edge_vector_basis",
    "geo_edge_type", "boundary_length",
    "boundary_scalar", "boundary_vector_basis", "boundary_target_index",
    "boundary_scene_index", "room_diag_D", "room_height_y",
)


def validate_inference_batch(batch: Dict[str, Any]) -> None:
    """Fail early when a dynamic builder omitted a model-forward field."""
    missing = [key for key in INFERENCE_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError("query-A inference batch : " + ", ".join(missing))
    if not bool(batch["has_geometry"]):
        raise ValueError("query-A  batch['has_geometry']=True")


class QueryAHardInterface:
    

    def __init__(
        self,
        config_path: str | Path = _COND / "configs/conditional_reasoning.yaml",
        checkpoint_path: str | Path = _COND / "checkpoints/best.pt",
        thresholds_path: str | Path = _COND / "checkpoints/best_thresholds.json",
        device: str | torch.device = "cuda",
        bert_path: str | Path = DEFAULT_BERT_PATH,
        online_edge_bert: bool = True,
        node_cache_dir: str | Path = DEFAULT_NODE_CACHE_DIR,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.thresholds_path = Path(thresholds_path).resolve()
        self.bert_path = Path(bert_path).resolve()
        self.node_cache_dir = Path(node_cache_dir).resolve()
        self.online_edge_bert = bool(online_edge_bert)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.model = ConditionalReasoningModel(
            m3_checkpoint_path=self.config["paths"]["m3_checkpoint"],
            semantic_checkpoint_path=self.config["paths"]["semantic_checkpoint"],
            **self.config["model"],
        ).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()
        actual_cache_dir = Path(self.model.context_adapter.semantic_branch.adapter.cache_dir).resolve()
        if actual_cache_dir != self.node_cache_dir:
            raise RuntimeError(
                ": "
                f"期望 {self.node_cache_dir}, 实际 {actual_cache_dir}"
            )
        self.room_type = self.model.context_adapter.semantic_branch.adapter.room_type
        cache_file = actual_cache_dir / f"node_bert_{self.room_type}.pt"
        if not cache_file.exists():
            raise FileNotFoundError(f": {cache_file}")
        print(f"[QueryAHardInterface] node BERT cache: {cache_file}")
        if self.online_edge_bert:
            self._enable_online_edge_bert()
        with self.thresholds_path.open("r", encoding="utf-8") as handle:
            self.thresholds = json.load(handle)
        for factor in ('position', 'size', 'rotation'):
            if not 0.0 < float(self.thresholds[factor]) < 1.0:
                raise ValueError(f"Invalid hard threshold for {factor}")

    def _enable_online_edge_bert(self) -> None:
        
        edge_projector = self.model.context_adapter.semantic_branch.adapter.edge_proj
        edge_projector.use_cache = False
        edge_projector.build_bert(str(self.bert_path), device=str(self.device))

    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run hard gates. ``batch`` must be collated with the GNN helper."""
        validate_inference_batch(batch)
        if self.online_edge_bert:
            # Never allow a stale precomputed edge_bert to bypass live BERT.
            batch = dict(batch)
            batch.pop("edge_bert", None)
        batch = move_batch_to_device(batch, self.device)
        result = self.model.predict_with_gates(
            batch,
            position_threshold=float(self.thresholds["position"]),
            size_threshold=float(self.thresholds["size"]),
            rotation_threshold=float(self.thresholds["rotation"]),
        )
        normalized = result["combined_7d"]
        result["normalized_7d"] = normalized
        result["physical_7d"] = to_physical_step(normalized)
        return result

    def correction(
        self,
        x0: torch.Tensor,
        scene_context: Dict[str, Any],
        batch_builder: Callable[[torch.Tensor, Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Build a current-state graph and predict query-A hard correction.

        ``batch_builder`` is intentionally required: geometry edges and wall
        relations depend on the current denoised state and cannot be inferred
        safely from a stale cache alone.
        """
        batch = batch_builder(x0, scene_context)
        if not isinstance(batch, dict):
            raise TypeError(f"batch_builder must return dict, got {type(batch)!r}")
        # Prefer exact cache scales whenever the builder provides them.
        if isinstance(batch.get("room_diag_D"), torch.Tensor) and batch["room_diag_D"].numel():
            scene_context["room_diag"] = float(batch["room_diag_D"].reshape(-1)[0].detach().cpu())
        if isinstance(batch.get("room_height_y"), torch.Tensor) and batch["room_height_y"].numel():
            scene_context["room_height"] = float(batch["room_height_y"].reshape(-1)[0].detach().cpu())
        result = self.predict(batch)
        result["slot_indices"] = batch["ddpm_slot_indices"].detach().cpu()
        return result

    @staticmethod
    def normalized_to_diffusion_delta(
        physical_7d: torch.Tensor,
        room_diag: float = 1.0,
        room_height: float = 1.0,
        diffusion_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
       
        if physical_7d.ndim != 2 or physical_7d.size(-1) != 7:
            raise ValueError(f"physical_7d must be [F,7], got {tuple(physical_7d.shape)}")
        if diffusion_scales is None:
            scales = physical_7d.new_tensor([room_diag, room_height, room_diag, room_diag, room_diag, room_height, 1.0])
        else:
            scales = torch.as_tensor(diffusion_scales, dtype=physical_7d.dtype, device=physical_7d.device)
            if scales.numel() != 7:
                raise ValueError("diffusion_scales must contain 7 values")
        return physical_7d / scales.clamp_min(1e-6)


def static_batch_builder(path: str | Path) -> Callable[[torch.Tensor, Dict[str, Any]], Dict[str, Any]]:
    
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "batch_size" in payload:
        batch = payload
    elif isinstance(payload, dict):
        batch = collate_cached_geometry_batch([payload])
    elif isinstance(payload, (list, tuple)):
        batch = collate_cached_geometry_batch(list(payload))
    else:
        raise TypeError(f"unsupported static batch payload: {type(payload)!r}")

    def _builder(_x0: torch.Tensor, _context: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    return _builder
