

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

class DDPMInterface:


    def __init__(
        self,
        config_path: str | Path,
        weight_file: str | Path,
        device: str | torch.device = "cuda",
        floor_plan_textures: Optional[str | Path] = None,
        scene_splits: Optional[list[str]] = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.weight_file = Path(weight_file).resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly for CPU checks")
       
        from training_utils import load_config
        from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
        from scene_synthesis.networks import build_network

        self.config = load_config(str(self.config_path))
        self.network_config = self.config["network"]
        from evaluation.layout_schema import LayoutSpec
        self.layout_spec = LayoutSpec.from_network(self.network_config)
        self.room_type = self.config['data']['filter_fn'].replace('threed_front_', '', 1)

        data_cfg = dict(self.config["data"])
      
        if "text" in data_cfg.get("encoding_type", "") and "textfix" not in data_cfg["encoding_type"]:
            data_cfg["encoding_type"] = data_cfg["encoding_type"].replace("text", "textfix")
        if "no_prm" not in data_cfg.get("encoding_type", ""):
            data_cfg["encoding_type"] += "_no_prm"

        split = scene_splits or self.config.get("validation", {}).get("splits", ["test"])
        self.raw_dataset, self.dataset = get_dataset_raw_and_encoded(
            data_cfg,
            filter_fn=filter_function(data_cfg, split=split),
            split=split,
        )
        self._scene_index = {tag.split("_", 1)[1]: i for i, tag in enumerate(self.raw_dataset._tags)}
        self.network, _, _ = build_network(
            self.dataset.feature_size,
            self.dataset.n_classes,
            self.config,
            str(self.weight_file),
            device=self.device,
        )
        self.network.eval()
        self.floor_plan_textures = str(floor_plan_textures) if floor_plan_textures else None

    def scene_context(self, scene_id: str, index: Optional[int] = None) -> Dict[str, Any]:
     
        import importlib.util
        from pathlib import Path

        utils_path = Path(__file__).resolve().parents[1] / "scripts" / "utils.py"
        spec = importlib.util.spec_from_file_location("_scripts_utils_floor_plan", utils_path)
        scripts_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scripts_utils)
        floor_plan_from_scene = scripts_utils.floor_plan_from_scene

        if index is None:
            index = self._scene_index.get(str(scene_id))
        if index is None:
            raise KeyError(f"scene_id not found in validation dataset: {scene_id}")
        raw_scene = self.raw_dataset[index]
        sample = self.dataset[index]
        _, _, room_mask = floor_plan_from_scene(
            raw_scene,
            self.floor_plan_textures,
            no_texture=True,
        )
        text = sample.get("description")
       
        faces = getattr(raw_scene, "floor_plan", (None, None))[1]
        floor_plan = None
        floor_plan_centroid = None

        if vertices is not None:
           
            vertices_tensor = torch.as_tensor(vertices, dtype=torch.float32)
            room_diag = float(torch.linalg.vector_norm(vertices_tensor[:, [0, 2]].amax(0) - vertices_tensor[:, [0, 2]].amin(0)).item())

         
            vertices_np = np.array(vertices, dtype=np.float32)
            faces_np = np.array(faces, dtype=np.int32) if faces is not None else np.array([], dtype=np.int32)
            floor_plan = [(vertices_np, faces_np)]  

           
            centroid = np.asarray(raw_scene.floor_plan_centroid, dtype=np.float32)
            floor_plan_centroid = np.expand_dims(centroid, axis=0)  # [1, 3] numpy array
        else:
            room_diag = 1.0

        room_height = 1.0
        bboxes = getattr(raw_scene, "bboxes", [])
        if bboxes:
            heights = [float(getattr(box, "size", [0.0, 0.0, 0.0])[1]) for box in bboxes]
            room_height = max(max(heights, default=0.0), 1.0)
        return {
            "room_type": self.room_type,
            "layout_spec": self.layout_spec,
            "class_names": list(self.dataset.class_labels),
            "scene_id": str(raw_scene.scene_id),
            "scene_index": int(index),
            "raw_scene": raw_scene,
            "sample": sample,
            "room_mask": room_mask.to(self.device),
            "text": text,
            "room_diag": max(room_diag, 1e-3),
            "room_height": room_height,
            "floor_plan": floor_plan,  
            "floor_plan_centroid": floor_plan_centroid,  
        }

    def _condition(self, context: Dict[str, Any]):
        
        cached = context.get("_ddpm_condition")
        if cached is not None:
            return cached
        room_mask = context["room_mask"]
        if room_mask.dim() == 3:
            room_mask = room_mask.unsqueeze(0)
        batch_size = room_mask.size(0)
        num_points = int(self.network_config["sample_num_points"])
        room_feature = self.network.fc_room_f(self.network.feature_extractor(room_mask))

        if self.network.instance_condition:
            if self.network.learnable_embedding:
                indices = torch.arange(num_points, device=self.device)[None, :].repeat(batch_size, 1)
                instance_feature = self.network.positional_embedding[indices, :]
            else:
                eye = torch.eye(num_points, device=self.device)[None].repeat(batch_size, 1, 1)
                instance_feature = self.network.fc_instance_condition(eye)
            condition = torch.cat([room_feature[:, None, :].repeat(1, num_points, 1), instance_feature], dim=-1)
        else:
            condition = room_feature[:, None, :].repeat(1, num_points, 1)

        condition_cross = None
        if self.network.text_condition:
            text = context.get("text") or ""
            if self.network.text_glove_embedding:
                condition_cross = self.network.fc_text_f(context["text"])
            elif self.network.text_clip_embedding:
                tokens = self.network.clip.tokenize(text).to(self.device)
                condition_cross = self.network.clip_model.encode_text(tokens)
            else:
                tokens = self.network.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
                text_feature = self.network.bertmodel(**tokens).last_hidden_state
                condition_cross = self.network.fc_text_f(text_feature)
        context["_ddpm_condition"] = (condition, condition_cross)
        return condition, condition_cross

    @torch.no_grad()
    def initial_state(self, context: Dict[str, Any], seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            noise = torch.randn(
                (1, int(self.network_config["sample_num_points"]), int(self.network_config["point_dim"])),
                generator=generator,
                device=self.device,
            )
        else:
            noise = torch.randn(
                (1, int(self.network_config["sample_num_points"]), int(self.network_config["point_dim"])),
                device=self.device,
            )
        return noise

    @torch.no_grad()
    def step(
        self,
        x_t: torch.Tensor,
        timestep: int,
        context: Dict[str, Any],
        clip_denoised: bool = True,
        noise: Optional[torch.Tensor] = None,
        posterior_x0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
      
        condition, condition_cross = self._condition(context)
        t = torch.full((x_t.shape[0],), int(timestep), dtype=torch.long, device=x_t.device)
        gd = self.network.diffusion.diffusion
        model_mean, variance, log_variance, pred_xstart = gd.p_mean_variance(
            self.network.diffusion._denoise,
            data=x_t,
            t=t,
            condition=condition,
            condition_cross=condition_cross,
            clip_denoised=clip_denoised,
            return_pred_xstart=True,
        )
        if posterior_x0 is not None:
            model_mean, _, _ = gd.q_posterior_mean_variance(posterior_x0, x_t, t)
        if noise is None:
            noise = torch.randn_like(x_t)
        nonzero = (1.0 - (t == 0).float()).reshape((x_t.shape[0],) + (1,) * (x_t.dim() - 1))
        x_prev = model_mean + nonzero * torch.exp(0.5 * log_variance) * noise
        return {
            "x_prev": x_prev,
            "x_t": x_t,
            "pred_xstart": pred_xstart,
            "model_mean": model_mean,
            "variance": variance,
            "log_variance": log_variance,
        }

    @torch.no_grad()
    def sample(
        self,
        context: Dict[str, Any],
        seed: Optional[int] = None,
        clip_denoised: bool = True,
        keep_trace: bool = True,
    ) -> Dict[str, Any]:
       
        x_t = self.initial_state(context, seed=seed)
        trace = []
        for timestep in reversed(range(int(self.network.diffusion.diffusion.num_timesteps))):
            result = self.step(x_t, timestep, context, clip_denoised=clip_denoised)
            if keep_trace:
                trace.append({
                    "t": timestep,
                    "x_t": x_t.detach().cpu(),
                    "pred_xstart": result["pred_xstart"].detach().cpu(),
                    "x_prev": result["x_prev"].detach().cpu(),
                })
            x_t = result["x_prev"]
        return {"layout": x_t, "trace": trace, "context": context}
