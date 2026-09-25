

from __future__ import annotations

import argparse
import hashlib
import numpy as np
import importlib
import json
import random
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import yaml
from tqdm import tqdm

from ddpm_interface import DDPMInterface
from query_a_hard_interface import QueryAHardInterface, static_batch_builder
from physcene_interface import PhySceneInterface
from combined_dynamic_builder import AllSlotsEmpty
from evaluation.layout_schema import DatasetBounds, DEFAULT_LAYOUT_SPEC, apply_query_correction

_COND_ROOT = Path(__file__).resolve().parents[1] / "GNN" / "iteration" / "factor_aware" / "conditional_reasoning"


def _load_callable(spec: str) -> Callable:
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("callable must use module:function syntax")
    return getattr(importlib.import_module(module_name), function_name)


def _scene_id_from_sample_dir(path: Path) -> str:
   
    name = path.name
    return name.split("_", 1)[1] if "_" in name else name


def _normalize_scene_id(scene_id: str) -> str:
   
    if "_" in scene_id and "-" in scene_id.split("_", 1)[0]:
        # Looks like UUID_SceneName format
        return scene_id.split("_", 1)[1]
    return scene_id


def select_scene_ids(
    sample_root: str | Path,
    scene_ids_file: str | Path,
    num_samples: int = 500,
    random_seed: int = 42,
    refresh: bool = False,
    use_json: bool = False,
    scene_ids_json: str | Path | None = None,
) -> list[tuple[str, str]]:
   
  
    if use_json and scene_ids_json:
        json_path = Path(scene_ids_json).expanduser().resolve()
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
           
            if isinstance(data, dict):
                ids = data.get("scene_ids", [])
            elif isinstance(data, list):
                ids = data
            else:
                raise ValueError(f"JSON file must contain a list or dict with 'scene_ids' key: {json_path}")

            if ids:
               
                result = [(_normalize_scene_id(sid), sid) for sid in ids]
                return result[:num_samples] if num_samples > 0 else result
        else:
            raise FileNotFoundError(f"JSON file not found: {json_path}")

  
    ids_path = Path(scene_ids_file).expanduser().resolve()
    if ids_path.exists() and not refresh:
        ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if num_samples > 0 and len(ids) < num_samples:
            raise ValueError(f"Requested {num_samples} distinct scenes but list only contains {len(ids)}")
        if len(set(_normalize_scene_id(sid) for sid in ids)) != len(ids):
            raise ValueError("Duplicate scene IDs in sampling list")
        if ids:
           
            result = [(_normalize_scene_id(sid), sid) for sid in ids]
            return result[:num_samples] if num_samples > 0 else result

    root = Path(sample_root).expanduser().resolve()
    candidates = sorted(
        _scene_id_from_sample_dir(path)
        for path in root.iterdir()
        if path.is_dir() and (path / "boxes.npz").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no sample directories containing boxes.npz under {root}")
    # Deduplicate IDs because the source UUID prefix is not part of scene_id.
    candidates = sorted(set(candidates))
    rng = random.Random(int(random_seed))
    if num_samples > 0 and len(candidates) > num_samples:
        candidates = rng.sample(candidates, num_samples)
        candidates.sort()
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text("\n".join(candidates) + "\n", encoding="utf-8")
    
    return [(sid, sid) for sid in candidates]


def _read_yaml(path: str | Path) -> dict:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"combined config must be a mapping: {path}")
    return value


class CombinedSampler:
    def __init__(
        self,
        ddpm: DDPMInterface,
        corrector: QueryAHardInterface,
        batch_builder: Callable[[torch.Tensor, Dict[str, Any]], Dict[str, Any]],
        physcene: Optional[PhySceneInterface] = None,
        guidance_start_t: int = 9,
        guidance_end_t: int = 0,
        guidance_strength: float = 1.0,
        guidance_interval: Optional[int] = None,
        guidance_steps: Optional[list[int]] = None,
        clip_denoised: bool = True,
        diffusion_scales: Optional[list[float]] = None,
        physcene_start_t: int = 9,
        physcene_end_t: int = 0,
        physcene_interval: Optional[int] = None,
        physcene_steps: Optional[list[int]] = None,
        dataset_bounds: Optional[DatasetBounds] = None,
    ) -> None:
        self.ddpm = ddpm
        self.layout_spec = getattr(ddpm, 'layout_spec', DEFAULT_LAYOUT_SPEC)
        self.corrector = corrector
        self.physcene = physcene  # 
        self.batch_builder = batch_builder
        self.guidance_start_t = int(guidance_start_t)
        self.guidance_end_t = int(guidance_end_t)
        self.guidance_strength = float(guidance_strength)
        self.guidance_interval = int(guidance_interval) if guidance_interval is not None else None
        self.guidance_steps = guidance_steps  #
        self.clip_denoised = clip_denoised
        self.diffusion_scales = diffusion_scales
        if diffusion_scales is not None and len(diffusion_scales) != 7:
            raise ValueError("diffusion_scales must contain exactly 7 comma-separated values")

     
        self.physcene_start_t = int(physcene_start_t)
        self.physcene_end_t = int(physcene_end_t)
        self.physcene_interval = int(physcene_interval) if physcene_interval is not None else None
        self.physcene_steps = physcene_steps
        self.dataset_bounds = dataset_bounds
        if self.dataset_bounds is None:
            raise ValueError("dataset_bounds is required for Query-A diffusion conversion")

    
        self._compute_guidance_steps()

    @staticmethod
    def _resolve_guidance_steps(
        start_t: int,
        end_t: int,
        interval: Optional[int],
        steps: Optional[list[int]],
        *,
        label: str,
    ) -> set[int]:
        
        if steps is not None:
           
            return set(int(s) for s in steps)
        if start_t < end_t:
            raise ValueError(f"{label}_start_t must be >= {label}_end_t")
        if interval is not None:
            
            if interval <= 0:
                raise ValueError(f"{label}_interval must be positive")
            return set(range(start_t, end_t - 1, -interval))
      
        return set(range(end_t, start_t + 1))

    def _compute_guidance_steps(self):
        
        self.active_guidance_steps = self._resolve_guidance_steps(
            self.guidance_start_t,
            self.guidance_end_t,
            self.guidance_interval,
            self.guidance_steps,
            label="guidance",
        )
        self.active_physcene_steps = self._resolve_guidance_steps(
            self.physcene_start_t,
            self.physcene_end_t,
            self.physcene_interval,
            self.physcene_steps,
            label="physcene",
        )

      
        self.last_guided_t = (
            min(self.active_guidance_steps) if self.active_guidance_steps else None
        )

    def should_apply_guidance(self, timestep: int) -> bool:
  
        return timestep in self.active_guidance_steps

    def should_apply_physcene(self, timestep: int) -> bool:
        
        return timestep in self.active_physcene_steps

    @torch.no_grad()
    def sample(self, context: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
       
        seed = int(seed) if seed is not None else int(torch.seed())
        step_generator = torch.Generator(device=self.ddpm.device).manual_seed(seed ^ 0x5DEECE66D)
        audit = dict(seed=seed, paired_step_noise=True, query_steps=[], query_nonzero_steps=[],
                     physcene_steps=[], physcene_nonzero_steps=[])
     
        x_t = self.ddpm.initial_state(context, seed=seed)
        x_t_raw = x_t.clone()  
        x_t_physcene = x_t.clone()  

        trace_corrected = []
        trace_raw = []
        trace_physcene = []

     
        x_t_at_fork = None  
       
        x_t_at_end = None

      
        self.skipped_guidance_steps = []

        total_steps = int(self.ddpm.network.diffusion.diffusion.num_timesteps)

        scene_id = context.get("scene_id", "unknown")
        pbar = tqdm(
            reversed(range(total_steps)),
            total=total_steps,
            desc=f"Sampling {scene_id}",
            unit="step",
            ncols=100,
        )

        for timestep in pbar:
            noise = torch.randn(x_t.shape, generator=step_generator, device=x_t.device, dtype=x_t.dtype)
          
            if timestep == self.guidance_start_t + 1:
                x_t_at_fork = x_t.clone()

          
            result = self.ddpm.step(x_t, timestep, context, clip_denoised=self.clip_denoised, noise=noise)
            pred_x0 = result["pred_xstart"]

            correction_record = None
            if self.should_apply_guidance(timestep):
              
                try:
                    correction_record = self.corrector.correction(
                        pred_x0, context, self.batch_builder
                    )
                except AllSlotsEmpty as exc:
                    self.skipped_guidance_steps.append(timestep)
                    tqdm.write(
                        f"  ! t={timestep}: 跳过 query-A 引导 —— {exc}"
                    )
                    correction_record = None

            if correction_record is not None:
                physical = correction_record["physical_7d"]
                slot_indices = correction_record.get("slot_indices")
                # Dynamic builder must provide the original DDPM slot mapping.
                # The fallback is deliberately rejected to prevent corrupting empty slots.
                if slot_indices is None:
                    raise RuntimeError("Query-A builder did not return slot_indices")
                application = apply_query_correction(
                    pred_x0, slot_indices, physical, self.dataset_bounds,
                    strength=self.guidance_strength,
                    spec=self.layout_spec,
                )
                corrected_x0 = application.corrected_x0
                audit["query_steps"].append(timestep)
                if torch.count_nonzero(corrected_x0 - pred_x0):
                    audit["query_nonzero_steps"].append(timestep)

              
                result = self.ddpm.step(
                    x_t,
                    timestep,
                    context,
                    clip_denoised=self.clip_denoised,
                    posterior_x0=corrected_x0,
                    noise=noise,
                )

            trace_corrected.append({
                "t": timestep,
                "x_t": x_t.detach().cpu(),
                "pred_xstart": pred_x0.detach().cpu(),
                "x_prev": result["x_prev"].detach().cpu(),
                "corrected": correction_record is not None,
                "correction": None if correction_record is None else {
                    "normalized_7d": correction_record["normalized_7d"].detach().cpu(),
                    "physical_7d": correction_record["physical_7d"].detach().cpu(),
                },
            })
            x_t = result["x_prev"]

            if self.last_guided_t is not None and timestep == self.last_guided_t:
                x_t_at_end = x_t.clone()

       
            result_raw = self.ddpm.step(x_t_raw, timestep, context, clip_denoised=self.clip_denoised, noise=noise)
            trace_raw.append({
                "t": timestep,
                "x_t": x_t_raw.detach().cpu(),
                "pred_xstart": result_raw["pred_xstart"].detach().cpu(),
                "x_prev": result_raw["x_prev"].detach().cpu(),
                "corrected": False,
                "correction": None,
            })
            x_t_raw = result_raw["x_prev"]

            result_physcene = self.ddpm.step(x_t_physcene, timestep, context, clip_denoised=self.clip_denoised, noise=noise)
            pred_x0_physcene = result_physcene["pred_xstart"]
            model_mean_physcene = result_physcene["model_mean"]

            physcene_applied = False
            if self.physcene and self.physcene.is_enabled() and self.should_apply_physcene(timestep):
             
                audit["physcene_steps"].append(timestep)
                variance = result_physcene["variance"]
                gradient = self.physcene.compute_gradient(model_mean_physcene, context, variance)

                if gradient is not None:
                  
                    model_mean_physcene = model_mean_physcene + gradient

                   
                   
                    log_variance = result_physcene["log_variance"]
                    if torch.count_nonzero(gradient):
                        audit["physcene_nonzero_steps"].append(timestep)
                    nonzero = 0.0 if timestep == 0 else 1.0
                    x_t_physcene = model_mean_physcene + nonzero * torch.exp(0.5 * log_variance) * noise
                    physcene_applied = True
                else:
                    x_t_physcene = result_physcene["x_prev"]
            else:
                x_t_physcene = result_physcene["x_prev"]

            trace_physcene.append({
                "t": timestep,
                "x_t": x_t_physcene.detach().cpu(),
                "pred_xstart": pred_x0_physcene.detach().cpu(),
                "x_prev": x_t_physcene.detach().cpu(),
                "physcene_applied": physcene_applied,
            })

           
            status_parts = []
            if correction_record is not None:
                status_parts.append("QueryA")
            if physcene_applied:
                status_parts.append("PhyScene")
            status = " + ".join(status_parts) if status_parts else "plain"
            pbar.set_postfix_str(f"t={timestep:3d} [{status}]")

        pbar.close()
        return {
            "audit": dict(audit, query_schedule=sorted(self.active_guidance_steps, reverse=True),
                          physcene_schedule=sorted(self.active_physcene_steps, reverse=True),
                          query_strength=self.guidance_strength),
            "layout": x_t,
            "layout_raw": x_t_raw,
            "layout_physcene": x_t_physcene if self.physcene is not None else None,  # 
            "layout_at_fork": x_t_at_fork,  
            "layout_at_end": x_t_at_end,  
            "last_guided_t": self.last_guided_t,
            "skipped_guidance_steps": list(self.skipped_guidance_steps),
            "trace": trace_corrected,
            "trace_raw": trace_raw,
            "trace_physcene": trace_physcene,  
            "context": context
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="DDPM + query-A hard combined inference")
    parser.add_argument("--combined-config", default=None, help="configuration for batch scene inference")
    parser.add_argument("--config", default=None, help="legacy: diffusion YAML")
    parser.add_argument("--weight-file", default=None, help="legacy: diffusion checkpoint")
    parser.add_argument("--scene-id", default=None, help="legacy single scene override")
    parser.add_argument("--batch-builder", default=None, help="module:function that builds a current-state GNN batch")
    parser.add_argument("--static-batch", default=None, help="torch file containing one GNN sample or collated batch")
    parser.add_argument("--query-config", default=None)
    parser.add_argument("--query-checkpoint", default=None)
    parser.add_argument("--query-thresholds", default=None)
    parser.add_argument("--bert-path", default=None, help="online edge-BERT model directory")
    parser.add_argument("--offline-edge-bert", action="store_true", help="use cached edge_bert (debug only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance-start-t", type=int, default=9)
    parser.add_argument("--guidance-end-t", type=int, default=0)
    parser.add_argument("--guidance-strength", type=float, default=1.0)
    parser.add_argument("--diffusion-scales", default=None, help="comma-separated 7-value x0 scales")
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help=(
            "omit the per-timestep sampling trace from the saved .pt files. "
            "The trace is only needed to debug guidance; it is ~8.5 MB per scene "
            "against ~3 KB for the final layout, and no Table 1 metric reads it."
        ),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = _read_yaml(args.combined_config) if args.combined_config else {}
    diffusion_cfg = cfg.get("diffusion", {})
    query_cfg = cfg.get("query", {})
    scene_cfg = cfg.get("scenes", {})
    batch_cfg = cfg.get("batch", {})
    guidance_cfg = cfg.get("guidance", {})
    runtime_cfg = cfg.get("runtime", {})

    diffusion_yaml = args.config or diffusion_cfg.get("config")
    weight_file = args.weight_file or diffusion_cfg.get("weight_file")
    if not diffusion_yaml or not weight_file:
        parser.error("provide --combined-config or both legacy --config and --weight-file")
    device = args.device if args.device != "cuda" else runtime_cfg.get("device", args.device)
    ddpm = DDPMInterface(
        diffusion_yaml,
        weight_file,
        device=device,
        floor_plan_textures=diffusion_cfg.get("floor_plan_textures"),
        scene_splits=diffusion_cfg.get("scene_splits"),
    )

   
    builder_spec = args.batch_builder or batch_cfg.get("builder")
    if builder_spec and "combined_dynamic_builder" in builder_spec:
        try:
            from combined_dynamic_builder import set_ddpm_interface
            set_ddpm_interface(ddpm)
            print("[run_combined]  builder ")
        except ImportError:
            pass 

    query = QueryAHardInterface(
        config_path=args.query_config or query_cfg.get("config", _COND_ROOT / "configs/conditional_reasoning.yaml"),
        checkpoint_path=args.query_checkpoint or query_cfg.get("checkpoint", _COND_ROOT / "checkpoints/best.pt"),
        thresholds_path=args.query_thresholds or query_cfg.get("thresholds", _COND_ROOT / "checkpoints/best_thresholds.json"),
        device=device,
        bert_path=args.bert_path or query_cfg.get("bert_path", _COND_ROOT.parents[3] / "hf_models" / "bert"),
        online_edge_bert=not (args.offline_edge_bert or query_cfg.get("offline_edge_bert", False)),
        node_cache_dir=query_cfg.get("node_cache_dir", _COND_ROOT.parents[3] / "GNN" / "adapters" / "cache"),
    )
    if query.room_type != ddpm.room_type:
        raise ValueError(f"Query-A room {query.room_type} does not match DiffuScene {ddpm.room_type}")

   
    physcene_cfg = cfg.get("physcene", {})
    physcene = None
    if physcene_cfg.get("enabled", False):
        print("\n[PhyScene] Initializing physics-based guidance...")
        if physcene_cfg.get("collision_type", "bbox_IOU") != "bbox_IOU":
            raise ValueError("Combined comparison currently requires bbox_IOU physics")
        physcene = PhySceneInterface(
            config=physcene_cfg, dataset=ddpm.raw_dataset, device=device,
            layout_spec=ddpm.layout_spec)
    if bool(args.batch_builder) == bool(args.static_batch) and not batch_cfg.get("builder") and not batch_cfg.get("static_batch_dir"):
        parser.error("provide batch.builder or batch.static_batch_dir in combined config")
    builder_spec = args.batch_builder or batch_cfg.get("builder")
    static_dir = args.static_batch or batch_cfg.get("static_batch_dir")
    scales_text = args.diffusion_scales or guidance_cfg.get("diffusion_scales")
    scales = None
    if isinstance(scales_text, str):
        scales = [float(v) for v in scales_text.split(",")]
    elif scales_text is not None:
        scales = [float(v) for v in scales_text]
    bounds_path = Path(ddpm.config["data"]["dataset_directory"]) / "dataset_stats.txt"
    dataset_bounds = DatasetBounds.from_stats(bounds_path)
    sampler_kwargs = dict(
        dataset_bounds=dataset_bounds,
        guidance_start_t=args.guidance_start_t if args.guidance_start_t != 9 else guidance_cfg.get("start_t", 9),
        guidance_end_t=args.guidance_end_t if args.guidance_end_t != 0 else guidance_cfg.get("end_t", 0),
        guidance_strength=args.guidance_strength if args.guidance_strength != 1.0 else guidance_cfg.get("strength", 1.0),
        guidance_interval=guidance_cfg.get("interval"),  
        guidance_steps=guidance_cfg.get("steps"),  
        diffusion_scales=scales,
    )

   
    physcene_window_cfg = physcene_cfg.get("guidance_window") or {}
    sampler_kwargs.update(
        physcene_start_t=int(physcene_window_cfg.get("start_t", 9)),
        physcene_end_t=int(physcene_window_cfg.get("end_t", 0)),
        physcene_interval=physcene_window_cfg.get("interval"),
        physcene_steps=physcene_window_cfg.get("steps"),
    )

    if args.scene_id:
       
        short_id = _normalize_scene_id(args.scene_id)
        full_id = args.scene_id
        scene_ids = [(short_id, full_id)]
    elif args.combined_config:
        scene_ids = select_scene_ids(
            scene_cfg.get("sample_root") or scene_cfg.get("data_dir"),
            scene_cfg.get("scene_ids_file", "scene_ids.txt"),
            int(scene_cfg.get("num_samples", 500)),
            int(scene_cfg.get("random_seed", 42)),
            bool(scene_cfg.get("refresh", False)),
            use_json=bool(scene_cfg.get("use_json", False)),
            scene_ids_json=scene_cfg.get("scene_ids_json"),
        )
    else:
        parser.error("legacy mode requires --scene-id")

    output_dir = Path(args.output or runtime_cfg.get("output_dir", "./combined_outputs")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

   
    output_raw_dir = output_dir.parent / (output_dir.name + "_raw")
    output_fork_dir = output_dir.parent / (output_dir.name + "_fork")
    output_end_dir = output_dir.parent / (output_dir.name + "_end")
    output_physcene_dir = output_dir.parent / (output_dir.name + "_physcene")

   
    failures = []
    total_scenes = len(scene_ids)

    for scene_index, (short_id, full_id) in enumerate(scene_ids, start=1):
        try:
            base_seed = args.seed if args.seed is not None else int(scene_cfg.get("random_seed", 42))
            scene_seed = int(hashlib.sha256(f"{base_seed}:{full_id}".encode()).hexdigest()[:8], 16)
            random.seed(scene_seed)
            np.random.seed(scene_seed)
            torch.manual_seed(scene_seed)
         
            context = ddpm.scene_context(short_id)
            
            context["full_scene_id"] = full_id
            context["static_dataset_directory"] = batch_cfg.get("static_dataset_directory")

            print(f"\n{'='*60}")
            print(f"Scene [{scene_index}/{total_scenes}]: {short_id}")
            print(f"{'='*60}")
            text_info = context.get("text_embed_input", context.get("description", context.get("text", "")))
            if text_info and isinstance(text_info, str):
                print(f"Text condition: {text_info}")
            else:
                print(f"Text condition: (embedded tensor or not available)")
            print(f"Context keys: {list(context.keys())}")
            print(f"{'='*60}\n")

            if builder_spec:
                builder = _load_callable(builder_spec)
            else:
                static_path = Path(static_dir).expanduser() / f"{short_id}.pt"
                builder = static_batch_builder(static_path)
            sampler = CombinedSampler(ddpm, query, builder, physcene=physcene, **sampler_kwargs)
            result = sampler.sample(context, seed=scene_seed)

            keep_trace = not args.no_trace
            output = output_dir / f"{short_id}.pt" if output_dir.suffix != ".pt" else output_dir
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "layout": result["layout"].detach().cpu(),
                "audit": result["audit"],
                "scene_id": short_id,
                "text": context.get("text", ""),  
                "room_mask": context.get("room_mask", None).detach().cpu(),  
            }
            if keep_trace:
                payload["trace"] = result["trace"]
            torch.save(payload, output)

      
            output_raw_dir.mkdir(parents=True, exist_ok=True)
            output_raw = output_raw_dir / f"{short_id}.pt"
            payload_raw = {
                "layout": result["layout_raw"].detach().cpu(),
                "audit": result["audit"],
                "scene_id": short_id,
                "text": context.get("text", ""),
                "room_mask": context.get("room_mask", None).detach().cpu(),
            }
            if keep_trace:
                payload_raw["trace"] = result["trace_raw"]
            torch.save(payload_raw, output_raw)

    
            output_fork = None
            if result.get("layout_at_fork") is not None:
                output_fork_dir.mkdir(parents=True, exist_ok=True)
                output_fork = output_fork_dir / f"{short_id}.pt"
                torch.save({
                    "layout": result["layout_at_fork"].detach().cpu(),
                    "scene_id": short_id,
                    "text": context.get("text", ""),
                    "room_mask": context.get("room_mask", None).detach().cpu(),
                    "fork_timestep": sampler.guidance_start_t + 1
                }, output_fork)

           
            output_end = None
            if result.get("layout_at_end") is not None and sampler.last_guided_t not in (None, 0):
                output_end_dir.mkdir(parents=True, exist_ok=True)
                output_end = output_end_dir / f"{short_id}.pt"
                torch.save({
                    "layout": result["layout_at_end"].detach().cpu(),
                    "scene_id": short_id,
                    "text": context.get("text", ""),
                    "room_mask": context.get("room_mask", None).detach().cpu(),
                    "end_timestep": sampler.last_guided_t,
                }, output_end)

        
            if result.get("layout_physcene") is not None:
                output_physcene_dir.mkdir(parents=True, exist_ok=True)
                output_physcene = output_physcene_dir / f"{short_id}.pt"
                payload_physcene = {
                    "layout": result["layout_physcene"].detach().cpu(),
                    "audit": result["audit"],
                    "scene_id": short_id,
                    "text": context.get("text", ""),
                    "room_mask": context.get("room_mask", None).detach().cpu(),
                }
                if keep_trace:
                    payload_physcene["trace"] = result.get("trace_physcene", [])
                torch.save(payload_physcene, output_physcene)
            else:
                output_physcene = None

            skipped = result.get("skipped_guidance_steps") or []
            print(json.dumps({
                "scene_id": short_id,
                "output_corrected": str(output),
                "output_raw": str(output_raw),
                "output_fork": str(output_fork) if output_fork is not None else None,
                "output_end": str(output_end) if output_end is not None else None,
                "output_physcene": str(output_physcene) if output_physcene is not None else None,
                "fork_timestep": sampler.guidance_start_t + 1,
                "end_timestep": sampler.last_guided_t,
                "guidance_timesteps": sorted(list(sampler.active_guidance_steps), reverse=True),
                "guidance_mode": "steps" if sampler.guidance_steps is not None else ("interval" if sampler.guidance_interval is not None else "continuous"),
                "physcene_timesteps": sorted(list(sampler.active_physcene_steps), reverse=True),
                "physcene_mode": "steps" if sampler.physcene_steps is not None else ("interval" if sampler.physcene_interval is not None else "continuous"),
                "skipped_guidance_steps": sorted(skipped, reverse=True),
                "trace_saved": keep_trace,
            }))

        except KeyboardInterrupt:
           
            raise
        except Exception as exc:
            failures.append({
                "scene_id": short_id,
                "scene_index": scene_index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(
                f"\n!!  {short_id} [{scene_index}/{total_scenes}] ，"
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            continue

    succeeded = total_scenes - len(failures)
    summary = {
        "total_scenes": total_scenes,
        "succeeded": succeeded,
        "failed": len(failures),
        "failures": failures,
        "output_dir": str(output_dir),
    }
    failure_path = output_dir.parent / f"{output_dir.name}_failures.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"{succeeded}/{total_scenes} 成功，{len(failures)} 失败")
    if failures:
        print(f"（ {failure_path}）：")
        for item in failures:
            print(f"  - {item['scene_id']}: {item['error_type']}: {item['error']}")
    print(f"{'='*60}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
