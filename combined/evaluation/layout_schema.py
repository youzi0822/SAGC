
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class LayoutSpec:


    schema_version: int = 1
    num_slots: int = 12
    point_dim: int = 62
    translation_dim: int = 3
    size_dim: int = 3
    angle_dim: int = 2
    furniture_class_dim: int = 21
    class_dim: int = 22
    objfeat_dim: int = 32

    @property
    def translation_slice(self) -> slice:
        return slice(0, 3)

    @property
    def size_slice(self) -> slice:
        return slice(3, 6)

    @property
    def angle_slice(self) -> slice:
        return slice(6, 8)

    @property
    def furniture_class_slice(self) -> slice:
        return slice(8, 8 + self.furniture_class_dim)

    @property
    def class_slice(self) -> slice:
        return slice(8, 8 + self.class_dim)

    @property
    def empty_index(self) -> int:
        return 8 + self.class_dim - 1

    @property
    def objfeat_slice(self) -> slice:
        return slice(8 + self.class_dim, self.point_dim)

    @classmethod
    def from_network(cls, network: Mapping[str, Any]) -> "LayoutSpec":
        if int(network.get("objectness_dim", 0)) != 0 or int(network.get("angle_dim", 2)) != 2:
            raise ValueError("Combined guidance requires class-block empty flag and cos/sin angles")
        spec = cls(num_slots=int(network['sample_num_points']), point_dim=int(network['point_dim']),
                   class_dim=int(network['class_dim']), furniture_class_dim=int(network['class_dim']) - 1,
                   objfeat_dim=int(network['objfeat_dim']))
        spec.validate()
        return spec

    def validate(self) -> None:
        total = (
            self.translation_dim
            + self.size_dim
            + self.angle_dim
            + self.class_dim
            + self.objfeat_dim
        )
        if total != self.point_dim:
            raise ValueError(f"layout dimensions sum to {total}, expected {self.point_dim}")
        if self.class_dim != self.furniture_class_dim + 1:
            raise ValueError("class_dim must contain furniture classes plus one empty class")
        if self.objfeat_slice.stop != self.point_dim:
            raise ValueError("feature slices do not cover point_dim exactly")


DEFAULT_LAYOUT_SPEC = LayoutSpec()
DEFAULT_LAYOUT_SPEC.validate()


def _finite_vector(value: Any, *, name: str, length: int = 3) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}], got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return tensor


@dataclass(frozen=True)
class DatasetBounds:
    """Physical training bounds for normalized translation and half-size XYZ."""

    translation_min: torch.Tensor
    translation_max: torch.Tensor
    size_min: torch.Tensor
    size_max: torch.Tensor

    def __post_init__(self) -> None:
        fields = {
            "translation_min": self.translation_min,
            "translation_max": self.translation_max,
            "size_min": self.size_min,
            "size_max": self.size_max,
        }
        converted = {
            name: _finite_vector(value, name=name)
            for name, value in fields.items()
        }
        for name, value in converted.items():
            object.__setattr__(self, name, value)
        if not bool((self.translation_max > self.translation_min).all()):
            raise ValueError("translation_max must be greater than translation_min on every axis")
        if not bool((self.size_max > self.size_min).all()):
            raise ValueError("size_max must be greater than size_min on every axis")

    @classmethod
    def from_stats(cls, path: str | Path) -> "DatasetBounds":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    @classmethod
    def from_mapping(cls, stats: Mapping[str, Any]) -> "DatasetBounds":
        translations = _finite_vector(
            stats.get("bounds_translations"),
            name="bounds_translations",
            length=6,
        )
        sizes = _finite_vector(
            stats.get("bounds_sizes"),
            name="bounds_sizes",
            length=6,
        )
        return cls(
            translation_min=translations[:3],
            translation_max=translations[3:],
            size_min=sizes[:3],
            size_max=sizes[3:],
        )

    def like(self, reference: torch.Tensor) -> tuple[torch.Tensor, ...]:
        kwargs = {"device": reference.device, "dtype": reference.dtype}
        return (
            self.translation_min.to(**kwargs),
            self.translation_max.to(**kwargs),
            self.size_min.to(**kwargs),
            self.size_max.to(**kwargs),
        )


@dataclass(frozen=True)
class CanonicalLayout:
    """One scene in room-centered physical coordinates."""

    slot_indices: torch.Tensor
    translations_xyz: torch.Tensor
    sizes_half_xyz: torch.Tensor
    yaw: torch.Tensor
    class_ids: torch.Tensor

    @property
    def num_objects(self) -> int:
        return int(self.slot_indices.numel())


@dataclass(frozen=True)
class CorrectionApplication:
    """Corrected state plus enough information to audit clipping and routing."""

    corrected_x0: torch.Tensor
    slot_indices: torch.Tensor
    requested_physical_7d: torch.Tensor
    applied_physical_7d: torch.Tensor
    translation_clipped: torch.Tensor
    size_clipped: torch.Tensor


def _scene_view(x0: torch.Tensor, spec: LayoutSpec) -> tuple[torch.Tensor, bool]:
    if not isinstance(x0, torch.Tensor):
        raise TypeError(f"x0 must be a torch.Tensor, got {type(x0)!r}")
    if not x0.is_floating_point():
        raise TypeError(f"x0 must be floating point, got {x0.dtype}")
    if x0.ndim == 2:
        scene = x0
        had_batch = False
    elif x0.ndim == 3 and x0.shape[0] == 1:
        scene = x0[0]
        had_batch = True
    else:
        raise ValueError(f"x0 must have shape [slots,features] or [1,slots,features], got {tuple(x0.shape)}")
    if scene.shape != (spec.num_slots, spec.point_dim):
        raise ValueError(
            f"x0 scene must have shape [{spec.num_slots},{spec.point_dim}], "
            f"got {tuple(scene.shape)}"
        )
    if not bool(torch.isfinite(scene).all()):
        raise ValueError("x0 contains NaN or infinity")
    return scene, had_batch


def validate_ddpm_state(
    x0: torch.Tensor,
    spec: LayoutSpec = DEFAULT_LAYOUT_SPEC,
) -> None:
    spec.validate()
    _scene_view(x0, spec)


def valid_slot_indices(
    x0: torch.Tensor,
    spec: LayoutSpec = DEFAULT_LAYOUT_SPEC,
) -> torch.Tensor:
    """Return slots whose DiffuScene empty-class channel is negative."""

    scene, _ = _scene_view(x0, spec)
    return torch.nonzero(scene[:, spec.empty_index] < 0, as_tuple=False).flatten()


def _descale(normalized: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
    return (normalized + 1.0) * 0.5 * (maximum - minimum) + minimum


def decode_ddpm_layout(
    x0: torch.Tensor,
    bounds: DatasetBounds,
    spec: LayoutSpec = DEFAULT_LAYOUT_SPEC,
) -> CanonicalLayout:
    """Decode one normalized 62D state into physical half-boxes in XYZ order."""

    scene, _ = _scene_view(x0, spec)
    slots = valid_slot_indices(scene, spec)
    trans_min, trans_max, size_min, size_max = bounds.like(scene)

    selected = scene.index_select(0, slots)
    translations = _descale(selected[:, spec.translation_slice], trans_min, trans_max)
    sizes = _descale(selected[:, spec.size_slice], size_min, size_max)
    angle = selected[:, spec.angle_slice]
    angle_norm = torch.linalg.vector_norm(angle, dim=-1)
    if angle.numel() and bool((angle_norm <= 1e-12).any()):
        bad_slots = slots[angle_norm <= 1e-12].detach().cpu().tolist()
        raise ValueError(f"undefined yaw: zero cos/sin vector at slots {bad_slots}")
    yaw = torch.atan2(angle[:, 1], angle[:, 0])
    class_ids = torch.argmax(selected[:, spec.furniture_class_slice], dim=-1)

    return CanonicalLayout(
        slot_indices=slots,
        translations_xyz=translations,
        sizes_half_xyz=sizes,
        yaw=yaw,
        class_ids=class_ids,
    )


def query_full_wdh_to_ddpm_half_xyz(full_wdh: torch.Tensor) -> torch.Tensor:
    """Convert Query-A full [W=X,D=Z,H=Y] to DDPM half [X,Y,Z]."""

    if full_wdh.ndim != 2 or full_wdh.shape[-1] != 3:
        raise ValueError(f"full_wdh must have shape [N,3], got {tuple(full_wdh.shape)}")
    if not bool(torch.isfinite(full_wdh).all()):
        raise ValueError("full_wdh contains NaN or infinity")
    width, depth, height = full_wdh.unbind(dim=-1)
    return 0.5 * torch.stack((width, height, depth), dim=-1)


def ddpm_half_xyz_to_query_full_wdh(half_xyz: torch.Tensor) -> torch.Tensor:
    """Convert DDPM half [X,Y,Z] to Query-A full [W=X,D=Z,H=Y]."""

    if half_xyz.ndim != 2 or half_xyz.shape[-1] != 3:
        raise ValueError(f"half_xyz must have shape [N,3], got {tuple(half_xyz.shape)}")
    if not bool(torch.isfinite(half_xyz).all()):
        raise ValueError("half_xyz contains NaN or infinity")
    half_x, half_y, half_z = half_xyz.unbind(dim=-1)
    return 2.0 * torch.stack((half_x, half_z, half_y), dim=-1)


def physical_translation_to_normalized(
    delta_xyz: torch.Tensor,
    bounds: DatasetBounds,
) -> torch.Tensor:
    if delta_xyz.ndim != 2 or delta_xyz.shape[-1] != 3:
        raise ValueError(f"delta_xyz must have shape [N,3], got {tuple(delta_xyz.shape)}")
    trans_min, trans_max, _, _ = bounds.like(delta_xyz)
    return 2.0 * delta_xyz / (trans_max - trans_min)


def physical_full_wdh_to_normalized_half_xyz(
    delta_full_wdh: torch.Tensor,
    bounds: DatasetBounds,
) -> torch.Tensor:
    delta_half_xyz = query_full_wdh_to_ddpm_half_xyz(delta_full_wdh)
    _, _, size_min, size_max = bounds.like(delta_full_wdh)
    return 2.0 * delta_half_xyz / (size_max - size_min)


def _validate_slots(slot_indices: torch.Tensor, count: int, spec: LayoutSpec, device: torch.device) -> torch.Tensor:
    slots = torch.as_tensor(slot_indices, device=device)
    if slots.ndim != 1 or slots.numel() != count:
        raise ValueError(f"slot_indices must have shape [{count}], got {tuple(slots.shape)}")
    if slots.is_floating_point() or slots.dtype == torch.bool:
        raise TypeError("slot_indices must use an integer dtype")
    slots = slots.to(dtype=torch.long)
    if slots.numel() and bool(((slots < 0) | (slots >= spec.num_slots)).any()):
        raise ValueError(f"slot_indices must be in [0,{spec.num_slots - 1}]")
    if torch.unique(slots).numel() != slots.numel():
        raise ValueError("slot_indices contains duplicates")
    return slots


def _normalized_translation_to_physical(
    delta_normalized: torch.Tensor,
    bounds: DatasetBounds,
) -> torch.Tensor:
    trans_min, trans_max, _, _ = bounds.like(delta_normalized)
    return 0.5 * delta_normalized * (trans_max - trans_min)


def _normalized_half_xyz_to_physical_full_wdh(
    delta_normalized: torch.Tensor,
    bounds: DatasetBounds,
) -> torch.Tensor:
    _, _, size_min, size_max = bounds.like(delta_normalized)
    delta_half_xyz = 0.5 * delta_normalized * (size_max - size_min)
    return ddpm_half_xyz_to_query_full_wdh(delta_half_xyz)


def apply_query_correction(
    x0: torch.Tensor,
    slot_indices: torch.Tensor,
    physical_7d: torch.Tensor,
    bounds: DatasetBounds,
    *,
    strength: float = 1.0,
    spec: LayoutSpec = DEFAULT_LAYOUT_SPEC,
    clip_geometry: bool = True,
) -> CorrectionApplication:
    """Apply physical Query-A corrections without touching class or objfeat.

    ``physical_7d`` uses [dx,dy,dz,dw,dd,dh,dtheta], where size is full WDH.
    The returned ``requested_physical_7d`` already includes ``strength``.
    """

    scene, had_batch = _scene_view(x0, spec)
    if physical_7d.ndim != 2 or physical_7d.shape[-1] != 7:
        raise ValueError(f"physical_7d must have shape [N,7], got {tuple(physical_7d.shape)}")
    if not physical_7d.is_floating_point():
        raise TypeError("physical_7d must be floating point")
    if not bool(torch.isfinite(physical_7d).all()):
        raise ValueError("physical_7d contains NaN or infinity")
    if not math.isfinite(float(strength)) or float(strength) < 0.0:
        raise ValueError("strength must be finite and non-negative")

    correction = physical_7d.to(device=scene.device, dtype=scene.dtype) * float(strength)
    slots = _validate_slots(slot_indices, correction.shape[0], spec, scene.device)
    current_valid = valid_slot_indices(scene, spec)
    if slots.numel() and not bool(torch.isin(slots, current_valid).all()):
        invalid = slots[~torch.isin(slots, current_valid)].detach().cpu().tolist()
        raise ValueError(f"correction targets empty slots: {invalid}")

    corrected_scene = scene.clone()
    before = scene.index_select(0, slots)

    translation_delta = physical_translation_to_normalized(correction[:, 0:3], bounds)
    size_delta = physical_full_wdh_to_normalized_half_xyz(correction[:, 3:6], bounds)
    translation_candidate = before[:, spec.translation_slice] + translation_delta
    size_candidate = before[:, spec.size_slice] + size_delta
    translation_clipped = (translation_candidate < -1.0) | (translation_candidate > 1.0)
    size_clipped = (size_candidate < -1.0) | (size_candidate > 1.0)
    if clip_geometry:
        translation_after = translation_candidate.clamp(-1.0, 1.0)
        size_after = size_candidate.clamp(-1.0, 1.0)
    else:
        translation_after = translation_candidate
        size_after = size_candidate
        translation_clipped = torch.zeros_like(translation_clipped)
        size_clipped = torch.zeros_like(size_clipped)

    old_angle = before[:, spec.angle_slice].clone()
    if old_angle.numel() and bool((torch.linalg.vector_norm(old_angle, dim=-1) <= 1e-12).any()):
        raise ValueError("cannot rotate a zero cos/sin vector")
    theta = correction[:, 6]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    angle_after = torch.stack(
        (
            old_angle[:, 0] * cos_theta - old_angle[:, 1] * sin_theta,
            old_angle[:, 0] * sin_theta + old_angle[:, 1] * cos_theta,
        ),
        dim=-1,
    )

    corrected_scene[slots, spec.translation_slice] = translation_after
    corrected_scene[slots, spec.size_slice] = size_after
    corrected_scene[slots, spec.angle_slice] = angle_after

    normalized_translation_applied = translation_after - before[:, spec.translation_slice]
    normalized_size_applied = size_after - before[:, spec.size_slice]
    translation_applied = _normalized_translation_to_physical(normalized_translation_applied, bounds)
    size_applied = _normalized_half_xyz_to_physical_full_wdh(normalized_size_applied, bounds)
    dot = (old_angle * angle_after).sum(dim=-1)
    cross = old_angle[:, 0] * angle_after[:, 1] - old_angle[:, 1] * angle_after[:, 0]
    angle_applied = torch.atan2(cross, dot).unsqueeze(-1)
    applied = torch.cat((translation_applied, size_applied, angle_applied), dim=-1)

    corrected_x0 = corrected_scene.unsqueeze(0) if had_batch else corrected_scene
    return CorrectionApplication(
        corrected_x0=corrected_x0,
        slot_indices=slots,
        requested_physical_7d=correction,
        applied_physical_7d=applied,
        translation_clipped=translation_clipped,
        size_clipped=size_clipped,
    )
