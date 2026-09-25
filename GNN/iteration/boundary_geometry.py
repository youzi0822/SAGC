
from typing import Dict, List, Optional, Tuple

import numpy as np

EPS = 1e-8

VERTEX_DEDUP_DECIMALS = 6

COLLINEAR_SIN_TOL = 1e-6

RECT_FALLBACK_AREA_RATIO_MIN = 0.99




def _dedup_vertices_xz(
    vertices_xz: np.ndarray, decimals: int = VERTEX_DEDUP_DECIMALS
) -> Tuple[np.ndarray, np.ndarray]:
  
    quantized = np.round(vertices_xz.astype(np.float64), decimals)
    unique_q, old_to_new = np.unique(quantized, axis=0, return_inverse=True)
    old_to_new = old_to_new.reshape(-1)

   
    unique_xz = np.zeros_like(unique_q, dtype=np.float64)
    seen = np.zeros(unique_q.shape[0], dtype=bool)
    for old_idx, new_idx in enumerate(old_to_new):
        if not seen[new_idx]:
            unique_xz[new_idx] = vertices_xz[old_idx].astype(np.float64)
            seen[new_idx] = True
    return unique_xz, old_to_new


def _boundary_edges_from_faces(
    faces: np.ndarray, old_to_new: np.ndarray
) -> List[Tuple[int, int]]:
    
    counts: Dict[Tuple[int, int], int] = {}
    for face in faces:
        if len(face) < 3:
            continue
        idx = [int(old_to_new[int(v)]) for v in face]
        n = len(idx)
        for k in range(n):
            a, b = idx[k], idx[(k + 1) % n]
            if a == b:
                continue  # 退化边（重复顶点造成）
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return [e for e, c in counts.items() if c == 1]


def _chain_rings(boundary_edges: List[Tuple[int, int]]) -> List[List[int]]:
  
    adj: Dict[int, List[int]] = {}
    for a, b in boundary_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    unused = set()
    for a, b in boundary_edges:
        unused.add((a, b) if a < b else (b, a))

    rings: List[List[int]] = []
    while unused:
        a0, b0 = next(iter(unused))
        unused.discard((a0, b0))
        ring = [a0, b0]
        while True:
            cur = ring[-1]
            nxt = None
            for cand in adj.get(cur, []):
                key = (cur, cand) if cur < cand else (cand, cur)
                if key in unused:
                    nxt = cand
                    unused.discard(key)
                    break
            if nxt is None:
                break
            if nxt == ring[0]:
                break  # 闭合
            ring.append(nxt)
        rings.append(ring)
    return rings


def _signed_area_xz(ring_xz: np.ndarray) -> float:
   
    x = ring_xz[:, 0]
    z = ring_xz[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def _point_in_polygon_xz(point: np.ndarray, rings_xz: List[np.ndarray]) -> bool:
   
    px, pz = float(point[0]), float(point[1])
    inside = False
    for ring in rings_xz:
        n = ring.shape[0]
        for k in range(n):
            x0, z0 = float(ring[k, 0]), float(ring[k, 1])
            x1, z1 = float(ring[(k + 1) % n, 0]), float(ring[(k + 1) % n, 1])
            if (z0 > pz) != (z1 > pz):
                t = (pz - z0) / (z1 - z0) if abs(z1 - z0) > EPS else 0.0
                x_cross = x0 + t * (x1 - x0)
                if px < x_cross:
                    inside = not inside
    return inside


def _merge_collinear(ring_xz: np.ndarray) -> np.ndarray:
    
    K = ring_xz.shape[0]
    if K <= 3:
        return ring_xz
    keep: List[int] = []
    for k in range(K):
        prev_v = ring_xz[(k - 1) % K]
        cur_v = ring_xz[k]
        next_v = ring_xz[(k + 1) % K]
        d0 = cur_v - prev_v
        d1 = next_v - cur_v
        n0 = float(np.linalg.norm(d0))
        n1 = float(np.linalg.norm(d1))
        if n0 < EPS or n1 < EPS:
            continue  # 重复顶点，丢弃
        cross = float(d0[0] * d1[1] - d0[1] * d1[0]) / (n0 * n1)
        if abs(cross) > COLLINEAR_SIN_TOL:
            keep.append(k)  # 拐点
    if len(keep) < 3:
        return ring_xz
    return ring_xz[keep]


def extract_boundary_segments(
    floor_plan_vertices: np.ndarray,
    floor_plan_faces: np.ndarray,
    floor_plan_centroid: np.ndarray,
) -> Dict:
   
    warnings: List[str] = []
    vertices = np.asarray(floor_plan_vertices, dtype=np.float64)
    faces = np.asarray(floor_plan_faces)
    centroid = np.asarray(floor_plan_centroid, dtype=np.float64).reshape(-1)

    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError(f"floor_plan_vertices 形状非法: {vertices.shape}")


    centered_xz = vertices[:, [0, 2]] - centroid[[0, 2]]

    unique_xz, old_to_new = _dedup_vertices_xz(centered_xz)
    boundary_edges = _boundary_edges_from_faces(faces, old_to_new)

    if len(boundary_edges) == 0:
        warnings.append("no_boundary_edges")
        return {
            "vertices_centered_xz": unique_xz.astype(np.float32),
            "segments_xz": np.zeros((0, 2, 2), dtype=np.float32),
            "tangent_xz": np.zeros((0, 2), dtype=np.float32),
            "inward_normal_xz": np.zeros((0, 2), dtype=np.float32),
            "segment_length": np.zeros((0,), dtype=np.float32),
            "n_rings": 0,
            "warnings": warnings,
        }

    rings_idx = _chain_rings(boundary_edges)
    rings_xz_raw = [unique_xz[np.asarray(r, dtype=np.int64)] for r in rings_idx if len(r) >= 3]
    if len(rings_xz_raw) == 0:
        warnings.append("no_valid_ring")
        return {
            "vertices_centered_xz": unique_xz.astype(np.float32),
            "segments_xz": np.zeros((0, 2, 2), dtype=np.float32),
            "tangent_xz": np.zeros((0, 2), dtype=np.float32),
            "inward_normal_xz": np.zeros((0, 2), dtype=np.float32),
            "segment_length": np.zeros((0,), dtype=np.float32),
            "n_rings": 0,
            "warnings": warnings,
        }

    
    areas_raw = [_signed_area_xz(r) for r in rings_xz_raw]
    outer_i = int(np.argmax([abs(a) for a in areas_raw]))
    rings_xz: List[np.ndarray] = []
    for i, ring in enumerate(rings_xz_raw):
        merged = _merge_collinear(ring)
        area = _signed_area_xz(merged)
        want_ccw = i == outer_i
        if (area > 0.0) != want_ccw:
            merged = merged[::-1]
        rings_xz.append(merged)

    if len(rings_xz) > 1:
        warnings.append(f"multi_ring_{len(rings_xz)}")

   
    seg_list: List[np.ndarray] = []
    tan_list: List[np.ndarray] = []
    nrm_list: List[np.ndarray] = []
    len_list: List[float] = []
    n_flipped = 0

    for ring in rings_xz:
        K = ring.shape[0]
        for k in range(K):
            p0 = ring[k]
            p1 = ring[(k + 1) % K]
            d = p1 - p0
            length = float(np.linalg.norm(d))
            if length < EPS:
                continue
            tangent = d / length
            
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

          
            probe_eps = min(0.01, 0.25 * length)
            if not _point_in_polygon_xz(0.5 * (p0 + p1) + probe_eps * normal, rings_xz):
                if _point_in_polygon_xz(0.5 * (p0 + p1) - probe_eps * normal, rings_xz):
                    normal = -normal
                    n_flipped += 1

            seg_list.append(np.stack([p0, p1], axis=0))
            tan_list.append(tangent)
            nrm_list.append(normal)
            len_list.append(length)

    if n_flipped > 0:
        warnings.append(f"inward_normal_flipped_{n_flipped}")

    return {
        "vertices_centered_xz": unique_xz.astype(np.float32),
        "segments_xz": np.asarray(seg_list, dtype=np.float32).reshape(-1, 2, 2),
        "tangent_xz": np.asarray(tan_list, dtype=np.float32).reshape(-1, 2),
        "inward_normal_xz": np.asarray(nrm_list, dtype=np.float32).reshape(-1, 2),
        "segment_length": np.asarray(len_list, dtype=np.float32).reshape(-1),
        "n_rings": len(rings_xz),
        "warnings": warnings,
    }


def floor_area_from_faces(
    floor_plan_vertices: np.ndarray, floor_plan_faces: np.ndarray
) -> float:
    
    vertices = np.asarray(floor_plan_vertices, dtype=np.float64)
    faces = np.asarray(floor_plan_faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.size == 0:
        return 0.0
    v_xz = vertices[:, [0, 2]]
    a = v_xz[faces[:, 0]]
    b = v_xz[faces[:, 1]]
    c = v_xz[faces[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    return 0.5 * float(np.abs(cross).sum())


def rectangle_boundary(width_x: float, length_z: float) -> Dict:
  
    W = float(width_x)
    L = float(length_z)
    if W <= 0.0 or L <= 0.0:
        return {
            "vertices_centered_xz": np.zeros((0, 2), dtype=np.float32),
            "segments_xz": np.zeros((0, 2, 2), dtype=np.float32),
            "tangent_xz": np.zeros((0, 2), dtype=np.float32),
            "inward_normal_xz": np.zeros((0, 2), dtype=np.float32),
            "segment_length": np.zeros((0,), dtype=np.float32),
            "n_rings": 0,
            "warnings": ["rect_fallback_nonpositive_dims"],
        }

    hw, hl = 0.5 * W, 0.5 * L
    ring = np.array(
        [[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]], dtype=np.float64
    )
    if _signed_area_xz(ring) < 0.0:
        ring = ring[::-1]

    seg_list, tan_list, nrm_list, len_list = [], [], [], []
    for k in range(ring.shape[0]):
        p0 = ring[k]
        p1 = ring[(k + 1) % ring.shape[0]]
        d = p1 - p0
        length = float(np.linalg.norm(d))
        if length < EPS:
            continue
        tangent = d / length
        seg_list.append(np.stack([p0, p1], axis=0))
        tan_list.append(tangent)
        nrm_list.append(np.array([-tangent[1], tangent[0]], dtype=np.float64))
        len_list.append(length)

    return {
        "vertices_centered_xz": ring.astype(np.float32),
        "segments_xz": np.asarray(seg_list, dtype=np.float32).reshape(-1, 2, 2),
        "tangent_xz": np.asarray(tan_list, dtype=np.float32).reshape(-1, 2),
        "inward_normal_xz": np.asarray(nrm_list, dtype=np.float32).reshape(-1, 2),
        "segment_length": np.asarray(len_list, dtype=np.float32).reshape(-1),
        "n_rings": 1,
        "warnings": ["rect_fallback"],
    }



def local_axes_from_yaw(yaw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
   
    yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
    c = np.cos(yaw)
    s = np.sin(yaw)
    width_axis = np.stack([c, s], axis=1)
    depth_axis = np.stack([-s, c], axis=1)
    return width_axis, depth_axis


def closest_point_on_segment(
    centers_xz: np.ndarray, seg_start_xz: np.ndarray, seg_end_xz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
   
    d = seg_end_xz - seg_start_xz
    denom = np.sum(d * d, axis=1)
    w = centers_xz - seg_start_xz
    t = np.where(denom > EPS, np.sum(w * d, axis=1) / np.maximum(denom, EPS), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = seg_start_xz + t[:, None] * d
    distance = np.linalg.norm(centers_xz - closest, axis=1)
    return closest, distance



GEO_EDGE_SCALAR_DIM = 15
LOG_RATIO_EPS = 1e-3


def build_edge_features(
    edge_index: np.ndarray,
    positions: np.ndarray,
    yaw: np.ndarray,
    full_wdh: np.ndarray,
    room_diag_D: float,
    room_height_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
   
    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(2, -1)
    E = edge_index.shape[1]
    positions = np.asarray(positions, dtype=np.float64)
    full_wdh = np.asarray(full_wdh, dtype=np.float64)

    D = max(float(room_diag_D), EPS)
    H = max(float(room_height_y), EPS)

    if E == 0:
        return (
            np.zeros((0, GEO_EDGE_SCALAR_DIM), dtype=np.float32),
            np.zeros((0, 2, 3), dtype=np.float32),
        )

    src = edge_index[0]
    dst = edge_index[1]

    width_axis, depth_axis = local_axes_from_yaw(yaw)  # [V, 2] each
    u_i = width_axis[dst]
    v_i = depth_axis[dst]

    r = positions[src] - positions[dst]              # [E, 3]
    r_xz = r[:, [0, 2]]
    dy = r[:, 1]

    local_w = np.sum(r_xz * u_i, axis=1)
    local_d = np.sum(r_xz * v_i, axis=1)
    dist_xz = np.linalg.norm(r_xz, axis=1)

    tgt = full_wdh[dst]
    src_wdh = full_wdh[src]

    yaw_arr = np.asarray(yaw, dtype=np.float64).reshape(-1)
    d_yaw = yaw_arr[src] - yaw_arr[dst]

    edge_scalar = np.stack(
        [
            local_w / D,
            local_d / D,
            dy / H,
            dist_xz / D,
            tgt[:, 0] / D,
            tgt[:, 1] / D,
            tgt[:, 2] / H,
            src_wdh[:, 0] / D,
            src_wdh[:, 1] / D,
            src_wdh[:, 2] / H,
            np.log((src_wdh[:, 0] + LOG_RATIO_EPS) / (tgt[:, 0] + LOG_RATIO_EPS)),
            np.log((src_wdh[:, 1] + LOG_RATIO_EPS) / (tgt[:, 1] + LOG_RATIO_EPS)),
            np.log((src_wdh[:, 2] + LOG_RATIO_EPS) / (tgt[:, 2] + LOG_RATIO_EPS)),
            np.sin(d_yaw),
            np.cos(d_yaw),
        ],
        axis=1,
    )

    # r_hat 与 J(r_hat)；零距离置零，保留距离/dy 标量
    safe = dist_xz > 1e-6
    r_hat_xz = np.zeros_like(r_xz)
    r_hat_xz[safe] = r_xz[safe] / dist_xz[safe, None]
    j_hat_xz = np.stack([-r_hat_xz[:, 1], r_hat_xz[:, 0]], axis=1)

    vector_basis = np.zeros((E, 2, 3), dtype=np.float64)
    vector_basis[:, 0, 0] = r_hat_xz[:, 0]
    vector_basis[:, 0, 2] = r_hat_xz[:, 1]
    vector_basis[:, 1, 0] = j_hat_xz[:, 0]
    vector_basis[:, 1, 2] = j_hat_xz[:, 1]

    return edge_scalar.astype(np.float32), vector_basis.astype(np.float32)



BOUNDARY_SCALAR_DIM = 15


def build_boundary_relations(
    furniture_positions: np.ndarray,
    furniture_yaw: np.ndarray,
    furniture_full_wdh: np.ndarray,
    furniture_geo_index: np.ndarray,
    segments_xz: np.ndarray,
    tangent_xz: np.ndarray,
    inward_normal_xz: np.ndarray,
    segment_length: np.ndarray,
    room_diag_D: float,
    room_height_y: float,
) -> Dict[str, np.ndarray]:
  
    F = int(np.asarray(furniture_positions).reshape(-1, 3).shape[0])
    S = int(np.asarray(segments_xz).reshape(-1, 2, 2).shape[0])

    D = max(float(room_diag_D), EPS)
    H = max(float(room_height_y), EPS)

    if F == 0 or S == 0:
        return {
            "boundary_scalar": np.zeros((0, BOUNDARY_SCALAR_DIM), dtype=np.float32),
            "boundary_vector_basis": np.zeros((0, 3, 3), dtype=np.float32),
            "boundary_target_index": np.zeros((0,), dtype=np.int64),
            "boundary_segment_index": np.zeros((0,), dtype=np.int64),
            "boundary_length": np.zeros((0,), dtype=np.float32),
        }

    positions = np.asarray(furniture_positions, dtype=np.float64).reshape(F, 3)
    yaw = np.asarray(furniture_yaw, dtype=np.float64).reshape(F)
    wdh = np.asarray(furniture_full_wdh, dtype=np.float64).reshape(F, 3)
    geo_index = np.asarray(furniture_geo_index, dtype=np.int64).reshape(F)

    segs = np.asarray(segments_xz, dtype=np.float64).reshape(S, 2, 2)
    tans = np.asarray(tangent_xz, dtype=np.float64).reshape(S, 2)
    nrms = np.asarray(inward_normal_xz, dtype=np.float64).reshape(S, 2)
    seg_len = np.asarray(segment_length, dtype=np.float64).reshape(S)

    f_idx = np.repeat(np.arange(F, dtype=np.int64), S)
    s_idx = np.tile(np.arange(S, dtype=np.int64), F)

    centers_xz = positions[f_idx][:, [0, 2]]
    p0 = segs[s_idx, 0, :]
    p1 = segs[s_idx, 1, :]
    closest, distance = closest_point_on_segment(centers_xz, p0, p1)

    width_axis, depth_axis = local_axes_from_yaw(yaw)
    u = width_axis[f_idx]
    v = depth_axis[f_idx]

    def to_local(vec_xz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.sum(vec_xz * u, axis=1), np.sum(vec_xz * v, axis=1)

    start_lw, start_ld = to_local(p0 - centers_xz)
    end_lw, end_ld = to_local(p1 - centers_xz)
    close_lw, close_ld = to_local(closest - centers_xz)
    tan_lw, tan_ld = to_local(tans[s_idx])
    nrm_lw, nrm_ld = to_local(nrms[s_idx])

    f_wdh = wdh[f_idx]

    boundary_scalar = np.stack(
        [
            start_lw / D,
            start_ld / D,
            end_lw / D,
            end_ld / D,
            close_lw / D,
            close_ld / D,
            distance / D,
            seg_len[s_idx] / D,
            f_wdh[:, 0] / D,
            f_wdh[:, 1] / D,
            f_wdh[:, 2] / H,
            tan_lw,
            tan_ld,
            nrm_lw,
            nrm_ld,
        ],
        axis=1,
    )

    R = boundary_scalar.shape[0]
    to_closest = closest - centers_xz
    to_closest_norm = np.linalg.norm(to_closest, axis=1)
    safe = to_closest_norm > 1e-6
    dir_closest = np.zeros_like(to_closest)
    dir_closest[safe] = to_closest[safe] / to_closest_norm[safe, None]

    vector_basis = np.zeros((R, 3, 3), dtype=np.float64)
    vector_basis[:, 0, 0] = dir_closest[:, 0]
    vector_basis[:, 0, 2] = dir_closest[:, 1]
    vector_basis[:, 1, 0] = tans[s_idx][:, 0]
    vector_basis[:, 1, 2] = tans[s_idx][:, 1]
    vector_basis[:, 2, 0] = nrms[s_idx][:, 0]
    vector_basis[:, 2, 2] = nrms[s_idx][:, 1]

    return {
        "boundary_scalar": boundary_scalar.astype(np.float32),
        "boundary_vector_basis": vector_basis.astype(np.float32),
        "boundary_target_index": geo_index[f_idx].astype(np.int64),
        "boundary_segment_index": s_idx.astype(np.int64),
        "boundary_length": seg_len[s_idx].astype(np.float32),
    }

