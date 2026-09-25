

import numpy as np
from scipy.spatial.distance import cdist



EDGE_TYPE_RADIUS       = 0   
EDGE_TYPE_VIRTUAL      = 1   
EDGE_TYPE_PROJ_OVERLAP = 2   
EDGE_TYPE_DOOR_WIN     = 3   


def build_node_features(categories, cat_to_idx, positions, sizes, thetas):
  
    N = len(categories)
    C = len(cat_to_idx)
    assert positions.shape == (N, 3), f"positions [N,3]，got {positions.shape}"
    assert sizes.shape    == (N, 3), f"sizes [N,3]，got {sizes.shape}"
    assert thetas.shape   == (N, 2), f"thetas  [N,2]=(cosθ,sinθ)，got {thetas.shape}"

    node_feats = np.zeros((N, C + 8), dtype=np.float32)
    for i in range(N):
        onehot = np.zeros(C, dtype=np.float32)
        onehot[cat_to_idx[categories[i]]] = 1.0
        x, y, z   = positions[i]
        w, d, h   = sizes[i]
        cos_t, sin_t = thetas[i]
        node_feats[i] = np.concatenate([onehot, [x, y, z, cos_t, sin_t, w, d, h]])

    return node_feats


def compute_bbox_overlap(pos_i, size_i, pos_j, size_j, height_i, height_j):
   
    half_i = np.array(size_i)
    half_j = np.array(size_j)
    min_i, max_i = pos_i - half_i, pos_i + half_i
    min_j, max_j = pos_j - half_j, pos_j + half_j

    inter_w = max(0.0, min(max_i[0], max_j[0]) - max(min_i[0], min_j[0]))
    inter_d = max(0.0, min(max_i[1], max_j[1]) - max(min_i[1], min_j[1]))
    inter_area = inter_w * inter_d

    area_i = (2 * size_i[0]) * (2 * size_i[1])
    area_j = (2 * size_j[0]) * (2 * size_j[1])
    union_area = area_i + area_j - inter_area
    if union_area <= 1e-8:
        return 0.0
    iou_2d = inter_area / union_area

    
    height_overlap = min(height_i, height_j) / max(height_i, height_j) \
        if max(height_i, height_j) > 1e-8 else 0.0
    return iou_2d * height_overlap


def _dtheta_cossin(cos_i, sin_i, cos_j, sin_j):
   
    cos_dt = float(cos_j * cos_i + sin_j * sin_i)
    sin_dt = float(sin_j * cos_i - cos_j * sin_i)
    return cos_dt, sin_dt


def build_radius_edges(positions, sizes, thetas, radius):
   
    N = len(positions)
    xz = positions[:, [0, 2]]       
    D = cdist(xz, xz)
    np.fill_diagonal(D, np.inf)

    edge_pairs = []
    edge_feats = []
    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] <= radius:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]   
                dz = positions[j, 2] - positions[i, 2]
                cos_dt, sin_dt = _dtheta_cossin(
                    thetas[i, 0], thetas[i, 1],
                    thetas[j, 0], thetas[j, 1]
                )
                overlap = compute_bbox_overlap(
                    xz[i], sizes[i, :2],
                    xz[j], sizes[j, :2],
                    height_i=sizes[i, 2], height_j=sizes[j, 2]
                )
                edge_pairs.append((i, j))
                edge_feats.append(np.array([dx, dy, dz, cos_dt, sin_dt, overlap],
                                           dtype=np.float32))
    return edge_pairs, edge_feats


def build_projection_overlap_edges(positions, sizes):
   
    N = len(positions)
    xz = positions[:, [0, 2]]

    edge_pairs = []
    edge_feats = []
    for i in range(N):
        for j in range(i + 1, N):
           
            iou = compute_bbox_overlap(
                xz[i], sizes[i, :2],
                xz[j], sizes[j, :2],
                height_i=1.0, height_j=1.0   
            )
            if iou > 0.0:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                edge_pairs.append((i, j))
             
                edge_feats.append(np.array([dx, dy, dz, 1.0, 0.0, iou],
                                           dtype=np.float32))
    return edge_pairs, edge_feats


def build_scene_graph(categories, cat_to_idx, positions, sizes, thetas,
                      radius=1.6, num_furniture=None, dw_radius=3.0):

    N = len(categories)
    M = num_furniture if num_furniture is not None else N   

    node_feats = build_node_features(categories, cat_to_idx, positions, sizes, thetas)

   
    virtual_feat = np.zeros((1, node_feats.shape[1]), dtype=np.float32)
    node_feats = np.concatenate([node_feats, virtual_feat], axis=0)

  
    furn_pos   = positions[:M]
    furn_sizes = sizes[:M]
    furn_theta = thetas[:M]
    radius_pairs, radius_feats = build_radius_edges(furn_pos, furn_sizes, furn_theta, radius)
    proj_pairs,   proj_feats   = build_projection_overlap_edges(furn_pos, furn_sizes)

    
    dw_pairs  = []   
    dw_feats  = []   
    if M < N:
        xz_all = positions[:, [0, 2]]
        for i in range(M):          #
            for j in range(M, N):   # 
                dist_xz = np.sqrt(((xz_all[i] - xz_all[j]) ** 2).sum())
                if dist_xz <= dw_radius:
                    dx = float(positions[j, 0] - positions[i, 0])
                    dy = float(positions[j, 1] - positions[i, 1])
                    dz = float(positions[j, 2] - positions[i, 2])
                    cos_dt, sin_dt = _dtheta_cossin(
                        thetas[i, 0], thetas[i, 1],
                        thetas[j, 0], thetas[j, 1]
                    )
                    overlap = compute_bbox_overlap(
                        xz_all[i], sizes[i, :2],
                        xz_all[j], sizes[j, :2],
                        height_i=sizes[i, 2], height_j=sizes[j, 2]
                    )
                    dw_pairs.append((i, j))
                    dw_feats.append(np.array([dx, dy, dz, cos_dt, sin_dt, overlap],
                                             dtype=np.float32))

    edge_index_list = []
    edge_feat_list  = []
    edge_type_list  = []

    
    for (i, j), feat in zip(radius_pairs, radius_feats):
        rev_feat = feat.copy()
        rev_feat[0:3] *= -1.0   #
        rev_feat[4]   *= -1.0   # 
        edge_index_list += [(i, j), (j, i)]
        edge_feat_list  += [np.append(feat, EDGE_TYPE_RADIUS),
                            np.append(rev_feat, EDGE_TYPE_RADIUS)]
        edge_type_list  += [EDGE_TYPE_RADIUS, EDGE_TYPE_RADIUS]

    
    for (i, j), feat in zip(proj_pairs, proj_feats):
        rev_feat = feat.copy()
        rev_feat[:3] = -feat[:3]   
        edge_index_list += [(i, j), (j, i)]
        edge_feat_list  += [np.append(feat, EDGE_TYPE_PROJ_OVERLAP),
                            np.append(rev_feat, EDGE_TYPE_PROJ_OVERLAP)]
        edge_type_list  += [EDGE_TYPE_PROJ_OVERLAP, EDGE_TYPE_PROJ_OVERLAP]

    
    for (i, j), feat in zip(dw_pairs, dw_feats):
        rev_feat = feat.copy()
        rev_feat[0:3] *= -1.0
        rev_feat[4]   *= -1.0
        edge_index_list += [(i, j), (j, i)]
        edge_feat_list  += [np.append(feat, EDGE_TYPE_DOOR_WIN),
                            np.append(rev_feat, EDGE_TYPE_DOOR_WIN)]
        edge_type_list  += [EDGE_TYPE_DOOR_WIN, EDGE_TYPE_DOOR_WIN]

  
    VNODE = N
    zero_feat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, EDGE_TYPE_VIRTUAL],
                         dtype=np.float32)
    for i in range(N):
        edge_index_list += [(i, VNODE), (VNODE, i)]
        edge_feat_list  += [zero_feat, zero_feat]
        edge_type_list  += [EDGE_TYPE_VIRTUAL, EDGE_TYPE_VIRTUAL]

    edge_index = np.array(edge_index_list, dtype=np.int64).T   # [2, E]
    edge_feats = np.array(edge_feat_list,  dtype=np.float32)   # [E, 7]
    edge_types = np.array(edge_type_list,  dtype=np.int64)     # [E]

    return node_feats, edge_index, edge_feats, edge_types


def to_pyg_data(node_feats, edge_index, edge_feats, edge_types, num_real_nodes):
   
    import torch
    from torch_geometric.data import Data

    N_total = node_feats.shape[0]   # M 

  
    is_virtual = torch.zeros(N_total, dtype=torch.bool)
    is_virtual[num_real_nodes] = True

    return Data(
        x          = torch.tensor(node_feats, dtype=torch.float32),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float32),
        edge_type  = torch.tensor(edge_types, dtype=torch.long),   # 
        is_virtual = is_virtual,                                    # 
    )


