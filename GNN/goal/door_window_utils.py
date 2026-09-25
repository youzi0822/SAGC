
import numpy as np
from typing import List, Dict, Tuple, Optional


def extract_door_window_nodes(
    door_win_data: Dict,
    floor_plan_centroid: np.ndarray,
    room_type: str = "bedroom"
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
   
    
    if door_win_data is None:
        return [], np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    if 'sample' not in door_win_data:
        return [], np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    sample = door_win_data['sample']

    if 'openings' not in sample:
        return [], np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    openings = sample['openings']

    # 
    if not isinstance(openings, list) or len(openings) == 0:
        return [], np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

   
    room_shell = sample.get('room_shell', {})
    width_x = room_shell.get('width_x', sample.get('width_x', 0.0))
    length_z = room_shell.get('length_z', sample.get('length_z', 0.0))
    offset_x = float(width_x) / 2.0
    offset_z = float(length_z) / 2.0

    categories = []
    positions_list = []
    sizes_list = []
    thetas_list = []

    for opening in openings:
     
        if not isinstance(opening, dict):
            continue

        opening_type = opening.get('type', None)
        if opening_type not in ['door', 'window']:
            continue

        categories.append(opening_type)

        center_local = opening.get('center_local', None)
        if center_local is None:
           
            center_world = opening.get('center_world', None)
            if center_world is not None and floor_plan_centroid is not None:
                center_local = [
                    center_world[0] - floor_plan_centroid[0],
                    center_world[1] - floor_plan_centroid[1],
                    center_world[2] - floor_plan_centroid[2],
                ]
            else:
             
                center_local = [0.0, 0.0, 0.0]

        pos_x = float(center_local[0]) - offset_x
        pos_y = float(center_local[1])              
        pos_z = float(center_local[2]) - offset_z

        positions_list.append([pos_x, pos_y, pos_z])

        size = opening.get('size', None)
        if size is None or len(size) < 3:
            
            if opening_type == 'door':
                size = [0.9, 2.0, 0.12]  # [width, height, thickness]
            else:  # window
                size = [1.2, 1.2, 0.12]

      
        half_w = float(size[0]) / 2.0  # opening_width  → half_w
        half_h = float(size[1]) / 2.0  # opening_height → half_h
        half_d = float(size[2]) / 2.0  # wall_thickness → half_d

        sizes_list.append([half_w, half_d, half_h])  

        
        yaw_norm = opening.get('yaw_norm', None)
        if yaw_norm is not None and isinstance(yaw_norm, dict):
            cos_yaw = yaw_norm.get('cos', 1.0)
            sin_yaw = yaw_norm.get('sin', 0.0)
        else:
           
            yaw_rad = opening.get('yaw_rad', 0.0)
            cos_yaw = np.cos(yaw_rad)
            sin_yaw = np.sin(yaw_rad)

        thetas_list.append([cos_yaw, sin_yaw])

  
    if len(categories) == 0:
        return [], np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

   
    positions = np.array(positions_list, dtype=np.float32)
    sizes = np.array(sizes_list, dtype=np.float32)
    thetas = np.array(thetas_list, dtype=np.float32)

    return categories, positions, sizes, thetas


def combine_furniture_and_door_window(
    furniture_categories: List[str],
    furniture_positions: np.ndarray,
    furniture_sizes: np.ndarray,
    furniture_thetas: np.ndarray,
    door_window_categories: List[str],
    door_window_positions: np.ndarray,
    door_window_sizes: np.ndarray,
    door_window_thetas: np.ndarray,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, int]:
  
    num_furniture = len(furniture_categories)
    num_door_window = len(door_window_categories)

    if num_door_window == 0:
        
        return (
            furniture_categories,
            furniture_positions,
            furniture_sizes,
            furniture_thetas,
            num_furniture,
        )


    all_categories = furniture_categories + door_window_categories
    all_positions = np.concatenate([furniture_positions, door_window_positions], axis=0)
    all_sizes = np.concatenate([furniture_sizes, door_window_sizes], axis=0)
    all_thetas = np.concatenate([furniture_thetas, door_window_thetas], axis=0)

    return all_categories, all_positions, all_sizes, all_thetas, num_furniture


if __name__ == "__main__":
    # 
    from pathlib import Path
    from supervised_dataset import load_door_win_json, load_boxes_npz

    PERTURBED_DIR = Path("SAGC/GNN/goal/collect_data/perturbed")
    sample_dir = PERTURBED_DIR / "0a9f5311-49e1-414c-ba7b-b42a171459a3_SecondBedroom-18509__boundary_rotate"

    door_win_data = load_door_win_json(sample_dir / "door_win.json")
    boxes_data = load_boxes_npz(sample_dir / "boxes.npz")

    floor_plan_centroid = boxes_data.get("floor_plan_centroid", np.array([0, 0, 0]))

    dw_categories, dw_positions, dw_sizes, dw_thetas = extract_door_window_nodes(
        door_win_data, floor_plan_centroid
    )

    print(f"{len(dw_categories)}")
    for i, cat in enumerate(dw_categories):
        print(f"  {i}: {cat}, pos={dw_positions[i]}, size={dw_sizes[i]}")
