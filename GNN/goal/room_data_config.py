"""Shared room-specific paths and the exact boxes.npz category order."""

import json
from pathlib import Path

ROOM_TYPES = ("bedroom", "diningroom", "livingroom")
GOAL_DIR = Path(__file__).resolve().parent
PROCESSED_ROOT = Path(
    "SAGC/data/3d_front_processed"
)


def original_root(room_type):
    if room_type not in ROOM_TYPES:
        raise ValueError(f"Unsupported room type: {room_type}")
    return PROCESSED_ROOT / f"{room_type}s_objfeats_32_64"


def output_root(room_type):
    return GOAL_DIR / "collect_data" / ("more_correct" if room_type == "bedroom" else room_type)


def load_class_names(dataset_root):
    # Preserve the stored order, including start/end. Graph vocabularies have
    # a different contract (replace start/end with door/window).
    with (Path(dataset_root) / "dataset_stats.txt").open() as handle:
        labels = json.load(handle)["class_labels"]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("dataset_stats.txt has an empty or duplicate category list")
    return labels


def add_room_arguments(parser):
    parser.add_argument("--room-type", choices=ROOM_TYPES, default="bedroom")
    parser.add_argument("--dataset-root", type=Path, help="Original/staged boxes.npz scene root")
