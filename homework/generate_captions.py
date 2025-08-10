from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info, generate_qa_pairs

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    ego = next((k for k in karts if k.get("is_center_kart") or k["instance_id"] == 0), None)
    if ego is None:
        return []

    MARGIN = 6
    caps = [f"The track is {track_name}.",
            f"There are {len(karts)} karts."]
    if not ego["kart_name"].startswith("kart_"):
        caps.append(f"The ego kart is {ego['kart_name']}.")

    def rel(dx, dy):
        parts = []
        if dy <= -MARGIN: parts.append("front")
        elif dy >= MARGIN: parts.append("back")
        if dx <= -MARGIN: parts.append("left")
        elif dx >= MARGIN: parts.append("right")
        return " and ".join(parts)

    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue
        dx = k["center"][0] - ego["center"][0]
        dy = k["center"][1] - ego["center"][1]
        r = rel(dx, dy)
        if r:
            caps.append(f"{k['kart_name']} is {r} of the ego car.")

    return caps


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate(split: str = "train", output_file: str = None, num_views: int = 5):
    """
    Generate and save QA pairs for all files in a dataset split.

    Args:
        split: One of 'train', 'valid', etc. (defaults to 'train')
        output_file: Path to save the output JSON file (default: data/{split}/all_qa_pairs.json)
        num_views: How many views to process per frame
    """
    split_dir = Path("data") / split
    output_file = output_file or (split_dir / "all_captions.json")  # ends with _captions.json

    all_caps = []
    info_files = sorted(split_dir.glob("*_info.json"))

    for info_path in info_files:
        for view_index in range(num_views):
            caps = generate_caption(str(info_path), view_index)
            if not caps:
                continue
            image_file = f"{split}/{info_path.stem.replace('_info', f'_{view_index:02d}_im.jpg')}"
            for c in caps:
                all_caps.append({"image_file": image_file, "caption": c})

    with open(output_file, "w") as f:
        json.dump(all_caps, f, indent=2)

    print(f"Saved {len(all_caps)} captions to {output_file}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate})


if __name__ == "__main__":
    main()
