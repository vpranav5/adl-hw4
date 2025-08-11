from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info, generate_qa_pairs

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100):
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    captions = [
        f"The track is {track_name}.",
        f"There are {len(karts)} karts."
    ]

    # Try to find ego kart, but still proceed if not found
    ego = next((k for k in karts if k.get("is_center_kart") or k["instance_id"] == 0), None)

    MARGIN = 6
    def rel(dx, dy):
        parts = []
        if dy <= -MARGIN: parts.append("front")
        elif dy >= MARGIN: parts.append("back")
        if dx <= -MARGIN: parts.append("left")
        elif dx >= MARGIN: parts.append("right")
        return " and ".join(parts)

    if ego:
        for k in karts:
            if k["instance_id"] == ego["instance_id"]:
                continue
            r = rel(k["center"][0] - ego["center"][0], k["center"][1] - ego["center"][1])
            if r:
                captions.append(f"{k['kart_name']} is {r} of the ego car.")
    else:
        # Describe visible karts without ego reference
        for k in karts:
            captions.append(f"A {k['kart_name']} is visible on the track.")

    return captions

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

def generate(split: str = "train", output_file: str = None, num_views: int = None):
    split_dir = Path("data") / split
    output_file = output_file or (split_dir / "all_captions.json")
    all_caps = []

    for info_path in sorted(split_dir.glob("*_info.json")):
        with open(info_path) as f:
            data = json.load(f)
        max_views = len(data.get("views", []))
        views_to_use = num_views if num_views else max_views

        for view_index in range(views_to_use):
            caps = generate_caption(str(info_path), view_index)
            if not caps:
                continue
            img_file = f"{split}/{info_path.stem.replace('_info', f'_{view_index:02d}_im.jpg')}"
            for c in caps:
                all_caps.append({"image_file": img_file, "caption": c})

    with open(output_file, "w") as f:
        json.dump(all_caps, f, indent=2)

    print(f"Saved {len(all_caps)} captions to {output_file}")
    return all_caps


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate})


if __name__ == "__main__":
    main()
