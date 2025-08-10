from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info

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
    # ego_kart = next((k for k in karts if k["is_center_kart"] or k["instance_id"] == 0), None)

    # if ego_kart is None:
    #     return []
    ego = next((k for k in karts if k.get("is_center_kart", False) or k["instance_id"] == 0), None)
    if ego is None:
        return []

    MARGIN = 10
    caps = [
        f"The track is {track_name}.",
        f"There are {len(karts)} karts.",
        f"The ego kart is {ego['kart_name']}."
    ]

    def rel(dx, dy):
        parts = []
        if dx <= -MARGIN: parts.append("left")
        elif dx >= MARGIN: parts.append("right")
        if dy <= -MARGIN: parts.append("front")
        elif dy >= MARGIN: parts.append("behind")
        return " and ".join(parts)

    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue
        dx = k["center"][0] - ego["center"][0]
        dy = k["center"][1] - ego["center"][1]
        r = rel(dx, dy)
        if r:
            caps.append(f"{k['kart_name']} is {r} of the ego car.")

    # Return multiple short captions so training doesn’t learn verbosity
    return caps
    
    # captions = []
    # captions.append(f"There are {len(karts)} karts.")
    # captions.append(f"The ego kart is {ego_kart['kart_name']}.")
    # captions.append(f"The track is {track_name}.")

    # # Relative descriptions
    # for k in karts:
    #     if k["instance_id"] == ego_kart["instance_id"]:
    #         continue
    #     dx = k["center"][0] - ego_kart["center"][0]
    #     dy = k["center"][1] - ego_kart["center"][1]
    #     rel = []
    #     if dx < -10: rel.append("left")
    #     elif dx > 10: rel.append("right")
    #     if dy < -10: rel.append("front")
    #     elif dy > 10: rel.append("behind")
    #     if rel:
    #         captions.append(f"{k['kart_name']} is to the {' and '.join(rel)} of the ego car.")

    # return [" ".join(captions)]


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


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()
