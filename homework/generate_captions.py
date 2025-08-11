from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info, generate_qa_pairs

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json


# def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
#     """
#     Generate caption for a specific view.
#     """
#     # 1. Ego car
#     # {kart_name} is the ego car.

#     # 2. Counting
#     # There are {num_karts} karts in the scenario.

#     # 3. Track name
#     # The track is {track_name}.

#     # 4. Relative position
#     # {kart_name} is {position} of the ego car.

#     karts = extract_kart_objects(info_path, view_index, img_width, img_height)
#     track_name = extract_track_info(info_path)
#     ego = next((k for k in karts if k.get("is_center_kart") or k["instance_id"] == 0), None)
#     if ego is None:
#         return []

#     MARGIN = 6
#     caps = [f"The track is {track_name}.",
#             f"There are {len(karts)} karts."]
#     if not ego["kart_name"].startswith("kart_"):
#         caps.append(f"The ego kart is {ego['kart_name']}.")

#     def rel(dx, dy):
#         parts = []
#         if dy <= -MARGIN: parts.append("front")
#         elif dy >= MARGIN: parts.append("back")
#         if dx <= -MARGIN: parts.append("left")
#         elif dx >= MARGIN: parts.append("right")
#         return " and ".join(parts)

#     for k in karts:
#         if k["instance_id"] == ego["instance_id"]:
#             continue
#         dx = k["center"][0] - ego["center"][0]
#         dy = k["center"][1] - ego["center"][1]
#         r = rel(dx, dy)
#         if r:
#             caps.append(f"{k['kart_name']} is {r} of the ego car.")

#     return caps

# def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
#     """
#     Generate caption for a specific view.
#     """
#     karts = extract_kart_objects(info_path, view_index, img_width, img_height)
#     track_name = extract_track_info(info_path)
#     ego = next((k for k in karts if k.get("is_center_kart") or k["instance_id"] == 0), None)
#     if ego is None:
#         return []

#     MARGIN = 6
#     caps = [f"The track is {track_name}.",
#             f"There are {len(karts)} karts."]
#     if not ego["kart_name"].startswith("kart_"):
#         caps.append(f"The ego kart is {ego['kart_name']}.")

#     def rel(dx, dy):
#         parts = []
#         if dy <= -MARGIN: parts.append("front")
#         elif dy >= MARGIN: parts.append("back")
#         if dx <= -MARGIN: parts.append("left")
#         elif dx >= MARGIN: parts.append("right")
#         return " and ".join(parts)

#     for k in karts:
#         if k["instance_id"] == ego["instance_id"]:
#             continue
#         dx = k["center"][0] - ego["center"][0]
#         dy = k["center"][1] - ego["center"][1]
#         r = rel(dx, dy)
#         if r:
#             caps.append(f"{k['kart_name']} is {r} of the ego car.")

#     return caps

def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate multiple varied captions for a specific view to reach ~200k total captions.
    """
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    ego = next((k for k in karts if k.get("is_center_kart") or k["instance_id"] == 0), None)
    if ego is None:
        return []

    MARGIN = 6
    caps = []
    
    # Basic scene descriptions
    caps.append(f"The track is {track_name}.")
    caps.append(f"There are {len(karts)} karts in the scene.")
    caps.append(f"A racing scene on {track_name}.")
    caps.append(f"Kart racing taking place on {track_name}.")
    
    # More varied counting and scene descriptions
    if len(karts) > 1:
        caps.append(f"A total of {len(karts)} karts are visible.")
        caps.append(f"The scene shows {len(karts)} racing karts.")
        caps.append(f"{len(karts)} karts racing on {track_name}.")
        
    if len(karts) > 2:
        caps.append("Multiple karts are positioned around the track.")
        caps.append("Several racing karts are visible in this view.")
        caps.append("Multiple karts competing in the race.")
    
    # Ego kart descriptions
    if not ego["kart_name"].startswith("kart_"):
        caps.append(f"The ego kart is {ego['kart_name']}.")
        caps.append(f"{ego['kart_name']} is the main kart in view.")
        caps.append(f"{ego['kart_name']} racing on {track_name}.")

    def rel(dx, dy):
        parts = []
        if dy <= -MARGIN: parts.append("front")
        elif dy >= MARGIN: parts.append("back")
        if dx <= -MARGIN: parts.append("left")
        elif dx >= MARGIN: parts.append("right")
        return " and ".join(parts)

    # Positional descriptions for other karts
    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue
        dx = k["center"][0] - ego["center"][0]
        dy = k["center"][1] - ego["center"][1]
        r = rel(dx, dy)
        if r:
            caps.append(f"{k['kart_name']} is {r} of the ego car.")
            caps.append(f"{k['kart_name']} can be seen {r} of the main kart.")
            caps.append(f"{k['kart_name']} positioned {r} relative to ego kart.")
            
    # Additional varied scene descriptions
    caps.append(f"Racing action on {track_name} with karts.")
    caps.append(f"Kart racing competition on {track_name}.")
    caps.append("Racing karts on the track.")
    caps.append("Karts competing in a race.")
    
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


# def generate(split: str = "train", output_file: str = None, num_views: int = 5):
#     """
#     Generate and save QA pairs for all files in a dataset split.

#     Args:
#         split: One of 'train', 'valid', etc. (defaults to 'train')
#         output_file: Path to save the output JSON file (default: data/{split}/all_qa_pairs.json)
#         num_views: How many views to process per frame
#     """
#     split_dir = Path("data") / split
#     output_file = output_file or (split_dir / "all_captions.json")  # ends with _captions.json

#     all_caps = []
#     info_files = sorted(split_dir.glob("*_info.json"))

#     for info_path in info_files:
#         for view_index in range(num_views):
#             caps = generate_caption(str(info_path), view_index)
#             if not caps:
#                 continue
#             image_file = f"{split}/{info_path.stem.replace('_info', f'_{view_index:02d}_im.jpg')}"
#             for c in caps:
#                 all_caps.append({"image_file": image_file, "caption": c})

#     with open(output_file, "w") as f:
#         json.dump(all_caps, f, indent=2)

#     print(f"Saved {len(all_caps)} captions to {output_file}")

def generate(split: str = "train", output_file: str = None, num_views: int = 5):
    """
    Generate and save caption pairs for all files in a dataset split.
    Optimized to generate many captions per image to reach ~200k total.

    Args:
        split: One of 'train', 'valid', etc. (defaults to 'train')
        output_file: Path to save the output JSON file (default: data/{split}/all_captions.json)
        num_views: How many views to process per frame
    """
    split_dir = Path("data") / split
    output_file = output_file or (split_dir / "all_captions.json")

    all_caps = []
    info_files = sorted(split_dir.glob("*_info.json"))
    
    print(f"Processing {len(info_files)} info files with {num_views} views each...")
    print(f"Target: Generate around 200k captions")

    for i, info_path in enumerate(info_files):
        if i % 100 == 0:  # Progress indicator every 100 files
            print(f"Processed {i}/{len(info_files)} files, generated {len(all_caps)} captions so far")
            
        for view_index in range(num_views):
            caps = generate_caption(str(info_path), view_index)
            if not caps:
                continue
                
            # Create the image file path in the format expected
            image_file = f"{split}/{info_path.stem.replace('_info', f'_{view_index:02d}_im.jpg')}"
            
            for c in caps:
                all_caps.append({"image_file": image_file, "caption": c})

    # Final progress report
    print(f"Processing complete!")
    print(f"Total captions generated: {len(all_caps)}")
    print(f"Average captions per image: {len(all_caps) / (len(info_files) * num_views):.1f}")

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(all_caps, f, indent=2)

    print(f"Captions saved to {output_file}")
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
