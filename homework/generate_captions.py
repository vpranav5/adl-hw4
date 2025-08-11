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
    # karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    # track_name = extract_track_info(info_path)
    # ego = next((k for k in karts if k.get("is_center_kart") or k["instance_id"] == 0), None)
    # if ego is None:
    #     return []

    # MARGIN = 6
    # caps = [f"The track is {track_name}.",
    #         f"There are {len(karts)} karts."]
    # if not ego["kart_name"].startswith("kart_"):
    #     caps.append(f"The ego kart is {ego['kart_name']}.")

    # def rel(dx, dy):
    #     parts = []
    #     if dy <= -MARGIN: parts.append("front")
    #     elif dy >= MARGIN: parts.append("back")
    #     if dx <= -MARGIN: parts.append("left")
    #     elif dx >= MARGIN: parts.append("right")
    #     return " and ".join(parts)

    # for k in karts:
    #     if k["instance_id"] == ego["instance_id"]:
    #         continue
    #     dx = k["center"][0] - ego["center"][0]
    #     dy = k["center"][1] - ego["center"][1]
    #     r = rel(dx, dy)
    #     if r:
    #         caps.append(f"{k['kart_name']} is {r} of the ego car.")

    # return caps
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    # Find ego kart
    ego = None
    for k in karts:
        if k.get("is_center_kart") or k["instance_id"] == 0:
            ego = k
            break
    
    if ego is None:
        return []  # Skip views without ego kart
    
    captions = []
    
    # Basic captions - always include these
    captions.append(f"The track is {track_name}.")
    captions.append(f"There are {len(karts)} karts in the scenario.")
    
    # Ego kart caption
    captions.append(f"{ego['kart_name']} is the ego car.")
    
    # Relative position captions
    MARGIN = 6
    
    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue
        
        dx = k["center"][0] - ego["center"][0]
        dy = k["center"][1] - ego["center"][1]
        
        position_parts = []
        if dy <= -MARGIN:
            position_parts.append("in front")
        elif dy >= MARGIN:
            position_parts.append("behind")
        
        if dx <= -MARGIN:
            position_parts.append("to the left")
        elif dx >= MARGIN:
            position_parts.append("to the right")
        
        if position_parts:
            position = " and ".join(position_parts)
            captions.append(f"{k['kart_name']} is {position} of the ego car.")
    
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
    # split_dir = Path("data") / split
    # output_file = output_file or (split_dir / "all_captions.json")  # ends with _captions.json

    # all_caps = []
    # info_files = sorted(split_dir.glob("*_info.json"))

    # for info_path in info_files:
    #     for view_index in range(num_views):
    #         caps = generate_caption(str(info_path), view_index)
    #         if not caps:
    #             continue
    #         image_file = f"{split}/{info_path.stem.replace('_info', f'_{view_index:02d}_im.jpg')}"
    #         for c in caps:
    #             all_caps.append({"image_file": image_file, "caption": c})

    # with open(output_file, "w") as f:
    #     json.dump(all_caps, f, indent=2)

    # print(f"Saved {len(all_caps)} captions to {output_file}")

    split_dir = Path("data") / split
    
    if output_file is None:
        output_file = split_dir / "all_captions.json"
    else:
        output_file = Path(output_file)
    
    all_caption_pairs = []
    info_files = sorted(split_dir.glob("*_info.json"))
    
    print(f"Processing {len(info_files)} info files from {split_dir}")
    
    processed_count = 0
    skipped_count = 0
    
    for info_path in info_files:
        # Process each view
        for view_index in range(num_views):
            # Check if image exists
            base_name = info_path.stem.replace("_info", "")
            image_path = split_dir / f"{base_name}_{view_index:02d}_im.jpg"
            
            if not image_path.exists():
                continue
            
            # Generate captions
            captions = generate_caption(str(info_path), view_index)
            
            if not captions:
                skipped_count += 1
                continue
            
            # Create caption pairs
            image_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"
            
            for caption in captions:
                all_caption_pairs.append({
                    "image_file": image_file,
                    "caption": caption
                })
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} views, generated {len(all_caption_pairs)} captions...")
    
    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(all_caption_pairs, f, indent=2)
    
    print(f"\nGenerated {len(all_caption_pairs)} caption pairs")
    print(f"Saved to {output_file}")
    print(f"Processed {processed_count} views successfully")
    print(f"Skipped {skipped_count} views (no ego kart)")
    
    return all_caption_pairs


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate})


if __name__ == "__main__":
    main()
