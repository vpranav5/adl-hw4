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

    captions = []

    # Find ego kart (center kart or instance_id 0)
    ego = None
    for k in karts:
        if k.get("is_center_kart") or k["instance_id"] == 0:
            ego = k
            break

    # 1. Ego car caption
    if ego:
        captions.append(f"{ego['kart_name']} is the ego car.")

    # 2. Counting caption
    captions.append(f"There are {len(karts)} karts in the scenario.")

    # 3. Track name caption
    captions.append(f"The track is {track_name}.")

    # 4. Relative position captions - only makes sense if we have an ego kart
    if ego:
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
    split_dir = Path("data") / split
    
    # Default output file
    if output_file is None:
        output_file = split_dir / "all_captions.json"
    else:
        output_file = Path(output_file)
    
    all_caption_pairs = []
    
    # Process all info files in the split
    info_files = sorted(split_dir.glob("*_info.json"))
    print(f"Processing {len(info_files)} info files from {split_dir}")
    
    processed_count = 0
    error_count = 0
    
    for info_path in info_files:
        try:
            # Load info to get number of views
            with open(info_path) as f:
                info = json.load(f)
            
            num_views = len(info.get("views", []))
            
            # Process each view
            for view_index in range(num_views):
                try:
                    # Generate captions for this view
                    captions = generate_captions(str(info_path), view_index)
                    
                    if not captions:
                        continue
                    
                    # Construct image file path relative to data directory
                    base_name = info_path.stem.replace("_info", "")
                    img_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"
                    
                    # Create caption pairs
                    for caption in captions:
                        all_caption_pairs.append({
                            "image_file": img_file,
                            "caption": caption
                        })
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} views, generated {len(all_caption_pairs)} caption pairs so far...")
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Only print first few errors
                        print(f"Error processing {info_path.name} view {view_index}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error loading {info_path}: {e}")
            continue
    
    # Save to JSON
    with open(output_file, "w") as f:
        json.dump(all_caption_pairs, f, indent=2)
    
    # Print statistics
    print(f"\nGenerated {len(all_caption_pairs)} caption pairs")
    print(f"Saved to {output_file}")
    print(f"Processed {processed_count} views successfully")
    if error_count > 0:
        print(f"Encountered {error_count} errors during processing")
    
    # Calculate statistics
    unique_images = len(set(pair["image_file"] for pair in all_caption_pairs))
    if unique_images > 0:
        avg_captions = len(all_caption_pairs) / unique_images
        print(f"Unique images: {unique_images}")
        print(f"Average captions per image: {avg_captions:.2f}")
    
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
