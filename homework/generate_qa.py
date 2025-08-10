import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


# def extract_kart_objects(
#     info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
# ) -> list:
#     """
#     Extract kart objects from the info.json file, including their center points and identify the center kart.
#     Filters out karts that are out of sight (outside the image boundaries).

#     Args:
#         info_path: Path to the corresponding info.json file
#         view_index: Index of the view to analyze
#         img_width: Width of the image (default: 150)
#         img_height: Height of the image (default: 100)

#     Returns:
#         List of kart objects, each containing:
#         - instance_id: The track ID of the kart
#         - kart_name: The name of the kart
#         - center: (x, y) coordinates of the kart's center
#         - is_center_kart: Boolean indicating if this is the kart closest to image center
#     """

#     with open(info_path) as f:
#         info = json.load(f)

#     detections = info["detections"][view_index]
#     names = info.get("names", {})

#     scale_x = img_width / 600
#     scale_y = img_height / 400

#     karts = []
#     for det in detections:
#         class_id, track_id, x1, y1, x2, y2 = det
#         if int(class_id) != 1:
#             continue

#         x1_scaled, y1_scaled = x1 * scale_x, y1 * scale_y
#         x2_scaled, y2_scaled = x2 * scale_x, y2 * scale_y
#         width, height = (x2_scaled - x1_scaled), (y2_scaled - y1_scaled)

#         if width < min_box_size or height < min_box_size:
#             continue

#         center = ((x1_scaled + x2_scaled) / 2, (y1_scaled + y2_scaled) / 2)
#         #cx, cy = (x1_scaled + x2_scaled) / 2.0, (y1_scaled + y2_scaled) / 2.0
#         track_label = int(track_id)
#         kart_name  = names.get(str(track_label), f"kart_{track_label}")
#         karts.append({
#             "instance_id": int(track_id),
#             "kart_name": kart_name,
#             "center": center
#         })

#     # Flag ego kart
#     for k in karts:
#         k["is_center_kart"] = (k["instance_id"] == 0)
    
#     # Fallback: if id 0 not visible in this view, optionally mark the closest to center as ego-ish
#     if not any(k["is_center_kart"] for k in karts):
#         img_cx, img_cy = img_width / 2, img_height / 2
#         if karts:
#             best = min(karts, key=lambda kk: (kk["center"][0]-img_cx)**2 + (kk["center"][1]-img_cy)**2)
#             best["is_center_kart"] = True

#     return karts

def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    with open(info_path) as f:
        info = json.load(f)

    detections = info["detections"][view_index]
    names = info.get("names", {})

    scale_x = img_width / 600
    scale_y = img_height / 400

    karts = []
    for det in detections:
        class_id, track_id, x1, y1, x2, y2 = det
        class_id = int(class_id)
        track_id = int(track_id)

        # only real karts with known names
        # if class_id != 1 or str(track_id) not in names:
        #     continue
        # new
        if (class_id not in (0, 1)) or (str(track_id) not in names):
            continue

        x1s, y1s = x1 * scale_x, y1 * scale_y
        x2s, y2s = x2 * scale_x, y2 * scale_y

        w, h = x2s - x1s, y2s - y1s
        if w < min_box_size or h < min_box_size:
            continue

        # drop fully out-of-frame
        if x2s < 0 or x1s > img_width or y2s < 0 or y1s > img_height:
            continue

        cx, cy = (x1s + x2s) / 2.0, (y1s + y2s) / 2.0
        karts.append({
            "instance_id": track_id,
            "kart_name": names[str(track_id)],
            "center": (cx, cy),
            "is_center_kart": False,
        })

    # mark exactly one ego: the center-most kart
    if karts:
        img_cx, img_cy = img_width / 2.0, img_height / 2.0
        karts.sort(key=lambda k: (k["center"][0]-img_cx)**2 + (k["center"][1]-img_cy)**2)
        karts[0]["is_center_kart"] = True

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "Unknown") # should be word track not track_name

def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # need an ego
    ego = next((k for k in karts if k.get("is_center_kart")), None)
    if ego is None:
        return []

    split_name = Path(info_path).parent.name  # "train" or "valid"
    image_file = f"{split_name}/{Path(info_path).stem.replace('_info', f'_{view_index:02d}_im.jpg')}"

    qs = []

    # 1) Ego identity (now a real name from `names`)
    if ego["kart_name"]:
        qs.append({"image_file": image_file,
                   "question": "What kart is the ego car?",
                   "answer": ego["kart_name"]})

    # 2) Total count (named karts only, since we filtered)
    qs.append({"image_file": image_file,
               "question": "How many karts are there in the scenario?",
               "answer": str(len(karts))})

    # 3) Track
    qs.append({"image_file": image_file,
               "question": "What track is this?",
               "answer": track_name})

    # 4) Relative positions (use one consistent margin + 'back')
    MARGIN = 14
    def parts(dx, dy):
        out = []
        if dy <= -MARGIN: out.append("front")
        elif dy >= MARGIN: out.append("back")
        if dx <= -MARGIN: out.append("left")
        elif dx >= MARGIN: out.append("right")
        return out

    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue
        dx = k["center"][0] - ego["center"][0]
        dy = k["center"][1] - ego["center"][1]
        p = parts(dx, dy)
        if p:
            qs.append({
                "image_file": image_file,
                "question": f"Where is {k['kart_name']} relative to the ego car?",
                "answer": " and ".join(p),
            })

    # 5) Counting by side (use the SAME margin thresholds)
    left = right = front = back = 0
    for k in karts:
        if k["instance_id"] == ego["instance_id"]:
            continue
        dx = k["center"][0] - ego["center"][0]
        dy = k["center"][1] - ego["center"][1]
        if dx <= -MARGIN: left += 1
        elif dx >= MARGIN: right += 1
        if dy <= -MARGIN: front += 1
        elif dy >= MARGIN: back += 1

    for direction, count in [("left", left), ("right", right), ("front", front), ("behind", back)]:
        # NOTE: the question text here uses 'behind' because that’s how the provided QA uses it,
        # but the 'Where is ...' answers must say 'back'.
        qs.append({
            "image_file": image_file,
            "question": f"How many karts are to the {direction} of the ego car?",
            "answer": str(count),
        })

    return qs

# def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
#     karts = extract_kart_objects(info_path, view_index, img_width, img_height)
#     track_name = extract_track_info(info_path)

#     # ✅ pick ego as id==0 if present; otherwise the one flagged by extract_kart_objects
#     ego = next((k for k in karts if k["instance_id"] == 0), None)
#     if ego is None:
#         ego = next((k for k in karts if k.get("is_center_kart")), None)
#     if ego is None:
#         return []

#     split_name = Path(info_path).parent.name
#     image_file = f"{split_name}/{Path(info_path).stem.replace('_info', f'_{view_index:02d}_im.jpg')}"

#     qs = []

#     # 1) Ego kart name
#     qs.append({"image_file": image_file,
#                "question": "What kart is the ego car?",
#                "answer": ego["kart_name"]})

#     # 2) Total count
#     qs.append({"image_file": image_file,
#                "question": "How many karts are there in the scenario?",
#                "answer": str(len(karts))})

#     # 3) Track
#     qs.append({"image_file": image_file,
#                "question": "What track is this?",
#                "answer": track_name})

#     # Shared margin for **all** geometry
#     MARGIN = 10  # px

#     def rel_from_dxdy(dx, dy):
#         # y increases downward in images → smaller y = 'front'
#         parts = []
#         if dy <= -MARGIN: parts.append("front")
#         elif dy >= MARGIN: parts.append("back")   # ✅ use 'back'
#         if dx <= -MARGIN: parts.append("left")
#         elif dx >= MARGIN: parts.append("right")
#         return parts  # list of tokens

#     # 4) Per-kart relative and **binary** questions
#     for k in karts:
#         if k["instance_id"] == ego["instance_id"]:
#             continue

#         dx = k["center"][0] - ego["center"][0]
#         dy = k["center"][1] - ego["center"][1]
#         parts = rel_from_dxdy(dx, dy)
#         if not parts:
#             # ambiguous → skip
#             continue

#         # 4a) "Where is X relative..."
#         qs.append({
#             "image_file": image_file,
#             "question": f"Where is {k['kart_name']} relative to the ego car?",
#             "answer": " and ".join(parts)
#         })

#         # 4b) Binary left/right (only if confidently lateral)
#         if "left" in parts or "right" in parts:
#             lr = "left" if "left" in parts else "right"
#             qs.append({
#                 "image_file": image_file,
#                 "question": f"Is {k['kart_name']} to the left or right of the ego car?",
#                 "answer": lr
#             })

#         # 4c) Binary front/back (only if confidently longitudinal)
#         if "front" in parts or "back" in parts:
#             fb = "front" if "front" in parts else "back"
#             qs.append({
#                 "image_file": image_file,
#                 "question": f"Is {k['kart_name']} in front of or back of the ego car?",
#                 "answer": fb
#             })

#     # 5) Counts by side (use the SAME margins)
#     left = right = front = back = 0
#     for k in karts:
#         if k["instance_id"] == ego["instance_id"]:
#             continue
#         dx = k["center"][0] - ego["center"][0]
#         dy = k["center"][1] - ego["center"][1]
#         if dx <= -MARGIN: left += 1
#         elif dx >= MARGIN: right += 1
#         if dy <= -MARGIN: front += 1
#         elif dy >= MARGIN: back += 1

#     for direction, count in [("left", left), ("right", right), ("front", front), ("back", back)]:
#         qs.append({
#             "image_file": image_file,
#             "question": f"How many karts are to the {direction} of the ego car?",
#             "answer": str(count),
#         })

#     return qs
# def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
#     karts = extract_kart_objects(info_path, view_index, img_width, img_height)
#     track_name = extract_track_info(info_path)

#     ego_kart = next((k for k in karts if k["is_center_kart"] or k["instance_id"] == 0), None)
#     if ego_kart is None:
#         return []
    
#     split_name = Path(info_path).parent.name  # "train" or "valid"
#     image_file = f"{split_name}/{Path(info_path).stem.replace('_info', f'_{view_index:02d}_im.jpg')}"
#     questions = []

#     # 1. Ego car
#     questions.append({"image_file": image_file, "question": "What kart is the ego car?", "answer": ego_kart["kart_name"]})

#     # 2. Total kart count
#     questions.append({"image_file": image_file, "question": "How many karts are there in the scenario?", "answer": str(len(karts))})

#     # 3. Track
#     questions.append({"image_file": image_file, "question": "What track is this?", "answer": track_name})

#     MARGIN = 10  # pixels; must match counting logic below

#     # ---- Q4: relative position per kart (vs ego) ----
#     for k in karts:
#         if k["instance_id"] == ego_kart["instance_id"]:
#             continue

#         dx = k["center"][0] - ego_kart["center"][0]
#         dy = k["center"][1] - ego_kart["center"][1]

#         rel_parts = []
#         if dx <= -MARGIN:
#             rel_parts.append("left")
#         elif dx >= MARGIN:
#             rel_parts.append("right")

#         # y increases downward; negative dy means 'front'
#         if dy <= -MARGIN:
#             rel_parts.append("front")
#         elif dy >= MARGIN:
#             rel_parts.append("behind")

#         if not rel_parts:
#             # too close to call; skip noisy label
#             continue

#         rel = " and ".join(rel_parts)
#         q = f"Where is {k['kart_name']} relative to the ego car?"
#         questions.append({"image_file": image_file, "question": q, "answer": rel})

#     # 5. Counting by position: one above is relative position, this is counting the position with the center kart
#     left = right = front = behind = 0
#     for k in karts:
#         if k["instance_id"] == ego_kart["instance_id"]:
#             continue
#         dx = k["center"][0] - ego_kart["center"][0]
#         dy = k["center"][1] - ego_kart["center"][1]
#         if dx < 0: left += 1 # try
#         if dx > 0: right += 1
#         if dy < 0: front += 1
#         if dy > 0: behind += 1

#     for direction, count in zip(["left", "right", "front", "behind"], [left, right, front, behind]):
#         questions.append({
#             "image_file": image_file,
#             "question": f"How many karts are to the {direction} of the ego car?",
#             "answer": str(count),
#         })

#     return questions


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate(split: str = "train", output_file: str = None, num_views: int = 5):
    """
    Generate and save QA pairs for all files in a dataset split.

    Args:
        split: One of 'train', 'valid', etc. (defaults to 'train')
        output_file: Path to save the output JSON file (default: data/{split}/all_qa_pairs.json)
        num_views: How many views to process per frame
    """
    split_dir = Path("data") / split
    output_file = output_file or (split_dir / "all_qa_pairs.json")

    all_qa_pairs = []
    info_files = sorted(split_dir.glob("*_info.json"))

    for info_path in info_files:
        for view_index in range(num_views):
            qa_pairs = generate_qa_pairs(str(info_path), view_index)
            if len(qa_pairs) == 0:
                print(f"No QA pairs for {info_path.name}, view {view_index}")
            else:
                print(f"{len(qa_pairs)} pairs from {info_path.name}, view {view_index}")
            all_qa_pairs.extend(qa_pairs)

    with open(output_file, "w") as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"Saved {len(all_qa_pairs)} QA pairs to {output_file}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    #fire.Fire({"check": check_qa_pairs})
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate, 
    })


if __name__ == "__main__":
    main()