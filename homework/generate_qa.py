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


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path) as f:
        info = json.load(f)
    karts = []
    detections = info["detections"][view_index]
    names = info.get("names", {})

    scale_x = img_width / 600
    scale_y = img_height / 400

    for det in detections:
        class_id, track_id, x1, y1, x2, y2 = det
        if int(class_id) != 1:
            continue
        x1_scaled, y1_scaled = x1 * scale_x, y1 * scale_y
        x2_scaled, y2_scaled = x2 * scale_x, y2 * scale_y
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        center = ((x1_scaled + x2_scaled) / 2, (y1_scaled + y2_scaled) / 2)
        karts.append({
            "instance_id": int(track_id),
            "kart_name": names.get(str(int(track_id)), f"kart_{track_id}"),
            "center": center
        })

    # Mark center kart
    for k in karts:
        x, y = k["center"]
        k["is_center_kart"] = (abs(x - img_width / 2) < 5 and abs(y - img_height / 2) < 5)

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
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    ego_kart = next((k for k in karts if k["is_center_kart"] or k["instance_id"] == 0), None)
    if ego_kart is None:
        return []

    questions = []
    #image_file = str(Path(info_path).stem.replace("_info", f"_{view_index:02d}_im.jpg"))
    #image_file = str(Path(info_path).parent / Path(info_path).stem.replace("_info", f"_{view_index:02d}_im.jpg"))

    split_name = Path(info_path).parent.name  # "train" or "valid"
    image_file = f"{split_name}/{Path(info_path).stem.replace('_info', f'_{view_index:02d}_im.jpg')}"

    # 1. Ego car
    questions.append({"image_file": image_file, "question": "What kart is the ego car?", "answer": ego_kart["kart_name"]})

    # 2. Total kart count
    questions.append({"image_file": image_file, "question": "How many karts are there in the scenario?", "answer": str(len(karts))})

    # 3. Track
    questions.append({"image_file": image_file, "question": "What track is this?", "answer": track_name})



    # 4. Relative position questions

    # for each cart, if center

    # for k in karts:
    #     if k["instance_id"] == ego_kart["instance_id"]:
    #         continue

    #     # for 2 karts left and right check their center points
    #     if (k["center"][0] <  ego_kart["center"][0]) : left,
    #     if (k["center"][1] - ego_kart["center"][1]

    #     # this should return "left, right, front and behind, very simple it should just be a bunch of if statements"

    #     rel = position_from_center(k["center"][0] - ego_kart["center"][0],
    #                                 k["center"][1] - ego_kart["center"][1])
    #     if not rel:
    #         continue
    #     q = f"Where is {k['kart_name']} relative to the ego car?"
    #     questions.append({"image_file": image_file, "question": q, "answer": rel})

    # def position_from_center(x, y):
    #     dx = x - img_width / 2
    #     dy = img_height / 2 - y

    #     # cart is on left side of other one, doesn't matter which pos of image so no need to check dx dy

    #     pos = []
    #     if abs(dx) > 10:
    #         pos.append("left" if dx < 0 else "right")
    #     if abs(dy) > 10:
    #         pos.append("front" if dy > 0 else "behind")
    #     return " and ".join(pos) if pos else "center"

    # for k in karts:
    #     if k["instance_id"] == ego_kart["instance_id"]:
    #         continue
    #     rel = position_from_center(k["center"][0] - ego_kart["center"][0],
    #                                 k["center"][1] - ego_kart["center"][1])
    #     if not rel:
    #         continue
    #     q = f"Where is {k['kart_name']} relative to the ego car?"
    #     questions.append({"image_file": image_file, "question": q, "answer": rel})

    MARGIN = 10  # pixels; must match counting logic below

    # ---- Q4: relative position per kart (vs ego) ----
    for k in karts:
        if k["instance_id"] == ego_kart["instance_id"]:
            continue

        dx = k["center"][0] - ego_kart["center"][0]
        dy = k["center"][1] - ego_kart["center"][1]

        rel_parts = []
        if dx <= -MARGIN:
            rel_parts.append("left")
        elif dx >= MARGIN:
            rel_parts.append("right")

        # y increases downward; negative dy means 'front'
        if dy <= -MARGIN:
            rel_parts.append("front")
        elif dy >= MARGIN:
            rel_parts.append("behind")

        if not rel_parts:
            # too close to call; skip noisy label
            continue

        rel = " and ".join(rel_parts)
        q = f"Where is {k['kart_name']} relative to the ego car?"
        questions.append({"image_file": image_file, "question": q, "answer": rel})

    # 5. Counting by position: one above is relative position, this is counting the position with the center kart
    left = right = front = behind = 0
    for k in karts:
        if k["instance_id"] == ego_kart["instance_id"]:
            continue
        dx = k["center"][0] - ego_kart["center"][0]
        dy = k["center"][1] - ego_kart["center"][1]
        if dx < 0: left += 1 # try
        if dx > 0: right += 1
        if dy < 0: front += 1
        if dy > 0: behind += 1

    for direction, count in zip(["left", "right", "front", "behind"], [left, right, front, behind]):
        questions.append({
            "image_file": image_file,
            "question": f"How many karts are to the {direction} of the ego car?",
            "answer": str(count),
        })

    return questions


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