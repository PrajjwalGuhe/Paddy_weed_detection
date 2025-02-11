import json
from collections import Counter

# Load COCO JSON file
with open("/home/prajjwal/Paddy_weed_detection/paddy_weed-1/train/_annotations_fixed.coco.json", "r") as f:
    dataset = json.load(f)

# Extract category IDs
category_counts = Counter([ann["category_id"] for ann in dataset["annotations"]])

print("Class Distribution in Dataset:", category_counts)
