# Object Detection with YOLOv8

# Load libraries
import os
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# Set the path to the prepared MIO-TCD Localization dataset
base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "processed_mio_localization"
)

# Set the dataset splits
splits = ["train", "val", "test"]

# Check the number of images and label files in each dataset split
for split in splits:
    image_dir = os.path.join(base_dir, "images", split)
    label_dir = os.path.join(base_dir, "labels", split)

    # Get the image and label files
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    label_files = [
        f for f in os.listdir(label_dir)
        if f.endswith(".txt")
    ]

    # Print the number of images and labels
    print(f"{split.upper()} SET:")
    print(f"  Images: {len(image_files)}")
    print(f"  Labels: {len(label_files)}")
    print()

# Load the class names from data.yaml
class_names_path = os.path.join(base_dir, "data.yaml")

with open(class_names_path, "r") as f:
    data_yaml = yaml.safe_load(f)

# Get the class names and number of classes
class_name_map = data_yaml["names"]

if isinstance(class_name_map, dict):
    class_names = [
        class_name_map[i] if i in class_name_map
        else class_name_map[str(i)]
        for i in range(len(class_name_map))
    ]
else:
    class_names = class_name_map

number_of_classes = len(class_names)

print("Class names:", class_names)
print("Number of classes:", number_of_classes)

# Count the number of objects belonging to each class
class_counts = defaultdict(int)

for split in splits:
    label_dir = os.path.join(base_dir, "labels", split)

    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            label_path = os.path.join(label_dir, filename)

            # Read the annotation file
            with open(label_path, "r") as f:
                for line in f:
                    if line.strip():
                        # Get the class ID from the first value
                        class_id = int(line.strip().split()[0])

                        # Count each object class
                        class_counts[class_id] += 1

# Print the class distribution
print("\nClass Distribution:")

for class_id in sorted(class_counts.keys()):
    class_name = class_names[class_id]

    print(
        f"Class {class_name} (ID: {class_id}): "
        f"{class_counts[class_id]} instances"
    )

# Prepare the class labels and counts for plotting
class_ids = sorted(class_counts.keys())
labels = [class_names[class_id] for class_id in class_ids]
counts = [class_counts[class_id] for class_id in class_ids]

# Plot the class distribution
plt.figure(figsize=(8, 4))
plt.bar(labels, counts, color="green")
plt.xlabel("Vehicle Class")
plt.ylabel("Number of Instances")
plt.title("Vehicle Class Distribution Across All Splits")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Set the path to the dataset configuration file
data_path = os.path.abspath(class_names_path)

# Load the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train YOLOv8 using Apple's GPU
results = model.train(
    data=data_path,
    imgsz=416,
    batch=8,
    epochs=5,
    patience=3,
    workers=2,
    seed=42,
    device="mps",
    name="yolov8_vehicle_detection"
)