
import os
from pathlib import Path
import sys
from ultralytics import YOLO
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

data_path = str(config.YOLO_DATA_YAML)
runs_dir = str(config.RUNS_DIR)

# Get the folder containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory
project_root = os.path.dirname(script_dir)

data_path = os.path.join(
    project_root,
    "processed_mio_localization",
    "data.yaml"
)

# Folder where all training runs are saved
runs_dir = os.path.join(
    script_dir,
    "runs",
    "detect"
)

# Check that the dataset configuration exists
if not os.path.isfile(data_path):
    raise FileNotFoundError(
        f"Dataset configuration not found: {data_path}"
    )

# Find all saved last.pt checkpoints
checkpoints = []

if os.path.isdir(runs_dir):
    for folder in os.listdir(runs_dir):
        checkpoint_path = os.path.join(
            runs_dir,
            folder,
            "weights",
            "last.pt"
        )

        if os.path.isfile(checkpoint_path):
            checkpoints.append(checkpoint_path)

# Stop if no checkpoint is available
if not checkpoints:
    raise FileNotFoundError(
        f"No last.pt checkpoints found in {runs_dir}"
    )

# Select the most recently modified checkpoint
checkpoint_path = max(
    checkpoints,
    key=os.path.getmtime
)

print("Loading checkpoint:", checkpoint_path)

# Load the previously trained model
model = YOLO(checkpoint_path)

# Continue training from the saved model weights
results = model.train(
    data=data_path,
    imgsz=416,
    batch=8,
    epochs=3,
    patience=3,
    workers=2,
    seed=42,
    device="mps",
    project=runs_dir,
    name="yolov8_continuation",
    exist_ok=False
)

print("Training completed.")
print("New results saved in:", results.save_dir)
