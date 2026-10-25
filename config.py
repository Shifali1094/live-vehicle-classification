import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


DATA_DIR = Path(os.environ.get("VEHICLE_DATA_DIR", ROOT / "data"))

CLASSIFICATION_PROCESSED = DATA_DIR / "MIO-TCD-Classification" / "processed_mio_classification"
CLASSIFICATION_RAW_TRAIN = DATA_DIR / "MIO-TCD-Classification" / "train"

LOCALIZATION_PROCESSED = (
    DATA_DIR / "MIO-TCD-Localization" / "MIO-TCD-Localization" / "processed_mio_localization"
)
YOLO_DATA_YAML = LOCALIZATION_PROCESSED / "data.yaml"

RUNS_DIR = ROOT / "Preprocessing Scripts" / "runs" / "detect"
HOG_RESULTS_DIR = ROOT / "hog_svm_results"
FRCNN_DIR = ROOT / "Faster-RCNN"