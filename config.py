import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("VEHICLE_DATA_DIR", ROOT / "data")).expanduser().resolve()

# MIO-TCD Classification
CLASSIFICATION_DIR = DATA_DIR / "MIO-TCD-Classification"
CLASSIFICATION_RAW_TRAIN = CLASSIFICATION_DIR / "train"
CLASSIFICATION_PROCESSED = CLASSIFICATION_DIR / "processed_mio_classification"

# MIO-TCD Localization
LOCALIZATION_DIR = (DATA_DIR / "MIO-TCD-Localization" / "MIO-TCD-Localization")
LOCALIZATION_PROCESSED = LOCALIZATION_DIR / "processed_mio_localization"
YOLO_DATA_YAML = LOCALIZATION_PROCESSED / "data.yaml"

# Model outputs
RUNS_DIR = ROOT / "Preprocessing Scripts" / "runs" / "detect"
HOG_RESULTS_DIR = ROOT / "hog_svm_results"
FRCNN_DIR = ROOT / "Faster-RCNN"