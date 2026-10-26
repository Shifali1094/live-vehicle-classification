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

# UA-DETRAC: point these at the extracted image and XML folders.
DETRAC_DIR = DATA_DIR / "UA-DETRAC"
DETRAC_TRAIN_IMAGES = DETRAC_DIR / "DETRAC-Images" / "Insight-MVT_Annotation_Train"
DETRAC_TEST_IMAGES = DETRAC_DIR / "DETRAC-Images" / "Insight-MVT_Annotation_Test"
DETRAC_TRAIN_XML = DETRAC_DIR / "DETRAC-Train-Annotations-XML"
DETRAC_TEST_XML = DETRAC_DIR / "DETRAC-Test-Annotations-XML"
DETRAC_PROCESSED = DETRAC_DIR / "processed_ua_detrac"

# Preprocessing settings shared by the scripts.
PREPROCESS_SEED = 42
VEHICLE_CLASSES = ['car', 'bus', 'van', 'others']
CLASSIFICATION_IMAGE_SIZE = (64, 64)
CLASSIFICATION_FLIP_PROBABILITY = 0.5
DETECTION_IMAGE_SIZE = 640
MIO_TRAIN_FRACTION = 0.8
MIO_VALIDATION_FRACTION = 0.1
DETRAC_VALIDATION_FRACTION = 0.2
