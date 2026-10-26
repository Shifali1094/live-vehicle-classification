# Dataset Preprocessing and Model Training

This repository contains preprocessing scripts and model implementations for vehicle classification, object detection and tracking using the MIO-TCD and UA-DETRAC datasets.

## 1. Project Configuration

Dataset and output paths are managed centrally through `config.py` in the repository root. Scripts import these paths rather than relying on hardcoded local directories.

By default, datasets are expected inside the `data/` directory. To use a different location, set the `VEHICLE_DATA_DIR` environment variable.

For example, in PowerShell:

```powershell
$env:VEHICLE_DATA_DIR = "D:\Datasets\Vehicle"
```

The expected directory structure is:

```text
live-vehicle-classification/
|-- config.py
|-- data/
|   |-- MIO-TCD-Classification/
|   |   |-- train/<class>/
|   |   `-- processed_mio_classification/
|   |-- MIO-TCD-Localization/MIO-TCD-Localization/
|   |   |-- train/
|   |   |-- test/
|   |   |-- gt_train.csv
|   |   `-- processed_mio_localization/
|   `-- UA-DETRAC/
|       |-- DETRAC-Images/
|       |   |-- Insight-MVT_Annotation_Train/MVI_*/img*.jpg
|       |   `-- Insight-MVT_Annotation_Test/MVI_*/img*.jpg
|       |-- DETRAC-Train-Annotations-XML/MVI_*.xml
|       |-- DETRAC-Test-Annotations-XML/MVI_*.xml
|       `-- processed_ua_detrac/
|-- Preprocessing Scripts/
|-- Faster-RCNN/
`-- hog_svm_results/
```

The processed dataset folders are generated during preprocessing and do not need to exist beforehand.

## 2. Dataset Preprocessing

Install dependencies once in your chosen Python environment, then run the scripts from the repository root. The scripts use `config.py` for paths, seed, image sizes, class names, split fractions and classification flip probability. If your extracted folders differ from the layout above, edit the corresponding paths in `config.py`.

Extract dataset archives before running. The scripts do not search for datasets, extract archives, create environments or install packages. Existing nonempty output folders are protected; rename them before rerunning (classification requires its output folder not to exist).

```powershell
py -m pip install -r requirements.txt
py "Preprocessing Scripts/preprocess_mio_classification.py"
py "Preprocessing Scripts/preprocess_mio_localization.py"
py "Preprocessing Scripts/preprocess_ua_detrac.py"
```

### MIO-TCD Classification

**Input:** The extracted MIO-TCD Classification dataset, containing `train/<class>/` directories.

**Processing:**
- Maps the original vehicle categories to four classes: `car`, `bus`, `van` and `others`.
- Excludes non-target categories.
- Creates training, validation and test splits.
- Resizes images to 64 × 64 grayscale PNGs.
- Balances the training classes.

**Output:** `data/MIO-TCD-Classification/processed_mio_classification/`

The processed dataset contains `train/`, `val/` and `test/` folders, organised by vehicle class.

### MIO-TCD Localization

**Input:** The extracted MIO-TCD Localization dataset, containing `train/`, `test/` and `gt_train.csv`.

**Output:** `data/MIO-TCD-Localization/MIO-TCD-Localization/processed_mio_localization/`

The processed dataset contains:

- `images/`: Training, validation and test splits containing 640 × 640 colour images.
- `labels/`: YOLO-format class labels and bounding box annotations.
- `data.yaml`: Dataset configuration and class names for YOLOv8.
- `coco_train.json`, `coco_val.json`, `coco_test.json`: COCO-format annotations for Faster R-CNN.

### UA-DETRAC

**Input:** Extracted train/test image folders and extracted train/test XML annotations at the explicit `DETRAC_*` paths in `config.py`. Use XML annotations rather than MAT files.

**Output:** `data/UA-DETRAC/processed_ua_detrac/`, containing:

- `images/`, `labels/` and `data.yaml` for YOLO training.
- `coco_*.json` annotation files for Faster R-CNN.
- `tracking/` containing training, validation and test sequences.

Each tracking sequence contains:

```text
tracking/
└── <split>/
    └── <sequence>/
        ├── img1/
        └── gt/
            └── gt.txt
```

`img1/` contains ordered 640 × 640 video frames.

`gt.txt` contains tracking annotations in the format:

```text
frame, vehicle_id, left, top, width, height, ignore, class, ignore
```

## 3. Model Training

The repository contains implementations of three vehicle recognition approaches.

### YOLOv8

YOLOv8 is used for vehicle detection with the processed MIO-TCD Localization dataset.

Training uses the dataset configuration defined by `YOLO_DATA_YAML` in `config.py`.

Training outputs and model checkpoints are saved under:

```text
Preprocessing Scripts/runs/detect/
```

The `continue_yolov8.py` script can load the most recently modified `last.pt` checkpoint and start an additional training run.

### Faster R-CNN

Faster R-CNN uses the processed localization dataset and COCO-format bounding box annotations.

Dataset paths are obtained from `config.LOCALIZATION_PROCESSED`.

### HOG + SVM

A traditional machine learning baseline for vehicle classification using Histogram of Oriented Gradients (HOG) features and a linear Support Vector Machine (SVM).

The pipeline:

1. Preprocesses the MIO-TCD Classification dataset into four target classes.
2. Extracts HOG features from 64 × 64 grayscale images.
3. Trains linear SVM models using different regularisation parameters.
4. Selects the model with the highest validation macro-F1 score.
5. Evaluates the selected model on the held-out test set.

Results are saved in `hog_svm_results/`, including:

- `hog_svm_model.joblib`: Trained classifier.
- `metrics.json`: Evaluation metrics and hyperparameter tuning results.
- `classification_report.txt`: Per-class performance.
- `confusion_matrix.png`: Test-set confusion matrix.

## 4. Notes

- Dataset paths should be configured through `config.py` rather than hardcoded into individual scripts.
- Raw datasets and generated model outputs should be excluded from Git where appropriate.
- GPU availability and training device settings may differ between operating systems.

## 5. Preprocessing checks

Small synthetic datasets exercise all three pipelines without needing the real datasets:

```powershell
py -m unittest discover -s tests -v
```
