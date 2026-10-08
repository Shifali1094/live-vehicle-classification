# Using the preprocessing scripts

Put the appropriate script beside the downloaded data and run it with Python in a powershell window

```powershell
py -3.13 preprocess_mio_classification.py
py -3.13 preprocess_mio_localization.py
py -3.13 preprocess_ua_detrac.py
```

## Classification

### Instructions

Place `preprocess_mio_classification.py` beside `MIO-TCD-Classification.tar`, or the extracted folder containing `train/<class>/`.

### Output

- train, test, validation folders - each containing 64x64 greyscale pngs, training classes have equal counts.

## Localization

### Instructions

Place `preprocess_mio_localization.py` beside the full `MIO-TCD-Localization.tar`, or the extracted folder containing `train/`, `test/` and `gt_train.csv`.

### Output

- images folder, train/test/validation, 640x640 color images.
- labels folder, containing classes and bounding boxes for YOLO.
- data.yaml, dataset paths/class names, give to YOLO when training.
- coco_train.json, coco_val.json, coco_test.json, image classes and bounding boxes for R-CNN

## UA-DETRAC

### Instructions

Place `preprocess_ua_detrac.py` beside the `DETRAC-Images` folder, `DETRAC-Train-Annotations-XML.zip` and `DETRAC-Test-Annotations-XML.zip` (extracted). Use XML files, not MAT.

### Output

- images, labels, data.yaml, coco_*.json, same as localization dataset
- tracking, train/test/validation folders each containing:
    - img1/, ordered 640×640 frames
    - gt/gt.txt, contains [frame #, vehicle id, bounding box left edge, bounding box top edge, box width, box height, ignore, vehicle class, ignore]
