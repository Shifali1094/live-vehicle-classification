# Using the preprocessing scripts

Put the appropriate script beside the downloaded data and run it with Python in a powershell window

```powershell
py -3.13 preprocess_mio_classification.py
py -3.13 preprocess_mio_localization.py
py -3.13 preprocess_ua_detrac.py
```

## Classification

Place `preprocess_mio_classification.py` beside `MIO-TCD-Classification.tar`, or the extracted folder containing `train/<class>/`.

## Localization

Place `preprocess_mio_localization.py` beside the full `MIO-TCD-Localization.tar`, or the extracted folder containing `train/`, `test/` and `gt_train.csv`.

## UA-DETRAC

Place `preprocess_ua_detrac.py` beside the `DETRAC-Images` folder, `DETRAC-Train-Annotations-XML.zip` and `DETRAC-Test-Annotations-XML.zip` (extracted). Use XML files, not MAT.

