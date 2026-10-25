import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Reproducibility
SEED = 42

# Target labels
CLASSES = ["car", "bus", "van", "others"]

# Original MIO-TCD folder -> target class
CLASS_MAPPING = {
    "car": "car",
    "bus": "bus",
    "work_van": "van",
    "articulated_truck": "others",
    "pickup_truck": "others",
    "single_unit_truck": "others",
    "motorcycle": "others",
    "background": None,
    "bicycle": None,
    "pedestrian": None,
    "non_motorized_vehicle": None,
}

# Preprocessing and split settings
IMAGE_SIZE = (64, 64)
FLIP_PROBABILITY = 0.5
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# SVM hyperparameters to compare
SVM_C_VALUES = [0.1, 1.0, 3.0]

# Output folders
RAW_TRAIN_DIR = config.CLASSIFICATION_RAW_TRAIN
PROCESSED_DIR = config.CLASSIFICATION_PROCESSED
RESULTS_DIR = config.HOG_RESULTS_DIR

print("Target classes:", CLASSES)
print("Raw training folder exists:", config.CLASSIFICATION_RAW_TRAIN.is_dir())
print("Processed data directory:", PROCESSED_DIR)
print("Results directory:", RESULTS_DIR)

def preprocess_dataset():
    # - Maps original labels to car, bus, van, and others.
    # - Excludes background, bicycle, pedestrian, and non-motorized vehicle.
    # - Creates seeded 80/10/10 splits and balances the training classes.
    # - Resizes images to 64x64 grayscale and flips some training images.

    if not RAW_TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Training folder not found: {RAW_TRAIN_DIR}\n"
            "Extract MIO-TCD-Classification.tar so train/<class>/ exists."
        )

    if PROCESSED_DIR.exists():
        expected = [PROCESSED_DIR / "summary.json"]
        if all(path.exists() for path in expected):
            print(f"Using existing processed dataset: {PROCESSED_DIR}")
            return
        raise FileExistsError(
            f"{PROCESSED_DIR} already exists but has no summary.json. "
            "Rename or inspect it before rerunning; raw data is left untouched."
        )

    rng = random.Random(SEED)
    merged = {name: [] for name in CLASSES}
    excluded_count = 0

    for folder in sorted(path for path in RAW_TRAIN_DIR.iterdir() if path.is_dir()):
        key = folder.name.lower().replace("-", "_").replace(" ", "_")
        category = CLASS_MAPPING.get(key)
        images = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if category is None:
            excluded_count += len(images)
        else:
            merged[category].extend(images)

    splits = {"train": {}, "val": {}, "test": {}}
    for category, paths in merged.items():
        rng.shuffle(paths)
        train_end = int(len(paths) * TRAIN_RATIO)
        val_end = train_end + int(len(paths) * VALIDATION_RATIO)
        splits["train"][category] = paths[:train_end]
        splits["val"][category] = paths[train_end:val_end]
        splits["test"][category] = paths[val_end:]

    per_class = min(len(paths) for paths in splits["train"].values())
    if per_class == 0:
        raise ValueError(
            "Could not build all four classes. Check the original class-folder names."
        )

    for category in CLASSES:
        splits["train"][category] = rng.sample(
            splits["train"][category], per_class
        )

    summary = {
        "seed": SEED,
        "image_size": list(IMAGE_SIZE),
        "excluded_images": excluded_count,
        "train_images_per_class": per_class,
        "split_method": (
            "Seeded image-level split; camera/near-duplicate independence "
            "is not guaranteed."
        ),
        "splits": {},
    }

    PROCESSED_DIR.mkdir(parents=True)

    for split_name, categories in splits.items():
        summary["splits"][split_name] = {}
        for category, paths in categories.items():
            destination = PROCESSED_DIR / split_name / category
            destination.mkdir(parents=True)
            summary["splits"][split_name][category] = len(paths)

            print(f"Preprocessing {split_name}/{category}: {len(paths):,} images")
            for path in tqdm(paths, desc=f"{split_name}/{category}"):
                output_path = destination / f"{path.parent.name}__{path.name}.png"
                with Image.open(path) as image:
                    gray = ImageOps.pad(
                        image.convert("L"),
                        IMAGE_SIZE,
                        method=Image.Resampling.BILINEAR,
                        color=0,
                    )
                if split_name == "train" and rng.random() < FLIP_PROBABILITY:
                    gray = ImageOps.mirror(gray)
                gray.save(output_path)

    (PROCESSED_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Preprocessing complete: {PROCESSED_DIR}")
    print(f"Balanced training images per class: {per_class:,}")

def image_to_hog(image_path):
    # - Loads a processed grayscale image.
    # - Extracts HOG descriptors (1,764 features for 64x64 images).
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if image.shape != (64, 64):
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

def build_features(split_name):
    # - Reads each class folder and returns features, labels, and source paths.
    features, labels, paths = [], [], []
    split_dir = PROCESSED_DIR / split_name

    for category in CLASSES:
        class_dir = split_dir / category
        image_paths = sorted(class_dir.glob("*.png"))
        for image_path in tqdm(image_paths, desc=f"HOG {split_name}/{category}"):
            try:
                features.append(image_to_hog(image_path))
                labels.append(category)
                paths.append(str(image_path))
            except Exception as error:
                print(f"Skipping {image_path}: {error}")

    if not features:
        raise ValueError(
            f"No features generated for '{split_name}'. Check {split_dir}."
        )

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        paths,
    )

def train_model(X_train, y_train, X_val, y_val):
    # - Trains three Linear SVM configurations.
    # - Selects the model with the highest validation macro-F1.
    results = []
    best_model = None
    best_score = -1.0
    best_c = None

    for c_value in SVM_C_VALUES:
        print(f"\nTraining Linear SVM with C={c_value}...")
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=c_value,
                class_weight="balanced",
                random_state=SEED,
                max_iter=10000,
            ),
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        macro_f1 = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        accuracy = accuracy_score(y_val, predictions)

        result = {
            "C": c_value,
            "validation_accuracy": float(accuracy),
            "validation_macro_f1": float(macro_f1),
        }
        results.append(result)
        print(
            f"Validation accuracy={accuracy:.4f}; "
            f"macro-F1={macro_f1:.4f}"
        )

        if macro_f1 > best_score:
            best_model = model
            best_score = macro_f1
            best_c = c_value

    return best_model, best_c, results

def evaluate_and_save(model, best_c, tuning_results, X_test, y_test):
    # - Evaluates once on the held-out test split.
    # - Saves the model, metrics, classification report, and confusion matrix.

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    report = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        labels=CLASSES,
        target_names=CLASSES,
        zero_division=0,
    )

    print("\nSelected C:", best_c)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")
    print("\nClassification report:\n")
    print(report_text)

    joblib.dump(model, RESULTS_DIR / "hog_svm_model.joblib")

    metrics = {
        "classes": CLASSES,
        "selected_C": best_c,
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "validation_tuning": tuning_results,
        "classification_report": report,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    cm = confusion_matrix(y_test, predictions, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES,
    ).plot(ax=ax, xticks_rotation=35, colorbar=False, values_format="d")
    ax.set_title("HOG + Linear SVM: Four-Class Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.show()

    print(f"\nSaved outputs to: {RESULTS_DIR}")
    print("- hog_svm_model.joblib")
    print("- metrics.json")
    print("- classification_report.txt")
    print("- confusion_matrix.png")

def main():
    start = time.time()
    preprocess_dataset()

    print("\nLoading processed train split...")
    X_train, y_train, _ = build_features("train")
    print("Training feature shape:", X_train.shape)

    print("\nLoading processed validation split...")
    X_val, y_val, _ = build_features("val")
    print("Validation feature shape:", X_val.shape)

    print("\nLoading processed test split...")
    X_test, y_test, _ = build_features("test")
    print("Test feature shape:", X_test.shape)

    model, best_c, tuning_results = train_model(X_train, y_train, X_val, y_val)
    evaluate_and_save(model, best_c, tuning_results, X_test, y_test)
    print(f"\nTotal elapsed time: {(time.time() - start) / 60:.1f} minutes")

main()