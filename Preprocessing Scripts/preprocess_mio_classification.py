"""Preprocess the extracted dataset using paths and settings from config.py."""
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from PIL import Image, ImageOps

SIZE = config.CLASSIFICATION_IMAGE_SIZE
FLIP_PROBABILITY = config.CLASSIFICATION_FLIP_PROBABILITY              # Horizontal flipping applies to training images only.
SEED = config.PREPROCESS_SEED
TRAIN, VALIDATION = config.MIO_TRAIN_FRACTION, config.MIO_VALIDATION_FRACTION         # Remaining labeled images become the internal test set.
CLASSES = config.VEHICLE_CLASSES
MAPPING = {
    'car': 'car', 'bus': 'bus', 'work_van': 'van', 'van': 'van',
    'articulated_truck': 'others', 'art_truck': 'others', 'pickup_truck': 'others',
    'single_unit_truck': 'others', 'motorcycle': 'others', 'motorized_vehicle': 'others',
    'background': None, 'bicycle': None, 'pedestrian': None, 'non_motorized_vehicle': None,
}


def preprocess():
    training = config.CLASSIFICATION_RAW_TRAIN
    if not (training/'car').is_dir():
        raise FileNotFoundError(f'Missing extracted classification training data: {training}')
    output = config.CLASSIFICATION_PROCESSED
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier result or delete raw data.
    rng = random.Random(SEED)
    merged = {name: [] for name in CLASSES}
    excluded = 0
    for folder in sorted(p for p in training.iterdir() if p.is_dir()):
        category = MAPPING[folder.name.lower().replace('-', '_').replace(' ', '_')]
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if category is None:
            excluded += len(paths)
        else:
            merged[category].extend(paths)
    splits = {split: {} for split in ['train', 'val', 'test']}
    for category, paths in merged.items():
        rng.shuffle(paths)
        end_train = max(1, int(len(paths)*TRAIN))
        end_val = end_train + int(len(paths)*VALIDATION)
        splits['train'][category] = paths[:end_train]
        splits['val'][category] = paths[end_train:end_val]
        splits['test'][category] = paths[end_val:]
    per_class = min(len(paths) for paths in splits['train'].values())
    if per_class == 0:
        raise ValueError('The training data must contain car, bus, work_van/van and at least one others class.')
    for category in CLASSES:
        splits['train'][category] = rng.sample(splits['train'][category], per_class)
    summary = {'seed': SEED, 'size': SIZE, 'excluded_images': excluded,
               'train_images_per_class': per_class, 'splits': {},
               'split_method': 'Seeded image-level split; camera/near-duplicate independence is not guaranteed.'}
    for split, categories in splits.items():
        summary['splits'][split] = {category: len(paths) for category, paths in categories.items()}
        for category, paths in categories.items():
            destination = output/split/category
            destination.mkdir(parents=True)
            print(f'{split}/{category}: processing {len(paths):,} images...', flush=True)
            for path in paths:
                with Image.open(path) as image:
                    gray = ImageOps.pad(image.convert('L'), SIZE, method=Image.Resampling.BILINEAR, color=0)
                flipped = split == 'train' and rng.random() < FLIP_PROBABILITY
                if flipped:
                    gray = ImageOps.mirror(gray)
                gray.save(destination/f'{path.parent.name}__{path.name}.png')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    print(f'Done: {output}\nTraining images per class: {per_class:,}')


if __name__ == '__main__':
    preprocess()
