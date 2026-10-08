"""Four-class MIO preprocessing. Put beside the dataset and run with Python."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import venv

try:
    from PIL import Image, ImageOps
except ImportError:
    environment = Path(__file__).resolve().parent / '.vehicle-preprocess-env'
    python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(python), '-m', 'pip', 'install', 'Pillow==12.1.1'], check=True)
    sys.exit(subprocess.call([str(python), str(Path(__file__).resolve())]))

SIZE = (64, 64)
FLIP_PROBABILITY = 0.5              # Horizontal flipping applies to training images only.
SEED = 42
TRAIN, VALIDATION = 0.8, 0.1         # Remaining labeled images become the internal test set.
CLASSES = ['car', 'bus', 'van', 'others']
MAPPING = {
    'car': 'car', 'bus': 'bus', 'work_van': 'van', 'van': 'van',
    'articulated_truck': 'others', 'art_truck': 'others', 'pickup_truck': 'others',
    'single_unit_truck': 'others', 'motorcycle': 'others', 'motorized_vehicle': 'others',
    'background': None, 'bicycle': None, 'pedestrian': None, 'non_motorized_vehicle': None,
}


def find_training(home):
    candidates = [home/'MIO-TCD-Classification/train', home/'train', home,
                  home/'_mio_extracted/train', home/'_mio_extracted/MIO-TCD-Classification/train']
    for folder in candidates:
        if (folder/'car').is_dir():
            return folder
    archive = home/'MIO-TCD-Classification.tar'
    extracted = home/'_mio_extracted'
    extracted.mkdir(exist_ok=False)
    print('Extracting MIO-TCD-Classification.tar...', flush=True)
    with tarfile.open(archive) as data:
        data.extractall(extracted, filter='data')
    return extracted/'train' if (extracted/'train').is_dir() else extracted/'MIO-TCD-Classification/train'


def preprocess(home):
    output = home/'processed_mio_classification'
    output.mkdir(exist_ok=False)     # Never overwrite an earlier result or delete raw data.
    training = find_training(home)
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
    preprocess(Path(__file__).resolve().parent)
