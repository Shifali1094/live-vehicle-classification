"""Preprocess the extracted dataset using paths and settings from config.py."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

from PIL import Image

SIZE, SEED = config.DETECTION_IMAGE_SIZE, config.PREPROCESS_SEED
CLASSES = config.VEHICLE_CLASSES


def resize(image):
    w, h = image.size
    scale = min(SIZE/w, SIZE/h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    dx, dy = (SIZE-nw)//2, (SIZE-nh)//2
    result = Image.new('RGB', (SIZE, SIZE), (114, 114, 114))
    result.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (dx, dy))
    return result, (nw/w, nh/h, dx, dy)


def transform(box, width, height, scale):
    x1, y1, x2, y2 = box
    if not all(math.isfinite(x) for x in box) or x2 <= x1 or y2 <= y1:
        raise ValueError(f'Invalid box: {box}')
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'Box outside image: {box}')
    sx, sy, dx, dy = scale
    return [x1*sx+dx, y1*sy+dy, x2*sx+dx, y2*sy+dy]


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=95, subsampling=0)


def export_detection(image, objects, split, name, output, coco):
    # ponytail: COCO annotations stay in RAM; stream JSON if full-data memory use becomes limiting.
    relative = f'images/{split}/{name}.jpg'
    save_image(image, output/relative)
    labels = output/'labels'/split/(name+'.txt')
    labels.parent.mkdir(parents=True, exist_ok=True)
    data = coco[split]
    image_id = len(data['images'])+1
    data['images'].append(dict(id=image_id, file_name=relative, width=SIZE, height=SIZE))
    with labels.open('w') as file:
        for obj in objects:
            x1, y1, x2, y2 = obj['box']; category = obj['class_id']
            file.write(f'{category} {(x1+x2)/2/SIZE:.9f} {(y1+y2)/2/SIZE:.9f} {(x2-x1)/SIZE:.9f} {(y2-y1)/SIZE:.9f}\n')
            data['annotations'].append(dict(id=len(data['annotations'])+1, image_id=image_id,
                category_id=category+1, bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1), iscrowd=0))


def finish(output, coco, summary):
    for split, data in coco.items():
        data['categories'] = [dict(id=i+1, name=name) for i, name in enumerate(CLASSES)]
        (output/f'coco_{split}.json').write_text(json.dumps(data), encoding='utf8')
        (output/'images'/split).mkdir(parents=True, exist_ok=True)
        summary[split] = {'detector_images': len(data['images']), 'boxes': len(data['annotations'])}
    (output/'data.yaml').write_text('path: '+json.dumps(output.resolve().as_posix())+
        '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+
        ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASSES)), encoding='utf8')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    print(f'Done: {output}', flush=True)


MAPPING = {'car': 0, 'bus': 1, 'work_van': 2, 'articulated_truck': 3, 'art_truck': 3,
           'pickup_truck': 3, 'single_unit_truck': 3, 'motorcycle': 3, 'motorized_vehicle': 3,
           'bicycle': None, 'pedestrian': None, 'non_motorized_vehicle': None}


def preprocess():
    source = config.LOCALIZATION_DIR
    if not (source/'gt_train.csv').is_file() or not (source/'train').is_dir():
        raise FileNotFoundError(f'Missing extracted localization train/ or gt_train.csv: {source}')
    output = config.LOCALIZATION_PROCESSED
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Rename the existing output folder first: {output}')
    output.mkdir(parents=True, exist_ok=True)
    annotations = defaultdict(list)
    with (source/'gt_train.csv').open(newline='', encoding='utf-8-sig') as file:
        for row in csv.reader(file):
            if not row: continue
            item, raw, x1, y1, x2, y2 = row
            category = MAPPING[raw.lower().replace('-', '_').replace(' ', '_')]
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            if x2 < x1 or y2 < y1: raise ValueError(f'Reversed box for {item}')
            annotations[item].append({'class_id': category, 'raw_class': raw, 'box': [x1, y1, x2+1, y2+1]})
    paths = {p.stem: p for p in (source/'train').glob('*.jpg')}
    if not paths or set(paths)-set(annotations):
        raise ValueError('No training images found, or a kept image has no ground-truth annotation.')
    skipped = len(set(annotations)-set(paths))
    print(f'Using {len(paths):,} images; skipped {skipped:,} missing images.', flush=True)
    items = sorted(paths); random.Random(SEED).shuffle(items)
    end_train = max(1, int(config.MIO_TRAIN_FRACTION*len(items)))
    end_val = end_train + int(config.MIO_VALIDATION_FRACTION*len(items))
    assignments = [(item, 'train' if i < end_train else 'val' if i < end_val else 'test') for i, item in enumerate(items)]
    coco = {s: {'images': [], 'annotations': []} for s in ['train', 'val', 'test']}
    excluded = 0
    with (output/'manifest.jsonl').open('w', encoding='utf8') as manifest:
        for index, (item, split) in enumerate(assignments):
            with Image.open(paths[item]) as raw: image = raw.convert('RGB')
            w, h = image.size; image, scale = resize(image)
            objects = []
            for obj in annotations[item]:
                if obj['class_id'] is None:
                    excluded += 1; continue
                objects.append({**obj, 'source_box': obj['box'], 'box': transform(obj['box'], w, h, scale)})
            export_detection(image, objects, split, item, output, coco)
            manifest.write(json.dumps(dict(source=str(paths[item]), split=split, image=f'images/{split}/{item}.jpg',
                original_size=[w, h], transform=scale, objects=objects))+'\n')
            if (index+1) % 10000 == 0: print(f'{index+1:,}/{len(items):,} labeled images', flush=True)
    finish(output, coco, dict(seed=SEED, size=SIZE, excluded_objects=excluded, skipped_images=skipped,
        split_method='Seeded image-level split of labeled training images; camera independence is not guaranteed.'))


if __name__ == '__main__':
    preprocess()
