"""Put beside the downloaded dataset and run with Python 3.13."""
import csv
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import venv

try:
    from PIL import Image
    from defusedxml import ElementTree as ET
except ImportError:
    environment = Path(__file__).resolve().parent / '.vehicle-preprocess-env'
    python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(python), '-m', 'pip', 'install', 'Pillow==12.1.1', 'defusedxml==0.7.1'], check=True)
    sys.exit(subprocess.call([str(python), str(Path(__file__).resolve())]))

SIZE, SEED = 640, 42
CLASSES = ['car', 'bus', 'van', 'others']


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


import tarfile
from collections import defaultdict

MAPPING = {'car': 0, 'bus': 1, 'work_van': 2, 'articulated_truck': 3, 'art_truck': 3,
           'pickup_truck': 3, 'single_unit_truck': 3, 'motorcycle': 3, 'motorized_vehicle': 3,
           'bicycle': None, 'pedestrian': None, 'non_motorized_vehicle': None}


def preprocess(home):
    output = home/'processed_mio_localization'
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Rename the existing output folder first: {output}')
    output.mkdir(exist_ok=True)
    candidates = [home/'MIO-TCD-Localization', home, home/'_mio_localization_extracted/MIO-TCD-Localization',
                  home/'_mio_localization_extracted']
    source = next((p for p in candidates if (p/'gt_train.csv').is_file() and (p/'train').is_dir()), None)
    if source is None:
        extracted = home/'_mio_localization_extracted'; extracted.mkdir(exist_ok=False)
        print('Extracting MIO-TCD-Localization.tar...', flush=True)
        with tarfile.open(home/'MIO-TCD-Localization.tar') as archive:
            archive.extractall(extracted, filter='data')
        source = extracted/'MIO-TCD-Localization' if (extracted/'MIO-TCD-Localization').is_dir() else extracted
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
    end_train, end_val = max(1, int(.8*len(items))), max(1, int(.8*len(items)))+int(.1*len(items))
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
        split_method='80/10/10 image-level split of labeled training images; camera independence is not guaranteed.'))


if __name__ == '__main__':
    preprocess(Path(__file__).resolve().parent)
