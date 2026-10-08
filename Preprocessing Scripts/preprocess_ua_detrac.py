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


import zipfile


def folders(home):
    found, pending = [], [(home, 0)]
    while pending:
        path, depth = pending.pop(); found.append(path)
        if depth < 4:
            pending.extend((p, depth+1) for p in path.iterdir() if p.is_dir() and
                not p.name.startswith(('.', 'MVI_', 'processed_')) and p.name not in ['train', 'test', 'val'])
    return found


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def xyxy(element):
    x, y, w, h = [float(element.attrib[k]) for k in ['left', 'top', 'width', 'height']]
    return [x-1, y-1, x-1+w, y-1+h]


def preprocess(home):
    output = home/'processed_ua_detrac'
    if output.exists() and any(output.iterdir()): raise FileExistsError(f'Rename the existing output folder first: {output}')
    output.mkdir(exist_ok=True)
    for path in sorted(home.glob('*.zip')):
        name = path.name.lower()
        if ('detrac' not in name and 'insight-mvt' not in name) or any(t in name for t in ['-mat', 'detection', 'toolkit']):
            continue
        target = home/'_detrac_extracted'/path.stem
        if (target/'EXTRACTION_COMPLETE').exists(): continue
        target.mkdir(parents=True, exist_ok=True)
        print(f'Extracting {path.name}...', flush=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = member.filename.replace('\\', '/')
                if ':' in name or not (target/name).resolve().is_relative_to(target.resolve()):
                    raise ValueError('Unsafe archive path: '+name)
            archive.extractall(target)
        (target/'EXTRACTION_COMPLETE').touch()
    directories = folders(home)
    directories.sort(key=lambda p: ('_detrac_extracted' not in p.parts, str(p)))
    partitions = {}
    for split in ['train', 'test']:
        matches = [p for p in directories if split in p.name.lower() and 'xml' in p.name.lower() and next(p.glob('MVI_*.xml'), None)]
        if not matches: raise ValueError(f'Missing DETRAC {split} XML annotations.')
        partitions[split] = sorted(matches[0].glob('MVI_*.xml'))
    if {p.stem for p in partitions['train']} & {p.stem for p in partitions['test']}:
        raise ValueError('A sequence appears in both official train and test annotations.')
    sequence_paths, skipped = {}, []
    for xml in partitions['train']+partitions['test']:
        matches = [p/xml.stem for p in directories if (p/xml.stem).is_dir()]
        if not matches:
            skipped.append(xml.stem); continue
        sequence_paths[xml.stem] = matches[0]
    if not sequence_paths: raise ValueError('No image sequences found for the supplied annotations.')
    partitions = {split: [p for p in files if p.stem in sequence_paths] for split, files in partitions.items()}
    print(f'Using {len(sequence_paths)} sequences; skipped {len(skipped)} missing folders.', flush=True)
    training = partitions['train'][:]; random.Random(SEED).shuffle(training)
    validation = {p.stem for p in training[:max(1, round(len(training)*.2))]} if len(training)>1 else set()
    jobs = [(p, 'val' if p.stem in validation else 'train') for p in partitions['train']]+[(p, 'test') for p in partitions['test']]
    coco = {s: {'images': [], 'annotations': []} for s in ['train', 'val', 'test']}
    summary = dict(seed=SEED, size=SIZE, split_method='Whole sequences; official test kept separate.',
                   tracking_frames=0, tracking_rows=0, unlabeled_frames=0, detector_frames_skipped=0,
                   skipped_sequences=skipped, sequences={})
    for xml, split in jobs:
        name = xml.stem; root = ET.parse(xml).getroot()
        frames = {int(f.attrib['num']): f for f in root.findall('frame')}
        if root.get('name') != name or len(frames) != len(root.findall('frame')):
            raise ValueError(f'Wrong sequence name or duplicate XML frame in {name}')
        paths = {int(p.stem[3:]): p for p in sequence_paths[name].glob('img*.jpg')}
        if not paths or set(paths) != set(range(1, max(paths)+1)) or not set(frames) <= set(paths):
            raise ValueError(f'Missing/non-contiguous images in {name}.')
        track = output/'tracking'/split/name; (track/'gt').mkdir(parents=True)
        summary['sequences'][name] = split
        with (track/'gt/gt.txt').open('w') as gt:
            for number, path in sorted(paths.items()):
                with Image.open(path) as raw: image = raw.convert('RGB')
                w, h = image.size; image, scale = resize(image)
                regions = [xyxy(b) for b in root.findall('ignored_region/box')]
                ignored = [transform(b, w, h, scale) for b in regions if overlap(b, [0, 0, w, h])]
                masks = [[math.floor(b[0]), math.floor(b[1]), math.ceil(b[2]), math.ceil(b[3])] for b in ignored]
                frame = frames.get(number); objects = []
                if frame is not None:
                    if frame.find('target_list') is None: raise ValueError(f'Missing target_list in {name}/{number}')
                    ids = set()
                    for target in frame.findall('target_list/target'):
                        tid = int(target.attrib['id'])
                        if tid <= 0 or tid in ids: raise ValueError(f'Invalid/duplicate track ID in {name}/{number}')
                        ids.add(tid)
                        category = CLASSES.index(target.find('attribute').attrib['vehicle_type'].lower())
                        source_box = xyxy(target.find('box')); box = transform(source_box, w, h, scale)
                        objects.append(dict(class_id=category, track_id=tid, box=box, source_box=source_box))
                        x1, y1, x2, y2 = box
                        gt.write(f'{number},{tid},{x1+1:.6f},{y1+1:.6f},{x2-x1:.6f},{y2-y1:.6f},1,{category+1},-1\n')
                save_image(image, track/'img1'/f'{number:06}.jpg')
                summary['tracking_frames'] += 1; summary['tracking_rows'] += len(objects)
                if frame is None:
                    summary['unlabeled_frames'] += 1; continue
                if any(overlap(o['box'], b) for o in objects for b in masks):
                    summary['detector_frames_skipped'] += 1; continue
                for b in masks: image.paste((114, 114, 114), tuple(b))
                export_detection(image, objects, split, f'{name}/img{number:05}', output, coco)
        print(f'{name}: {len(paths):,} frames -> {split}', flush=True)
    finish(output, coco, summary)


if __name__ == '__main__':
    preprocess(Path(__file__).resolve().parent)
