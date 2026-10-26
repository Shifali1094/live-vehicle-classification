import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'Preprocessing Scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (100, 50), (200, 100, 50)).save(path)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)

    def test_classification(self):
        source = self.home/'raw'/'train'
        for category in ['car', 'bus', 'work_van', 'motorcycle', 'bicycle']:
            for i in range(10):
                image(source/category/f'{i}.jpg')
        output = self.home/'separate'/'processed'
        with patch.multiple(config, CLASSIFICATION_RAW_TRAIN=source, CLASSIFICATION_PROCESSED=output):
            module = load_script('preprocess_mio_classification')
            module.preprocess()
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['excluded_images'], 10)
            self.assertEqual(summary['splits'], {
                split: dict.fromkeys(config.VEHICLE_CLASSES, count)
                for split, count in [('train', 8), ('val', 1), ('test', 1)]})
            with Image.open(next(output.rglob('*.png'))) as result:
                self.assertEqual(result.size, (64, 64))
                self.assertEqual(result.mode, 'L')
            with self.assertRaises(FileExistsError):
                module.preprocess()

    def test_localization(self):
        source = self.home/'raw'
        for i in range(10):
            image(source/'train'/f'{i}.jpg')
        (source/'gt_train.csv').write_text(''.join(
            f'{i},car,10,5,29,14\n{i},bicycle,0,0,1,1\n' for i in range(10)))
        output = self.home/'separate'/'processed'
        with patch.multiple(config, LOCALIZATION_DIR=source, LOCALIZATION_PROCESSED=output):
            module = load_script('preprocess_mio_localization')
            module.preprocess()
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['excluded_objects'], 10)
            for split, count in [('train', 8), ('val', 1), ('test', 1)]:
                self.assertEqual(summary[split], {'detector_images': count, 'boxes': count})
                coco = json.loads((output/f'coco_{split}.json').read_text())
                self.assertEqual(coco['annotations'][0]['bbox'], [64, 192, 128, 64])
            label = next((output/'labels').rglob('*.txt')).read_text().split()
            self.assertEqual([float(x) for x in label], [0, .2, .35, .2, .1])
            self.assertIn(output.resolve().as_posix(), (output/'data.yaml').read_text())
            self.assertEqual(len((output/'manifest.jsonl').read_text().splitlines()), 10)
            with self.assertRaises(FileExistsError):
                module.preprocess()

    def test_detrac(self):
        settings = {'DETRAC_PROCESSED': self.home/'separate'/'processed'}
        for split, names in [('train', ['MVI_1', 'MVI_2']), ('test', ['MVI_3'])]:
            images = self.home/'images'/split
            xmls = self.home/f'{split}-xml'
            xmls.mkdir()
            settings[f'DETRAC_{split.upper()}_IMAGES'] = images
            settings[f'DETRAC_{split.upper()}_XML'] = xmls
            for name in names:
                for number in [1, 2, 3]:
                    image(images/name/f'img{number:05}.jpg')
                # Frame 2 overlaps an ignored region; frame 3 has no XML label.
                (xmls/f'{name}.xml').write_text(f'''<sequence name="{name}">
                    <ignored_region><box left="61" top="31" width="10" height="10"/></ignored_region>
                    <frame num="1"><target_list><target id="1">
                    <box left="11" top="6" width="20" height="10"/>
                    <attribute vehicle_type="car"/></target></target_list></frame>
                    <frame num="2"><target_list><target id="1">
                    <box left="61" top="31" width="10" height="10"/>
                    <attribute vehicle_type="car"/></target></target_list></frame></sequence>''')
        output = settings['DETRAC_PROCESSED']
        with patch.multiple(config, **settings):
            module = load_script('preprocess_ua_detrac')
            module.preprocess()
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['sequences']['MVI_3'], 'test')
            self.assertEqual(set(summary['sequences'].values()), {'train', 'val', 'test'})
            self.assertEqual(summary['tracking_frames'], 9)
            self.assertEqual(summary['tracking_rows'], 6)
            self.assertEqual(summary['unlabeled_frames'], 3)
            self.assertEqual(summary['detector_frames_skipped'], 3)
            for split in ['train', 'val', 'test']:
                self.assertEqual(summary[split], {'detector_images': 1, 'boxes': 1})
                gt = next((output/'tracking'/split).rglob('gt.txt'))
                self.assertEqual(len(gt.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError):
                module.preprocess()

    def test_missing_inputs_do_not_create_outputs(self):
        for name, settings in [
            ('preprocess_mio_classification', {'CLASSIFICATION_RAW_TRAIN': self.home/'missing',
                                             'CLASSIFICATION_PROCESSED': self.home/'output'}),
            ('preprocess_mio_localization', {'LOCALIZATION_DIR': self.home/'missing',
                                           'LOCALIZATION_PROCESSED': self.home/'output'}),
            ('preprocess_ua_detrac', {'DETRAC_TRAIN_IMAGES': self.home/'missing',
                                    'DETRAC_PROCESSED': self.home/'output'}),
        ]:
            with self.subTest(script=name), patch.multiple(config, **settings):
                with self.assertRaises(FileNotFoundError):
                    load_script(name).preprocess()
                self.assertFalse((self.home/'output').exists())


if __name__ == '__main__':
    unittest.main()
