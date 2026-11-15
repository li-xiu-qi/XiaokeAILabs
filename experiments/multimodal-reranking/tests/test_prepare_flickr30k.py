"""Exercise the data contract without accessing a dataset or model."""
import ast
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_flickr30k.py'
spec = importlib.util.spec_from_file_location('prepare_flickr30k', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(split=split, img_id=i,
                          caption=[f'caption {i} {j}' for j in range(5)],
                          image=Image.new('RGB', (2, 2), (i, 0, 0)))
                     for i, split in enumerate(['train', 'val', 'test'] * 2)]

    def test_sampling_and_duplicate_guard(self):
        counts = {'train': 2, 'val': 1, 'test': 1}
        chosen = exporter.choose_indices(self.rows, counts, 42)
        self.assertEqual(chosen, exporter.choose_indices(self.rows, counts, 42))
        self.assertEqual({s: sum(self.rows[i]['split'] == s for i in chosen)
                          for s in counts}, counts)
        with self.assertRaises(ValueError):
            exporter.choose_indices(self.rows + [self.rows[0]], counts, 42)
        with self.assertRaises(ValueError):
            exporter.choose_indices(self.rows, {'train': 99}, 42)

    def test_export_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'export'
            exporter.export_rows(self.rows, [0, 1, 2], out)
            with (out / 'annotations.csv').open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(len(ast.literal_eval(row['raw'])), 5)
                self.assertTrue((out / 'images' / row['filename']).is_file())
            with self.assertRaises(FileExistsError):
                exporter.export_rows(self.rows, [0], out)


if __name__ == '__main__':
    unittest.main()
