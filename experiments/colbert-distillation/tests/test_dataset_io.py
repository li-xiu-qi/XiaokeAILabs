import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

script = Path(__file__).resolve().parents[1] / 'dataset_io.py'
spec = importlib.util.spec_from_file_location('dataset_io', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DatasetIOTests(unittest.TestCase):
    def test_unique_ids_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'records.jsonl'
            def write(rows):
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
            write([{'_id': '1'}, {'_id': '2'}])
            self.assertEqual([row['_id'] for row in module.read_records(path)], ['1', '2'])
            for rows in ([{'_id': '1'}, {'_id': '1'}], [{'_id': ''}], [{'_id': '  '}], [{'_id': 1}], [{}]):
                write(rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    module.read_records(path)


if __name__ == '__main__':
    unittest.main()
