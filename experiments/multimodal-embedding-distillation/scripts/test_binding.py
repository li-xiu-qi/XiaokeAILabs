"""CPU checks for sample, image and teacher-vector binding."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from validate_binding import validate_images, validate_teacher_binding


class Array:
    def __init__(self, value=None, shape=None):
        self.value = value
        self.shape = shape
        self.ndim = len(shape) if shape is not None else 0

    def item(self):
        return self.value

    def __len__(self):
        return self.shape[0]


class BindingTest(unittest.TestCase):
    def test_image_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / 'image.jpg'
            image.write_bytes(b'original image')
            samples = {'train': [{'filename': 'image.jpg', 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}]}
            validate_images(root, samples)
            image.write_bytes(b'replaced image')
            with self.assertRaisesRegex(ValueError, 'image bytes changed'):
                validate_images(root, samples)

    def test_teacher_manifest_and_shape_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'samples.json'
            path.write_bytes(b'original manifest')
            samples = {'train': [{}]}
            vectors = {'samples_sha256': Array(hashlib.sha256(path.read_bytes()).hexdigest()),
                       'train_text': Array(shape=(1, 2)), 'train_image': Array(shape=(1, 2))}
            validate_teacher_binding(path, samples, vectors)
            path.write_bytes(b'reordered manifest')
            with self.assertRaisesRegex(ValueError, 'sample manifest'):
                validate_teacher_binding(path, samples, vectors)
            path.write_bytes(b'original manifest')
            vectors['train_image'] = Array(shape=(2, 2))
            with self.assertRaisesRegex(ValueError, 'vector shapes'):
                validate_teacher_binding(path, samples, vectors)


if __name__ == '__main__':
    unittest.main()
