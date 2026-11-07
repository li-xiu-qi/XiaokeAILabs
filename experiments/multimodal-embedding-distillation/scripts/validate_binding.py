"""Prevent applying teacher vectors to a different or reordered sample manifest."""
import hashlib


def validate_images(image_root, samples):
    """Ensure both stages read the same image bytes recorded during preparation."""
    for split, rows in samples.items():
        for row in rows:
            path = image_root / row['filename']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row['sha256']:
                raise ValueError(f'{split}: image bytes changed after preparation: {row["filename"]}')


def validate_teacher_binding(path, samples, vectors):
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    if 'samples_sha256' not in vectors or str(vectors['samples_sha256'].item()) != expected:
        raise ValueError('teacher vectors do not match the exact sample manifest; regenerate them')
    for split, rows in samples.items():
        text, image = vectors[split + '_text'], vectors[split + '_image']
        if text.ndim != 2 or text.shape != image.shape or len(text) != len(rows):
            raise ValueError(f'{split}: teacher vector shapes do not match the samples')
