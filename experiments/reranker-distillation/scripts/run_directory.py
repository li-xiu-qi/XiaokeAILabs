"""Keep completed and partial experiment evidence from being overwritten."""
from pathlib import Path


def prepare_run_directory(root: Path, stage: str):
    if stage in ('teacher', 'all'):
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('teacher or all stage needs a new or empty output directory')
        root.mkdir(parents=True, exist_ok=True)
    elif stage == 'student':
        if not root.is_dir():
            raise FileNotFoundError('student stage requires an existing teacher output directory')
        teacher_files = {'manifest.json', 'teacher-status.json', 'teacher-scores.json', 'teacher-eval.json'}
        existing = {p.name for p in root.iterdir()}
        if not teacher_files <= existing:
            raise ValueError('student stage requires a complete teacher output directory')
        if existing - teacher_files:
            raise FileExistsError('student output already exists; use a fresh run directory')
    else:
        raise ValueError(f'unknown stage: {stage}')
