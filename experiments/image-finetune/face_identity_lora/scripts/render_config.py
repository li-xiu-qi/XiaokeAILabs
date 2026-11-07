#!/usr/bin/env python3
"""Replace the one training dataset path with an absolute, YAML-safe path."""
import json
import re
import sys
from pathlib import Path


def render(source: str, dataset: Path) -> str:
    if not dataset.is_dir():
        raise ValueError(f"dataset directory does not exist: {dataset}")
    pattern = r'(?m)^(\s*-\s*folder_path:\s*).*$'
    result, count = re.subn(pattern, lambda m: m.group(1) + json.dumps(str(dataset.resolve())), source)
    if count != 1:
        raise ValueError(f"expected exactly one dataset folder_path, found {count}")
    return result


if __name__ == '__main__':
    config, output, dataset = map(Path, sys.argv[1:])
    output.write_text(render(config.read_text(encoding='utf-8'), dataset), encoding='utf-8')
