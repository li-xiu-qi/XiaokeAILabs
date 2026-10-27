#!/usr/bin/env python3
"""List dynamic operations in an OpenVINO IR model."""

from collections import Counter
from pathlib import Path
import argparse
import openvino as ov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    model = ov.Core().read_model(str(args.model))
    dynamic = []
    for op in model.get_ops():
        shapes = [op.get_output_partial_shape(i) for i in range(len(op.outputs()))]
        if any(shape.is_dynamic for shape in shapes):
            dynamic.append((op.get_type_name(), op.get_friendly_name(), [str(s) for s in shapes]))
    print("model=", args.model)
    print("dynamic_count=", len(dynamic))
    print("dynamic_types=", dict(Counter(item[0] for item in dynamic)))
    for type_name, name, shapes in dynamic[: args.limit]:
        print(type_name, name, shapes)


if __name__ == "__main__":
    main()
