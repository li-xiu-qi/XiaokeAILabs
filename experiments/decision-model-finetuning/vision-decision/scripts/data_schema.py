"""ScienceQA model-input schema and validation for prepared rows."""
import json

INPUT_SCHEMA = "scienceqa-choices-hint-v1"


def build_state(choices, hint):
    letters = "ABCDEFGH"
    option_text = "\n".join(
        f"{letters[i] if i < len(letters) else i}. {choice}"
        for i, choice in enumerate(choices)
    )
    parts = [f"备选答案：\n{option_text}"]
    if hint:
        parts.append(f"提示：{hint}")
    return "\n\n".join(parts)


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            hint = row.get("context_hint")
            options = row.get("options")
            if (
                row.get("input_schema") != INPUT_SCHEMA
                or not isinstance(hint, str)
                or not isinstance(options, list)
                or not all(isinstance(option, str) for option in options)
                or row.get("state") != build_state(options, hint)
            ):
                raise ValueError(
                    f"{path}:{line_number}: prepared input schema mismatch; "
                    "regenerate the data with scripts/prepare_data.py"
                )
            rows.append(row)
    return rows


def require_checkpoint(directory):
    from pathlib import Path

    path = Path(directory) / "input_schema.json"
    try:
        with path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{directory}: checkpoint input provenance missing; train a fresh adapter"
        ) from exc
    if metadata.get("input_schema") != INPUT_SCHEMA:
        raise ValueError(f"{directory}: checkpoint input schema mismatch; train a fresh adapter")


def mark_checkpoint(directory):
    from pathlib import Path

    path = Path(directory) / "input_schema.json"
    path.write_text(json.dumps({"input_schema": INPUT_SCHEMA}) + "\n", encoding="utf-8")
