"""Gate legacy benchmarks that may delete fixed collections and local paths."""

import os


def require_isolated_instance():
    if os.environ.get("XL_VECTOR_DB_ALLOW_DESTRUCTIVE") != "isolated-instance":
        raise SystemExit(
            "Legacy benchmark may delete fixed collections or files. "
            "Use a dedicated isolated database instance and data directory; "
            "then set XL_VECTOR_DB_ALLOW_DESTRUCTIVE=isolated-instance. "
            "This confirmation does not enforce isolation."
        )
