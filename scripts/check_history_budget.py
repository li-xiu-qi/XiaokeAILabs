#!/usr/bin/env python3
"""Check the packed full-history size in bytes against the repository budget."""
import argparse

LIMIT_BYTES = 33 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, required=True, dest="packed_bytes")
    args = parser.parse_args()
    if args.packed_bytes < 0:
        parser.error("packed size cannot be negative")
    print(f"Packed history: {args.packed_bytes} bytes; budget: {LIMIT_BYTES} bytes (33 MiB)")
    if args.packed_bytes > LIMIT_BYTES:
        parser.exit(1, "Packed history exceeds the budget.\n")


if __name__ == "__main__":
    main()
