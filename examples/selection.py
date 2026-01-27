"""
Example: run HAST selection to find halos and Lagrangian regions.

Usage:
  python examples/selection.py /path/to/selection.ini
"""

import sys

import hast


def main() -> int:
    # Expect a single argument: path to the selection config file.
    if len(sys.argv) != 2:
        print("Usage: python examples/selection.py /path/to/selection.ini")
        return 1

    config_path = sys.argv[1]

    # Run the selection workflow. This reads RAMSES outputs and clump finder files
    # specified in the config and computes candidate halos and their Lagrangian regions.
    hast.select(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
