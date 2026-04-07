"""
Example: run HAST merge_regions to compute the convex hull enclosing two
Lagrangian regions and optionally compare their halo dynamics.

Usage:
  python examples/merge_regions.py /path/to/merge_regions.conf

Required config keys:
  region_a  — hull point file from decontaminate (_part / _part_wide) or select
  region_b  — second hull point file
  fname     — base name for outputs

Outputs:
  {fname}_part_merged   — convex hull vertices enclosing both regions (MUSIC-ready)
  {fname}_merged.pdf    — diagnostic PDF

PDF pages:
  Page 1  — three hull projections (xy / xz / yz) for A, B, and merged
  Page 2  — rotation curves side-by-side, colored by redshift
             (requires cache_a, cache_b)
  Page 3  — q and lambda vs lookback time, both halos overplotted
             (requires cache_a, cache_b, output_dir)
"""

import sys

import hast


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python examples/merge_regions.py /path/to/merge_regions.conf")
        return 1

    config_path = sys.argv[1]

    # Merge two Lagrangian region hull files into a single enclosing convex hull.
    hast.merge_regions(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
