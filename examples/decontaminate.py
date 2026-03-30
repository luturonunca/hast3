"""
Example: run HAST decontaminate to identify and trace contaminating particles.

Usage:
  python examples/decontaminate.py /path/to/decontaminate.conf

Example config file (decontaminate.conf):
  [decontamination]
  output_dir      = /path/to/simulation         ; directory containing output_????? folders
  fname           = /path/to/output/myrun       ; base name for output files
  output_zinit    = output_00001                ; label of the initial (high-z) output
  output_zlast    = output_00100                ; label of the final (low-z) output
  rbuffer         = 1.0                         ; buffer radius around zoom region
  ; --- optional ---
  rexclude        = 10.0    ; exclude coarse particles beyond this radius [kpc] at z_init (default 10)
  rvir            = 1.0     ; traceback sphere radius in units of R200 (default 1.0)
  rvir_track      = 0.25    ; particle-matching sphere radius in units of R200 (default 0.25)
  rvir_search     = 5.0     ; halo search radius in units of R200 (default 5.0)
  aexp_min        = 0.0     ; stop tracking below this expansion factor (default 0.0)
  halo_num        = 1       ; rank of target halo by cell count (default 1 = most massive)
  halo_coords     = -1,-1,-1 ; target halo coords [kpc]; overrides halo_num if positive (default disabled)
  halo_cutoff     = 1.0e3   ; minimum halo mass [Msol] to consider as companion (default 1e3)
  halo_massfrac   = 0.10    ; minimum companion mass as fraction of tracked halo (default 0.10)
  rank_function   = mass    ; halo ranking criterion (default mass)
  plot            = true    ; produce diagnostic PDF plots (default true)
  merger_tree     = true    ; build/cache merger tree page in PDF (default true; set false to skip expensive build)
  tree_nleaves    = 100     ; KDTree leaf size (default 100)
  point_shift     = 0,0,0   ; MUSIC integer point shift (default 0,0,0)
  point_shift_lmin = 1      ; level at which point_shift is defined (default 1)
"""

import sys

import hast


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python examples/decontaminate.py /path/to/decontaminate.conf")
        return 1

    config_path = sys.argv[1]

    # Run the decontamination workflow. Tracks the target halo backwards through
    # all snapshots, identifies coarse particles contaminating the zoom region,
    # and writes the initial-conditions particle list to {fname}_part plus a
    # halo track to {fname}_track.
    hast.decontaminate(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
