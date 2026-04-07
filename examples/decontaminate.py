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
  ; --- halo search ---
  rexclude        = 30.0    ; inner radius [kpc physical] — coarse particles found here are added to the new zoom region (default 10)
  rexclude_zmax   = 6.0     ; skip rexclude checks above this redshift, before halo formation (default 6)
  rvir            = 2.0     ; traceback sphere radius in units of R200 (default 1.0)
  rvir_track      = 0.25    ; particle-matching sphere radius in units of R200 (default 0.25)
  rvir_search     = 5.0     ; halo search radius in units of R200 (default 5.0)
  aexp_min        = 0.0     ; stop tracking below this expansion factor (default 0.0)
  halo_num        = 1       ; rank of target halo by cell count (default 1 = most massive)
  halo_coords     = -1,-1,-1 ; target halo coords [kpc]; overrides halo_num if positive (default disabled)
  halo_cutoff     = 1.0e3   ; minimum halo mass [Msol] to consider as companion (default 1e3)
  halo_massfrac   = 0.10    ; minimum companion mass as fraction of tracked halo (default 0.10)
  rank_function   = mass    ; halo ranking criterion: mass / ncell / rho_max / rho_ave (default mass)
  plot            = true    ; produce diagnostic PDF plots (default true)
  tree_nleaves    = 100     ; KDTree leaf size (default 100)
  point_shift     = 0,0,0   ; MUSIC integer point shift (default 0,0,0)
  point_shift_lmin = 1      ; level at which point_shift is defined (default 1)
  use_cache       = true    ; cache tracking loop results to disk and reuse on re-runs (default true)
  ; --- seed iords mode (optional) ---
  ; Provide a text file (one iord per line) to derive the target halo from a
  ; known particle set instead of using halo_num / halo_coords.
  ; seed_iords_file = ./my_particles.ids
  ; --- piggybacking analyses (all default true) ---
  rotation_curves = true    ; collect and plot rotation curves at each snapshot (PDF page 3)
  halo_dynamics   = true    ; collect and plot q / lambda vs lookback time (PDF page 4)
  merger_tree     = true    ; build merger tree and plot it (PDF page 5)
  lagrange_iords  = true    ; record Lagrangian particle sets at integer redshifts z=0..6 (PDF pages 6-7)
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
