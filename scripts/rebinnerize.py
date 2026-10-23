#!/usr/bin/env python3

import sys
import numpy as np

USAGE_TEXT = """
================================================================================
 Binned Energy Data Rebinnerizer - rebinnerize.py
================================================================================

Description:
  Rebins pre-binned energy data (with low/up boundaries) onto a new target 
  energy grid. Supports linear or logarithmic decade binning, custom external 
  energy meshes, and conservation of either integrated counts or average quantities.

Usage:
  python3 rebinnerize.py < config_file    (Piped STDIN mode)
  python3 rebinnerize.py -h               (Display this help message)

Interactive / STDIN Input Format (Line-by-Line):
--------------------------------------------------------------------------------
Line 1 : <filename>
         Path to the input text data file (e.g., transout_eV).

Line 2 : <num_cols> <idx_elow> <idx_eup> <idx_val> [idx_err]
         1-based column configuration:
           num_cols : Total number of columns in input file
           idx_elow : Column number for E_low
           idx_eup  : Column number for E_up
           idx_val  : Column number for Quantity
           idx_err  : (Optional) Column number for Uncertainty

Line 3 : <e_min_new> <e_max_new> <scale_type> <num_bins_param> <interp_type> <iflag>
         Rebinning parameters:
           e_min_new      : Lower energy boundary limit [eV] (0.0 = auto data min)
           e_max_new      : Upper energy boundary limit [eV] (0.0 = auto data max)
           scale_type     : Energy scale spacing (0 = Linear, 1 = Logarithmic)
           num_bins_param : Bins Per Decade (if scale_type=1) or Total Bins (if scale_type=0)
                            * Set to 0 to supply explicit grid boundaries from an external file
           interp_type    : Interpolation style ('linear' / 'spline')
           iflag          : Quantity normalization type:
                              'others' -> Conservation of bin average value
                              'counts' -> Conservation of total yield/counts

Line 4 : <grid_filename>  (ONLY REQUIRED IF num_bins_param == 0)
         Path to target custom energy grid file.

Line 5 : <grid_num_cols> <grid_idx_elow> <grid_idx_eup>  (ONLY REQUIRED IF num_bins_param == 0)
         1-based column indices for E_low and E_up boundaries in custom grid file.
================================================================================
"""


def prompt_read(prompt_msg):
    """Reads input with an interactive prompt if STDIN is a terminal."""
    if sys.stdin.isatty():
        sys.stdout.write(prompt_msg)
        sys.stdout.flush()
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return line.strip()


def read_config():
    """Prints descriptions/prompts and reads configuration parameters line-by-line."""
    
    # Line 1
    filename = prompt_read("# Line 1 required: [Data Filename]\n#   -> Enter data path (e.g., transout_eV): ")
    
    # Line 2
    line2_str = prompt_read("#\n# Line 2 required: [NumCols] [IdxElow] [IdxEup] [IdxVal] [IdxErr (optional)]\n#   -> Enter total columns and 1-based indices (e.g., '4 1 2 3 4' or '3 1 2 3'): ")
    line2 = line2_str.split()
    num_cols = int(line2[0])
    idx_elow = int(line2[1]) - 1
    idx_eup  = int(line2[2]) - 1
    idx_val  = int(line2[3]) - 1
    idx_err  = int(line2[4]) - 1 if len(line2) > 4 else None
    
    # Line 3
    line3_str = prompt_read("#\n# Line 3 required: [Elow_new] [Eup_new] [ScaleType (0-lin, 1-log)] [BinsPerDecade/Bin] [InterpType] [Flag ('counts'/'others')]\n#   -> Enter energy limits, scale, density, interpolation, and flag (e.g., 0 0 1 20 linear others): ")
    line3 = line3_str.split()
    e_min_new = float(line3[0])
    e_max_new = float(line3[1])
    scale_type = int(line3[2])
    num_bins_param = float(line3[3])
    interp_type = line3[4].lower()
    iflag = line3[5].lower()
        
    grid_filename = None
    grid_idx_elow = None
    grid_idx_eup = None

    # Optional Lines 4 and 5 (only if num_bins_param == 0)
    if num_bins_param == 0:
        grid_filename = prompt_read("#\n# Line 4 required (Custom Grid): [Custom Grid Filename]\n#   -> Enter custom grid filename: ")

        line5_str = prompt_read("#\n# Line 5 required (Custom Grid): [NumCols] [IdxElow] [IdxEup]\n#   -> Enter total columns and 1-based indices for custom grid (e.g., 2 1 2): ")
        line5 = line5_str.split()
        grid_idx_elow = int(line5[1]) - 1
        grid_idx_eup  = int(line5[2]) - 1

    if sys.stdin.isatty():
        print("#\n# --------------------------------------------------\n#")

    return {
        "filename": filename,
        "num_cols": num_cols,
        "idx_elow": idx_elow,
        "idx_eup": idx_eup,
        "idx_val": idx_val,
        "idx_err": idx_err,
        "e_min_new": e_min_new,
        "e_max_new": e_max_new,
        "scale_type": scale_type,
        "num_bins_param": num_bins_param,
        "interp_type": interp_type,
        "iflag": iflag,
        "grid_filename": grid_filename,
        "grid_idx_elow": grid_idx_elow,
        "grid_idx_eup": grid_idx_eup
    }


def generate_new_grid(cfg, old_elow, old_eup):
    if cfg["num_bins_param"] == 0:
        grid_data = np.loadtxt(cfg["grid_filename"])
        return grid_data[:, cfg["grid_idx_elow"]], grid_data[:, cfg["grid_idx_eup"]]

    pos_energies = old_elow[old_elow > 0]
    if len(pos_energies) == 0:
        raise ValueError("Cannot build logarithmic energy grid: all energies are <= 0.")

    data_min = np.min(pos_energies)
    data_max = np.max(old_eup)

    # Clamp e_min and e_max strictly within input file energy limits
    e_min = cfg["e_min_new"]
    e_max = cfg["e_max_new"]

    if e_min <= 0.0 or e_min < data_min:
        e_min = data_min
    if e_max <= 0.0 or e_max > data_max:
        e_max = data_max

    if e_min >= e_max:
        e_min, e_max = data_min, data_max

    if cfg["scale_type"] == 0:  # Linear scale
        edges = np.linspace(e_min, e_max, int(cfg["num_bins_param"]) + 1)
    else:  # Log scale (BinsPerDecade)
        decades = np.log10(e_max / e_min)
        n_bins = int(np.round(decades * cfg["num_bins_param"]))
        edges = np.logspace(np.log10(e_min), np.log10(e_max), n_bins + 1)

    return edges[:-1], edges[1:]


def rebin_data(old_elow, old_eup, old_val, old_err, new_elow, new_eup, iflag):
    n_new = len(new_elow)
    new_val = np.zeros(n_new)
    new_err = np.zeros(n_new) if old_err is not None else None

    i_starts = np.searchsorted(old_eup, new_elow, side='right')
    i_ends = np.searchsorted(old_elow, new_eup, side='left')

    for j in range(n_new):
        i_start = i_starts[j]
        i_end = i_ends[j]

        if i_start >= i_end:
            continue

        n_low = new_elow[j]
        n_up = new_eup[j]
        n_width = n_up - n_low

        if n_width <= 0:
            continue

        o_low = old_elow[i_start:i_end]
        o_up = old_eup[i_start:i_end]
        o_val = old_val[i_start:i_end]
        o_width = o_up - o_low

        overlap_low = np.maximum(o_low, n_low)
        overlap_up = np.minimum(o_up, n_up)
        overlap_width = np.maximum(0.0, overlap_up - overlap_low)

        if iflag == "counts":
            fraction = np.divide(overlap_width, o_width, out=np.zeros_like(overlap_width), where=o_width > 0)
            new_val[j] = np.sum(fraction * o_val)
            if old_err is not None:
                new_err[j] = np.sqrt(np.sum((fraction * old_err[i_start:i_end]) ** 2))
        else:
            weight = overlap_width / n_width
            new_val[j] = np.sum(weight * o_val)
            if old_err is not None:
                new_err[j] = np.sqrt(np.sum((weight * old_err[i_start:i_end]) ** 2))

    return new_val, new_err


def main():
    # Print usage and exit if explicitly requested (-h/--help) or if executed interactively without piped input
    if (len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]) or (len(sys.argv) == 1 and sys.stdin.isatty()):
        print(USAGE_TEXT)
        sys.exit(0)

    cfg = read_config()
    data = np.loadtxt(cfg["filename"])

    old_elow = data[:, cfg["idx_elow"]]
    old_eup  = data[:, cfg["idx_eup"]]
    old_val  = data[:, cfg["idx_val"]]
    old_err  = data[:, cfg["idx_err"]] if cfg["idx_err"] is not None else None

    new_elow, new_eup = generate_new_grid(cfg, old_elow, old_eup)

    new_val, new_err = rebin_data(
        old_elow, old_eup, old_val, old_err,
        new_elow, new_eup, cfg["iflag"]
    )

    out_filename = "out_rebinned"
    output = np.column_stack((new_elow, new_eup, new_val, new_err)) if new_err is not None else np.column_stack((new_elow, new_eup, new_val))
    header_text = "Elow Eup Quantity Uncertainty" if new_err is not None else "Elow Eup Quantity"

    np.savetxt(out_filename, output, fmt="%.6e", header=header_text, comments="# ")
    print(f"# Rebinning complete. Saved output to '{out_filename}'.")


if __name__ == "__main__":
    main()
