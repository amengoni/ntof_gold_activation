#!/usr/bin/env python3

import sys
import re
import numpy as np


def fix_fortran_exponent(val):
    """Converts Fortran format numbers (e.g., '1.2345-6' or '1.2345+6') to standard float strings ('1.2345e-6')."""
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    val = val.strip()
    # Replace cases like '1.2345-6' or '1.2345+6' with '1.2345e-6' / '1.2345e+6'
    # avoiding messing up scientific notation that already has 'e' or 'E'
    fixed = re.sub(r'(?<=[0-9])([+-][0-9]+)', r'e\1', val)
    return float(fixed)


def load_file_robust(filename):
    """Loads text data handling standard floats as well as Fortran exponent formats."""
    try:
        # First attempt fast standard loadtxt
        return np.loadtxt(filename)
    except ValueError:
        # Fallback to robust loading with Fortran exponent fixing
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        cleaned_data = []
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            # Fix Fortran formatted numbers in each line
            fixed_line = re.sub(r'(?<=[0-9])([+-][0-9]+)', r'e\1', line_str)
            cleaned_data.append([float(tok) for tok in fixed_line.split()])

        return np.array(cleaned_data)


def read_config():
    """Reads config from standard input (1-based column indexing)."""
    filename = sys.stdin.readline().strip()
    
    line2 = sys.stdin.readline().strip().split()
    num_cols = int(line2[0])
    idx_x = int(line2[1]) - 1
    idx_y = int(line2[2]) - 1
    idx_err = int(line2[3]) - 1 if num_cols > 3 and len(line2) > 3 else None
    
    line3 = sys.stdin.readline().strip().split()
    e_min_new = float(line3[0])
    e_max_new = float(line3[1])
    scale_type = int(line3[2])
    num_bins_param = float(line3[3])
    interp_type = line3[4].lower()
    iflag = line3[5].lower() if len(line3) > 5 else "others"
        
    grid_filename = None
    grid_idx_elow = None
    grid_idx_eup = None

    if num_bins_param == 0:
        grid_filename = sys.stdin.readline().strip()
        line5 = sys.stdin.readline().strip().split()
        grid_idx_elow = int(line5[1]) - 1
        grid_idx_eup  = int(line5[2]) - 1

    return {
        "filename": filename,
        "num_cols": num_cols,
        "idx_x": idx_x,
        "idx_y": idx_y,
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


def generate_new_grid(cfg, x_data):
    """Generates the target energy bin boundaries."""
    if cfg["num_bins_param"] == 0:
        grid_data = load_file_robust(cfg["grid_filename"])
        return grid_data[:, cfg["grid_idx_elow"]], grid_data[:, cfg["grid_idx_eup"]]

    pos_energies = x_data[x_data > 0]
    if len(pos_energies) == 0:
        raise ValueError("Cannot build energy grid: all energies are <= 0.")

    data_min = np.min(pos_energies)
    data_max = np.max(pos_energies)

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
        edges = np.logspace(np.log10(e_min), np.log10(e_max), max(1, n_bins) + 1)

    return edges[:-1], edges[1:]


def interpolate_values(xq, x, y, interp_type):
    """Interpolates values onto new evaluation nodes."""
    if interp_type == "log":
        mask = (x > 0.0) & (y > 0.0)
        return np.exp(np.interp(np.log(xq), np.log(x[mask]), np.log(y[mask])))
    elif interp_type in ["1/e", "1/e_flux"]:
        mask = x > 0.0
        y_weighted = y[mask] * x[mask]
        return np.interp(xq, x[mask], y_weighted) / xq
    else:
        return np.interp(xq, x, y)


def binnerize_data(x_raw, y_raw, y_err_raw, new_elow, new_eup, interp_type, iflag):
    """Binnerizes pointwise x,y data into target energy bin boundaries."""
    idx = np.argsort(x_raw)
    x = np.asarray(x_raw[idx], dtype=float)
    y = np.asarray(y_raw[idx], dtype=float)
    y_err = np.asarray(y_err_raw[idx], dtype=float) if y_err_raw is not None else None

    n_bins = len(new_elow)
    binned_y = np.zeros(n_bins)
    binned_err = np.zeros(n_bins) if y_err is not None else None

    i_starts = np.searchsorted(x, new_elow, side='left')
    i_ends = np.searchsorted(x, new_eup, side='right')

    for j in range(n_bins):
        elo = float(new_elow[j])
        eup = float(new_eup[j])
        delta_e = eup - elo

        if delta_e <= 0.0:
            continue

        i_start = max(0, i_starts[j] - 1)
        i_end = min(len(x), i_ends[j] + 1)

        sub_x = x[i_start:i_end]
        inside_mask = (sub_x > elo) & (sub_x < eup)
        eval_points = np.unique(np.concatenate(([elo], sub_x[inside_mask], [eup])))

        if len(eval_points) < 2:
            continue

        eval_y = interpolate_values(eval_points, x, y, interp_type)
        
        dx = np.diff(eval_points)
        y_mid = 0.5 * (eval_y[:-1] + eval_y[1:])
        integral = np.sum(dx * y_mid)

        if iflag == "counts":
            binned_y[j] = integral
        else:
            binned_y[j] = integral / delta_e

        if y_err is not None:
            eval_err = np.interp(eval_points, x, y_err)
            err_mid = 0.5 * (eval_err[:-1] + eval_err[1:])
            if iflag == "counts":
                binned_err[j] = np.sqrt(np.sum((dx * err_mid) ** 2))
            else:
                binned_err[j] = np.sqrt(np.sum((dx * err_mid) ** 2)) / delta_e

    return binned_y, binned_err


def main():
    cfg = read_config()
    data = load_file_robust(cfg["filename"])

    x_raw = data[:, cfg["idx_x"]]
    y_raw = data[:, cfg["idx_y"]]
    y_err_raw = data[:, cfg["idx_err"]] if cfg["idx_err"] is not None else None

    new_elow, new_eup = generate_new_grid(cfg, x_raw)

    binned_y, binned_err = binnerize_data(
        x_raw, y_raw, y_err_raw,
        new_elow, new_eup,
        cfg["interp_type"], cfg["iflag"]
    )

    out_filename = "out_binned"
    if binned_err is not None:
        output = np.column_stack((new_elow, new_eup, binned_y, binned_err))
        header_text = "Elow Eup Binned_Quantity Uncertainty"
    else:
        output = np.column_stack((new_elow, new_eup, binned_y))
        header_text = "Elow Eup Binned_Quantity"

    np.savetxt(out_filename, output, fmt="%.6e", header=header_text, comments="# ")
    print(f"# Binnerization complete. Saved output to '{out_filename}'.")


if __name__ == "__main__":
    main()
