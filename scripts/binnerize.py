#!/usr/bin/env python3

import sys
import numpy as np


def read_config():

    filename = sys.stdin.readline().strip()

    cols_info = sys.stdin.readline().strip().split()

    num_cols = int(cols_info[0])
    idx_x = int(cols_info[1]) - 1
    idx_y = int(cols_info[2]) - 1

    grid_info = sys.stdin.readline().strip().split()

    e_min_new = float(grid_info[0])
    e_max_new = float(grid_info[1])
    num_bins_param = float(grid_info[2])
    interp_type = grid_info[3].lower()

    if interp_type not in ["linear", "log"]:
        raise ValueError(
            f"Unsupported interpolation type: {interp_type}. "
            "Use 'linear' or 'log'."
        )

    grid_filename = None
    grid_idx_elow = None
    grid_idx_eup = None

    if num_bins_param == 0:

        grid_filename = sys.stdin.readline().strip()

        grid_cols_info = sys.stdin.readline().strip().split()

        grid_idx_elow = int(grid_cols_info[1]) - 1
        grid_idx_eup = int(grid_cols_info[2]) - 1

    return {
        "filename": filename,
        "idx_x": idx_x,
        "idx_y": idx_y,
        "e_min_new": e_min_new,
        "e_max_new": e_max_new,
        "num_bins_param": num_bins_param,
        "interp_type": interp_type,
        "grid_filename": grid_filename,
        "grid_idx_elow": grid_idx_elow,
        "grid_idx_eup": grid_idx_eup,
    }

def generate_new_grid(cfg, x_data):

    if cfg["num_bins_param"] == 0:

        grid_data = np.loadtxt(cfg["grid_filename"])

        new_elow = grid_data[:, cfg["grid_idx_elow"]]
        new_eup  = grid_data[:, cfg["grid_idx_eup"]]

        return new_elow, new_eup

    e_min = cfg["e_min_new"]
    e_max = cfg["e_max_new"]

    if e_min <= 0.0 or e_max <= 0.0 or e_min >= e_max:

        xpos = x_data[x_data > 0.0]

        if len(xpos) == 0:
            raise ValueError(
                "Cannot build logarithmic energy grid: "
                "all energies are <= 0."
            )

        e_min = np.min(xpos)
        e_max = np.max(xpos)

    decades = np.log10(e_max / e_min)

    n_bins = int(np.round(
        decades * cfg["num_bins_param"]
    ))

    n_bins = max(1, n_bins)

    edges = np.logspace(
        np.log10(e_min),
        np.log10(e_max),
        n_bins + 1
    )

    return edges[:-1], edges[1:]

def interpolate_values(xq, x, y, interp_type):

    if interp_type == "log":

        return np.interp(
            np.log(xq),
            np.log(x),
            y
        )

    return np.interp(
        xq,
        x,
        y
    )


def binnerize_data(
    x_raw,
    y_raw,
    new_elow,
    new_eup,
    interp_type
):

    idx = np.argsort(x_raw)

    x = np.asarray(x_raw[idx], dtype=float)
    y = np.asarray(y_raw[idx], dtype=float)

    mask = x > 0.0

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        raise ValueError(
            "Need at least two positive energy points."
        )

    n_bins = len(new_elow)

    binned_y = np.zeros(n_bins)

    for i in range(n_bins):

        elo = float(new_elow[i])
        eup = float(new_eup[i])

        if eup <= elo:
            continue

        inside = (x > elo) & (x < eup)

        points = np.concatenate(
            (
                [elo],
                x[inside],
                [eup]
            )
        )

        points = np.unique(points)

        if len(points) < 2:
            continue

        values = interpolate_values(
            points,
            x,
            y,
            interp_type
        )

        integral = np.trapezoid(
            values,
            points
        )

        binned_y[i] = integral / (eup - elo)

    return binned_y


def main():

    cfg = read_config()

    data = np.loadtxt(cfg["filename"])

    x_raw = data[:, cfg["idx_x"]]
    y_raw = data[:, cfg["idx_y"]]

    new_elow, new_eup = generate_new_grid(
        cfg,
        x_raw
    )

    binned_y = binnerize_data(
        x_raw,
        y_raw,
        new_elow,
        new_eup,
        cfg["interp_type"]
    )

    output = np.column_stack(
        (
            new_elow,
            new_eup,
            binned_y
        )
    )

    np.savetxt(
        "out_binned",
        output,
        fmt="%.6e",
        header="Elow Eup Binned_Quantity",
        comments="# "
    )

    print(
        "# Binnerization complete. "
        "Saved output to 'out_binned'."
    )


if __name__ == "__main__":
    main()
