#!/usr/bin/env python3
"""
calculate_ms3.py — 3D Monte Carlo Script using Native Cross-Section Master Grid

Features:
- Standardized CLI parameter parsing using argparse.
- Fixed kinematic bug & weight propagation across scattering generations.
- Handles (n,g), (n,nel), (n,2n), and (n,4n) PREPRO/NJOY cross-section channels.
- Full 3D Cylindrical Foil Geometry (depth tracking + radial boundary escape).
- Numba JIT-Compiled Fast Tracking Loop.
- Adaptive Batch Convergence Driver (Covariance Error Propagation).

Usage:
    1. Run directly with a params.txt configuration file:
           $ python3 calculate_ms3.py params.txt

    2. Override specific parameter flags via CLI:
           $ python3 calculate_ms3.py --channel n4n --thickness-cm 0.0017 --diameter-cm 1.3

    3. View complete argument documentation:
           $ python3 calculate_ms3.py -h
"""

import os
import sys
import math
import argparse
import numpy as np
from numba import njit

# ==========================================
# 1. Parameter Reader & CLI Parser
# ==========================================

def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="3D Monte Carlo Multiple Scattering Correction Driver (calculate_ms3.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python3 calculate_ms3.py
  python3 calculate_ms3.py params.txt
  python3 calculate_ms3.py --channel n4n --thickness-cm 0.0017
"""
    )
    parser.add_argument(
        "params_file", nargs="?", default=None,
        help="Path to configuration text file (default: params.txt if file exists)"
    )
    parser.add_argument("--scat-file", type=str, help="Path to elastic scattering cross section file")
    parser.add_argument("--capt-file", type=str, help="Path to capture (n,g) cross section file")
    parser.add_argument("--n2n-file", type=str, help="Path to (n,2n) cross section file")
    parser.add_argument("--n4n-file", type=str, help="Path to (n,4n) cross section file")
    parser.add_argument("--channel", type=str, help="Target reaction channel (ng, n2n, n4n, 41)")
    parser.add_argument("--target-err", type=float, help="Target relative precision (e.g., 0.005 for 0.5%%)")
    parser.add_argument("--min-histories", type=int, help="Minimum MC histories per energy point")
    parser.add_argument("--max-histories", type=int, help="Maximum MC histories cutoff")
    parser.add_argument("--batch-size", type=int, help="Batch size per convergence loop")
    parser.add_argument("--diameter-cm", type=float, help="Sample diameter in cm")
    parser.add_argument("--thickness-cm", type=float, help="Sample thickness in cm")
    parser.add_argument("--density-g-cm3", type=float, help="Sample density in g/cm3")
    parser.add_argument("--molar-mass", type=float, help="Sample molar mass in g/mol")
    
    # Automatically print help if script is run without arguments and no default params.txt exists
    if len(sys.argv) == 1 and not os.path.exists("params.txt"):
        parser.print_help()
        sys.exit(0)

    return parser.parse_args()

def load_params(cli_args):
    params = {
        "scat_file": "data/Au197.nel",
        "capt_file": "data/Au197.ng",
        "n2n_file":  "data/Au197.n2n",
        "n4n_file":  "data/Au197.n4n",
        "channel": "ng",
        "target_err": 0.005,
        "min_histories": 10000,
        "max_histories": 100000,
        "batch_size": 10000,
        "random_seed": 42,
        "diameter_cm": 1.3,
        "thickness_cm": 0.0017,
        "density_g_cm3": 19.32,
        "molar_mass": 196.9665
    }

    filepath = cli_args.params_file if cli_args.params_file else "params.txt"
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.split('#')[0].strip()
                if not line or '=' not in line:
                    continue
                key, val = [item.strip() for item in line.split('=', 1)]
                if key in ["min_histories", "max_histories", "batch_size", "random_seed"]:
                    params[key] = int(val)
                elif key in ["target_err", "diameter_cm", "thickness_cm", "density_g_cm3", "molar_mass"]:
                    params[key] = float(val)
                else:
                    params[key] = val

    # Override config file values with explicitly provided CLI options
    cli_map = {
        "scat_file": cli_args.scat_file,
        "capt_file": cli_args.capt_file,
        "n2n_file": cli_args.n2n_file,
        "n4n_file": cli_args.n4n_file,
        "channel": cli_args.channel,
        "target_err": cli_args.target_err,
        "min_histories": cli_args.min_histories,
        "max_histories": cli_args.max_histories,
        "batch_size": cli_args.batch_size,
        "diameter_cm": cli_args.diameter_cm,
        "thickness_cm": cli_args.thickness_cm,
        "density_g_cm3": cli_args.density_g_cm3,
        "molar_mass": cli_args.molar_mass,
    }
    for k, v in cli_map.items():
        if v is not None:
            params[k] = v

    return params

# ==========================================
# 2. Pointwise ASCII Cross-Section Loader
# ==========================================

def load_reconstructed_xs(filepath):
    """Loads 2-column pre-reconstructed ASCII data (Col 1: Energy [eV], Col 2: XS [barns])."""
    possible_paths = [
        filepath,
        os.path.join("data", os.path.basename(filepath)),
        os.path.join("data", os.path.basename(filepath).lower()),
        os.path.join("data", os.path.basename(filepath).upper())
    ]
    
    resolved_path = None
    for p in possible_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if not resolved_path:
        raise FileNotFoundError(f"Cross-section file '{filepath}' not found.")

    data = np.loadtxt(resolved_path, comments=('#', '//', 'C', 'c'), usecols=(0, 1), unpack=True)
    energy_grid = np.ascontiguousarray(data[0], dtype=np.float64)
    sigma_vals  = np.ascontiguousarray(data[1], dtype=np.float64)
    return energy_grid, sigma_vals

def build_unified_master_grid(grids_list):
    """Combines multiple 1D energy grids into a single sorted, unique master grid."""
    merged = np.concatenate(grids_list)
    unique_grid = np.unique(merged)
    return np.ascontiguousarray(sorted(unique_grid), dtype=np.float64)

# ==========================================
# 3. NUMBA JIT FULL 3D TRACKING KERNEL
# ==========================================

@njit(fastmath=True)
def get_xs_numba(E_eV, grid_nel, vals_nel, grid_target, vals_target):
    sig_nel    = np.interp(E_eV, grid_nel, vals_nel) if (E_eV >= grid_nel[0] and E_eV <= grid_nel[-1]) else 0.0
    sig_target = np.interp(E_eV, grid_target, vals_target) if (E_eV >= grid_target[0] and E_eV <= grid_target[-1]) else 0.0
    sig_tot    = sig_nel + sig_target
    return sig_tot, sig_nel, sig_target

@njit(fastmath=True)
def sample_isotropic_direction():
    phi = 2.0 * np.pi * np.random.random()
    w = 2.0 * np.random.random() - 1.0
    sin_th = np.sqrt(max(0.0, 1.0 - w**2))
    u = sin_th * np.cos(phi)
    v = sin_th * np.sin(phi)
    return u, v, w

@njit(fastmath=True)
def run_history_batch_3d_numba(E_inc, radius, thickness, N_atoms,
                                grid_nel, vals_nel, grid_target, vals_target,
                                batch_size=10000):
    batch_prim = np.zeros(batch_size, dtype=np.float64)
    batch_mult = np.zeros(batch_size, dtype=np.float64)
    
    AWR = 194.965017
    MAX_STACK = 256

    for i in range(batch_size):
        stack = np.zeros((MAX_STACK, 9), dtype=np.float64)
        
        sig_tot_inc, sig_nel_inc, sig_target_inc = get_xs_numba(
            E_inc, grid_nel, vals_nel, grid_target, vals_target
        )
        if sig_tot_inc <= 0.0:
            continue

        macro_tot_inc = sig_tot_inc * 1e-24 * N_atoms
        
        r_entry = radius * np.sqrt(np.random.random())
        th_entry = 2.0 * np.pi * np.random.random()
        x0 = r_entry * np.cos(th_entry)
        y0 = r_entry * np.sin(th_entry)
        z0 = 0.0
        u0, v0, w0 = 0.0, 0.0, 1.0

        P_interact = 1.0 - np.exp(-macro_tot_inc * thickness)
        if P_interact <= 0.0:
            continue

        batch_prim[i] = P_interact * (sig_target_inc / sig_tot_inc)
        dist_first = -np.log(1.0 - np.random.random() * P_interact) / macro_tot_inc
        
        stack[0, 0] = x0; stack[0, 1] = y0; stack[0, 2] = z0 + dist_first
        stack[0, 3] = u0; stack[0, 4] = v0; stack[0, 5] = w0
        stack[0, 6] = E_inc
        stack[0, 7] = 0.0; stack[0, 8] = 1.0
        stack_ptr = 1

        while stack_ptr > 0:
            stack_ptr -= 1
            x = stack[stack_ptr, 0]; y = stack[stack_ptr, 1]; z = stack[stack_ptr, 2]
            u = stack[stack_ptr, 3]; v = stack[stack_ptr, 4]; w = stack[stack_ptr, 5]
            E = stack[stack_ptr, 6]
            gen = int(stack[stack_ptr, 7])
            weight = stack[stack_ptr, 8]

            sig_tot, sig_nel, sig_target = get_xs_numba(
                E, grid_nel, vals_nel, grid_target, vals_target
            )
            
            if sig_tot <= 0.0:
                continue

            if gen > 0:
                macro_tot = sig_tot * 1e-24 * N_atoms
                dist = -np.log(np.random.random()) / macro_tot
                
                x += u * dist; y += v * dist; z += w * dist
                if (z < 0.0 or z > thickness) or (x**2 + y**2 > radius**2):
                    continue

            r_sample = np.random.random() * sig_tot
            if r_sample < sig_nel:
                alpha = ((AWR - 1.0) / (AWR + 1.0))**2
                mu_cm = 2.0 * np.random.random() - 1.0
                E_scattered = E * 0.5 * ((1.0 + alpha) + (1.0 - alpha) * mu_cm)
                
                if stack_ptr < MAX_STACK - 1:
                    us, vs, ws = sample_isotropic_direction()
                    stack[stack_ptr, 0] = x; stack[stack_ptr, 1] = y; stack[stack_ptr, 2] = z
                    stack[stack_ptr, 3] = us; stack[stack_ptr, 4] = vs; stack[stack_ptr, 5] = ws
                    stack[stack_ptr, 6] = E_scattered; stack[stack_ptr, 7] = gen + 1; stack[stack_ptr, 8] = weight
                    stack_ptr += 1
            else:
                if gen > 0:
                    batch_mult[i] += weight * P_interact

    return batch_prim, batch_mult

# ==========================================
# 4. Adaptive Convergence Driver
# ==========================================

def run_simulation_adaptive_3d(E_inc, radius, thickness, N_atoms,
                               grid_nel, vals_nel, grid_target, vals_target,
                               target_rel_err=0.005, min_hist=10000, max_hist=100000, batch_size=10000):
    
    hist_primary = np.zeros(max_hist, dtype=np.float64)
    hist_multiple = np.zeros(max_hist, dtype=np.float64)
    current_histories = 0

    while current_histories < max_hist:
        next_hist = current_histories + batch_size
        b_prim, b_mult = run_history_batch_3d_numba(
            E_inc, radius, thickness, N_atoms,
            grid_nel, vals_nel, grid_target, vals_target,
            batch_size
        )
        
        hist_primary[current_histories:next_hist] = b_prim
        hist_multiple[current_histories:next_hist] = b_mult
        current_histories = next_hist

        if current_histories >= min_hist:
            X = hist_primary[:current_histories]
            Y_tot = X + hist_multiple[:current_histories]
            mean_X = np.mean(X)
            mean_Ytot = np.mean(Y_tot)
            
            if mean_X < 1e-12:
                return 0.0, 0.0, 1.0, 0.0, current_histories

            f_ms = mean_Ytot / mean_X if mean_X > 0 else 1.0
            if np.sum(hist_multiple[:current_histories]) == 0:
                return mean_X, 0.0, 1.0, 0.0, current_histories

            var_X = np.var(X, ddof=1)
            var_Ytot = np.var(Y_tot, ddof=1)
            cov_X_Ytot = np.cov(X, Y_tot)[0, 1]

            var_mean_X = var_X / current_histories
            var_mean_Ytot = var_Ytot / current_histories
            cov_mean_X_Ytot = cov_X_Ytot / current_histories

            rel_variance = (var_mean_Ytot / (mean_Ytot**2) + 
                            var_mean_X / (mean_X**2) - 
                            2.0 * cov_mean_X_Ytot / (mean_X * mean_Ytot))

            f_ms_uncertainty = f_ms * np.sqrt(np.maximum(0.0, rel_variance))
            rel_err = f_ms_uncertainty / f_ms if f_ms > 0 else 0.0

            if rel_err <= target_rel_err:
                break

    mean_X = np.mean(hist_primary[:current_histories])
    mean_Ytot = np.mean(hist_primary[:current_histories] + hist_multiple[:current_histories])
    f_ms = mean_Ytot / mean_X if mean_X > 0 else 1.0
    
    return mean_X, np.mean(hist_multiple[:current_histories]), f_ms, f_ms_uncertainty, current_histories

# ==========================================
# 5. Main Execution Script
# ==========================================

if __name__ == "__main__":
    cli_args = parse_cli_args()
    config = load_params(cli_args)
    
    target_channel = str(config["channel"]).lower().strip()
    grid_nel, vals_nel = load_reconstructed_xs(config["scat_file"])
    
    if target_channel in ["n4n", "n,4n", "41", "17"]:
        clean_channel_name = "n4n"
        target_xs_file = config.get("n4n_file", "data/Au197.n4n")
        mt_code = 17
    elif target_channel in ["n2n", "n,2n", "16"]:
        clean_channel_name = "n2n"
        target_xs_file = config["n2n_file"]
        mt_code = 16
    else:
        clean_channel_name = "ng"
        target_xs_file = config["capt_file"]
        mt_code = 102

    grid_target, vals_target = load_reconstructed_xs(target_xs_file)
    master_energy_grid = build_unified_master_grid([grid_nel, grid_target])

    density = config["density_g_cm3"]
    thickness = config["thickness_cm"]
    radius = (config["diameter_cm"] / 2.0)
    molar_mass = config["molar_mass"]
    N_atoms = (density * 6.02214076e23) / molar_mass

    np.random.seed(config["random_seed"])

    # Warmup
    _ = run_history_batch_3d_numba(
        4.906, radius, thickness, N_atoms,
        grid_nel, vals_nel, grid_target, vals_target,
        batch_size=100
    )

    output_filename = f"ms_results_{clean_channel_name}.dat"
    with open(output_filename, 'w') as f_out:
        f_out.write(f"#reaction mode   : {clean_channel_name.upper()} (MT={mt_code})\n")
        f_out.write(f"#xs file used    : {target_xs_file}\n")
        f_out.write(f"#sample density   [g/cm3]: {density:.4f}\n")
        f_out.write(f"#sample diameter  [mm]   : {radius * 20.0:.1f}\n")
        f_out.write(f"#sample thickness [mm]   : {thickness * 10.0:.1e}\n")
        f_out.write(f"#target precision        : {config['target_err']*100:.2f}%\n")
        f_out.write(f"#total grid points       : {len(master_energy_grid)}\n")
        f_out.write("#" * 105 + "\n")
        f_out.write(f"{'#Energy (eV)':<18}{'Weighted Prim':<16}{'Weighted Multi':<16}{'F_ms':<12}{'Uncertainty':<14}{'Rel Err':<10}{'Histories':<10}\n")
        f_out.write("#" * 105 + "\n")

        for i in range(len(master_energy_grid)):
            E_inc = master_energy_grid[i]

            prim, mult, f_ms, f_ms_err, histories = run_simulation_adaptive_3d(
                E_inc=E_inc,
                radius=radius,
                thickness=thickness,
                N_atoms=N_atoms,
                grid_nel=grid_nel, vals_nel=vals_nel,
                grid_target=grid_target, vals_target=vals_target,
                target_rel_err=config["target_err"],
                min_hist=config["min_histories"],
                max_hist=config["max_histories"],
                batch_size=config["batch_size"]
            )

            rel_err_pct = (f_ms_err / f_ms) * 100 if f_ms > 0 else 0.0
            line = f"{E_inc:<18.6e}{prim*1000:<16.2f}{mult*1000:<16.2f}{f_ms:<12.5f}{f_ms_err:<14.5f}{rel_err_pct:<6.2f}%   {histories:<10}\n"
            f_out.write(line)

    print(f"    [3D MC OK] Saved results ({len(master_energy_grid)} points) to '{output_filename}'")
