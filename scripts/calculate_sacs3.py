#!/usr/bin/env python3

import sys
import subprocess
import os
import argparse
import warnings
import numpy as np
from scipy.optimize import curve_fit

def resolve_helper_script(script_name):
    local_path = os.path.join(os.getcwd(), "src", script_name)
    if os.path.exists(local_path):
        return local_path
    
    home_path = os.path.expanduser(os.path.join("~/cernbox/research/src", script_name))
    if os.path.exists(home_path):
        return home_path
        
    raise FileNotFoundError(
        f"Could not find '{script_name}' in './src/' or '{os.path.dirname(home_path)}'"
    )

BINNERIZE_PATH = resolve_helper_script("binnerize.py")
REBINNERIZE_PATH = resolve_helper_script("rebinnerize.py")

def interp_trend_extrap(x_new, x_orig, y_orig, n_pts_fit=5, is_loglog=True, is_threshold_rxn=False):
    """
    Interpolates data on x_new.
    For threshold reactions (like n2n, n4n), values below min(x_orig) are set strictly to 0.0
    to prevent unphysical log-log extrapolation into thermal/epithermal energy regions.
    """
    mask_orig = (x_orig > 0.0) & (y_orig > 0.0) if is_loglog else (x_orig > 0.0)
    if not np.any(mask_orig):
        return np.interp(x_new, x_orig, y_orig)

    x_valid = x_orig[mask_orig]
    y_valid = y_orig[mask_orig]

    log_x_valid = np.log(x_valid)
    log_y_valid = np.log(y_valid) if is_loglog else y_valid
    log_x_new = np.log(np.maximum(x_new, 1e-12))

    if is_loglog:
        y_new = np.exp(np.interp(log_x_new, log_x_valid, log_y_valid))
    else:
        y_new = np.interp(x_new, x_valid, y_valid)

    # Low-energy extrapolation / threshold floor check
    low_mask = x_new < x_valid[0]
    if np.any(low_mask):
        if is_threshold_rxn:
            y_new[low_mask] = 0.0
        else:
            n_pts = min(n_pts_fit, len(x_valid))
            if n_pts >= 2:
                p = np.polyfit(log_x_valid[:n_pts], log_y_valid[:n_pts], 1)
                slope, intercept = p[0], p[1]
                if is_loglog:
                    y_new[low_mask] = np.exp(slope * log_x_new[low_mask] + intercept)
                else:
                    y_new[low_mask] = slope * log_x_new[low_mask] + intercept
            else:
                y_new[low_mask] = y_valid[0]

    # High-energy trend extrapolation (E > max(x_valid))
    high_mask = x_new > x_valid[-1]
    if np.any(high_mask):
        if is_threshold_rxn:
            y_new[high_mask] = y_valid[-1]
        else:
            n_pts = min(n_pts_fit, len(x_valid))
            if n_pts >= 2:
                p = np.polyfit(log_x_valid[-n_pts:], log_y_valid[-n_pts:], 1)
                slope, intercept = p[0], p[1]
                if is_loglog:
                    y_new[high_mask] = np.exp(slope * log_x_new[high_mask] + intercept)
                else:
                    y_new[high_mask] = slope * log_x_new[high_mask] + intercept
            else:
                y_new[high_mask] = y_valid[-1]

    return y_new

def load_fms_file(filename):
    if not filename or filename == "NONE" or not os.path.exists(filename):
        return None, None
    try:
        energies = []
        fms_vals = []
        with open(filename, 'r') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                cols = line_str.split()
                if len(cols) >= 4:
                    try:
                        energies.append(float(cols[0]))
                        fms_vals.append(float(cols[3]))
                    except ValueError:
                        continue

        if not energies:
            return None, None

        energies = np.array(energies)
        fms_vals = np.array(fms_vals)
        sort_idx = np.argsort(energies)
        
        if np.max(energies) < 1000.0:
            energies = energies * 1.0e6

        return energies[sort_idx], fms_vals[sort_idx]

    except Exception:
        return None, None

def maxwell_boltzmann_spectrum(E, kT, A):
    if kT <= 0.0:
        return np.zeros_like(E)
    arg = np.clip(-E / kT, -700.0, 700.0)
    return A * (E / (kT**2)) * np.exp(arg)

def msacs_fun(kT, xs, E):
    if kT <= 0.0:
        return np.zeros_like(E)
    arg = np.clip(-E / kT, -700.0, 700.0)
    return (1.0 / (kT**2)) * E * xs * np.exp(arg)

def load_spectrum(filename):
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] >= 3:
        e_low = data[:, 0]
        e_up = data[:, 1]
        flux = data[:, 2]
        return 0.5 * (e_low + e_up), flux, e_up - e_low, e_low, e_up
    else:
        e_mid = data[:, 0]
        flux = data[:, 1]
        return e_mid, flux, np.zeros_like(e_mid), e_mid, e_mid

def load_2col(filename):
    """Loads 2-column data and automatically converts energy from MeV to eV if needed."""
    data = np.loadtxt(filename, comments='#', usecols=(0, 1), unpack=True)
    e_vals, xs_vals = data[0], data[1]
    
    if len(e_vals) > 0 and np.max(e_vals) < 1000.0:
        e_vals = e_vals * 1.0e6
        
    return e_vals, xs_vals

def resolve_sample_file(sample_material):
    primary_path = f"data/{sample_material}.ntot"
    if os.path.exists(primary_path):
        return primary_path

    possible_paths = [
        f"data/{sample_material}_Sigma.ntot",
        f"data/{sample_material}.Sigma_ntot",
        f"data/{sample_material}.Sigma.ntot"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Could not find microscopic total cross-section file for '{sample_material}'")

def resolve_reaction_file(iso, react):
    react_clean = str(react).lower().strip()
    if react_clean in ["41", "n4n", "n,4n", "17"]:
        primary = f"data/{iso}.n4n"
    elif react_clean in ["16", "n2n", "n,2n"]:
        primary = f"data/{iso}.n2n"
    elif react_clean in ["102", "ng", "n,g"]:
        primary = f"data/{iso}.ng"
    else:
        primary = f"data/{iso}.{react}"

    if os.path.exists(primary):
        return primary

    fallbacks = [
        f"data/{iso}.{react}",
        "data/Au197.n4n",
        "data/Au196.n4n"
    ]
    for p in fallbacks:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"Could not resolve reaction cross-section file for iso={iso}, react={react}")

def get_file_energy_bounds(input_file, is_3col=False):
    data = np.loadtxt(input_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if is_3col and data.shape[1] >= 2:
        e_min, e_max = np.min(data[:, 0]), np.max(data[:, 1])
    else:
        e_min, e_max = np.min(data[:, 0]), np.max(data[:, 0])

    if e_max < 1000.0:
        e_min *= 1.0e6
        e_max *= 1.0e6

    return e_min, e_max

def run_binnerize(input_file, num_cols, idx_x, idx_y, e_min, e_max, bpd, interp_type="linear"):
    file_min, file_max = get_file_energy_bounds(input_file, is_3col=(num_cols >= 3))
    eff_e_min = max(e_min, file_min)
    eff_e_max = min(e_max, file_max)

    stdin_data = f"{input_file}\n{num_cols} {idx_x} {idx_y}\n{eff_e_min:.6e} {eff_e_max:.6e} 1 {bpd:.0f} {interp_type}\n"
    process = subprocess.Popen(["python3", BINNERIZE_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=stdin_data)
    if process.returncode != 0:
        raise RuntimeError(f"binnerize.py failed:\n{stderr}")
    return np.loadtxt("out_binned")

def run_rebinnerize(input_file, num_cols, idx_elow, idx_eup, idx_val, e_min, e_max, bpd, interp_type="linear", iflag="others"):
    file_min, file_max = get_file_energy_bounds(input_file, is_3col=True)
    eff_e_min = max(e_min, file_min)
    eff_e_max = min(e_max, file_max)

    stdin_data = f"{input_file}\n{num_cols} {idx_elow} {idx_eup} {idx_val}\n{eff_e_min:.6e} {eff_e_max:.6e} 1 {bpd:.0f} {interp_type} {iflag}\n"
    process = subprocess.Popen(["python3", REBINNERIZE_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=stdin_data)
    if process.returncode != 0:
        raise RuntimeError(f"rebinnerize.py failed:\n{stderr}")
    return np.loadtxt("out_rebinned")

# ==========================================
# Main Workflow
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Spectral-Averaged Cross Section (SACS) Integration Driver (calculate_sacs3_2.py)"
    )
    parser.add_argument("nspectrum", help="Neutron spectrum filename in data/")
    parser.add_argument("iso", help="Target reaction isotope (e.g., Au197)")
    parser.add_argument("react", help="Reaction channel (ng, n2n, n4n, or line code 41)")
    parser.add_argument("fthick_mm", type=float, help="Filter thickness in mm")
    parser.add_argument("sample_mat", help="Sample element label (e.g., Au)")
    parser.add_argument("sample_thick_atoms_per_barn", type=float, help="Target thickness in atoms/barn")
    parser.add_argument("bpd", type=float, help="Bins per decade")
    parser.add_argument("bin_mode", type=int, choices=[0, 1], help="Binning mode: 0=Late, 1=Early")
    parser.add_argument("extrap_mode", nargs="?", default="1", help="Extrapolation mode (default: 1)")
    parser.add_argument("interp_flag", nargs="?", default="1/E", help="Flux interpolation shape ('1/E' or 'linear')")
    parser.add_argument("fms_file", nargs="?", default=None, help="Path to pointwise Monte Carlo F_ms output file")

    args = parser.parse_args()

    filter_file = "data/B4Cx_Sigma.ntot" if os.path.exists("data/B4Cx_Sigma.ntot") else "data/B4Cx.Sigma_ntot"
    fln_spec = f"data/{args.nspectrum}"
    fln_react = resolve_reaction_file(args.iso, args.react)
    fln_sample = resolve_sample_file(args.sample_mat) if args.sample_thick_atoms_per_barn > 0.0 else None

    ff = 0.1 * args.fthick_mm
    E1, E2 = 1.0e-3, 1.0e8

    react_clean = str(args.react).lower().strip()
    is_threshold_rxn = react_clean in ["n2n", "n,2n", "16", "n4n", "n,4n", "41", "17"]

    fms_x, fms_y = load_fms_file(args.fms_file)

    if args.bin_mode == 1:
        rebinned_spec = run_rebinnerize(fln_spec, 3, 1, 2, 3, E1, E2, args.bpd, interp_type=args.interp_flag, iflag="others")
        binned_filt = run_binnerize(filter_file, 2, 1, 2, E1, E2, args.bpd, interp_type="linear")
        binned_react = run_binnerize(fln_react, 2, 1, 2, E1, E2, args.bpd, interp_type="linear")

        if args.sample_thick_atoms_per_barn > 0.0:
            binned_samp = run_binnerize(fln_sample, 2, 1, 2, E1, E2, args.bpd, interp_type="linear")
            sigma_samp = binned_samp[:, 2]
        else:
            sigma_samp = np.zeros(len(binned_react))

        e_low = rebinned_spec[:, 0]
        e_up = rebinned_spec[:, 1]
        dE_grid = e_up - e_low
        E_grid = 0.5 * (e_low + e_up)

        phi_raw = rebinned_spec[:, 2]
        sigma_filt = binned_filt[:, 2]
        sigma_react = binned_react[:, 2]
        if is_threshold_rxn:
            x_r, y_r = load_2col(fln_react)
            e_thresh = np.min(x_r[y_r > 0]) if np.any(y_r > 0) else 0.0
            sigma_react = np.where(E_grid >= e_thresh, sigma_react, 0.0)
    else:
        x_spec, y_spec, dE_spec, e_low_spec, e_up_spec = load_spectrum(fln_spec)
        x_filt, y_filt = load_2col(filter_file)
        x_react, y_react = load_2col(fln_react)

        native_grid_list = [x_react[x_react > 0]]
        if args.sample_thick_atoms_per_barn > 0.0:
            x_samp, y_samp = load_2col(fln_sample)
            native_grid_list.append(x_samp[x_samp > 0])
        
        native_grid_list.extend([x_spec[x_spec > 0], x_filt[x_filt > 0]])
        E_master = np.unique(np.concatenate(native_grid_list))
        E_master.sort()
        
        if E_master[0] > E1:
            E_master = np.insert(E_master, 0, E1)
        if E_master[-1] < E2:
            E_master = np.append(E_master, E2)

        mask = (E_master >= E1) & (E_master <= E2)
        E_grid = E_master[mask]
        dE_grid = None

        if args.interp_flag == "1/E":
            y_spec_weighted = y_spec * x_spec
            phi_raw = interp_trend_extrap(E_grid, x_spec, y_spec_weighted, n_pts_fit=5, is_loglog=True) / E_grid
        else:
            phi_raw = interp_trend_extrap(E_grid, x_spec, y_spec, n_pts_fit=5, is_loglog=True)

        sigma_filt = interp_trend_extrap(E_grid, x_filt, y_filt, n_pts_fit=5, is_loglog=True)
        sigma_react = interp_trend_extrap(E_grid, x_react, y_react, n_pts_fit=5, is_loglog=True, is_threshold_rxn=is_threshold_rxn)
        sigma_samp = interp_trend_extrap(E_grid, x_samp, y_samp, n_pts_fit=5, is_loglog=True) if args.sample_thick_atoms_per_barn > 0.0 else np.zeros_like(E_grid)

    attn_filt = np.where(ff * sigma_filt < 99.0, np.exp(-sigma_filt * ff), 0.0)
    phi_filt = phi_raw * attn_filt

    samp_opt_depth = sigma_samp * args.sample_thick_atoms_per_barn
    ssf_factor = np.where(
        samp_opt_depth > 0.0,
        np.where(
            samp_opt_depth < 99.0,
            (1.0 - np.exp(-samp_opt_depth)) / samp_opt_depth,
            0.0
        ),
        1.0
    )
    phi_ssf = phi_filt * ssf_factor

    if fms_x is not None and len(fms_x) > 0:
        fms_grid = np.interp(E_grid, fms_x, fms_y, left=1.0, right=1.0)
        fms_grid = np.where(sigma_react > 1.0e-12, fms_grid, 1.0)
        sigma_react_corr = sigma_react * fms_grid
    else:
        sigma_react_corr = sigma_react

    peak_idx = np.argmax(phi_filt)
    E_peak = E_grid[peak_idx]
    fit_mask = (E_grid >= E_peak / 50.0) & (E_grid <= E_peak * 50.0) if E_peak > 0.0 else (E_grid >= E1) & (E_grid <= E2)

    E_fit = E_grid[fit_mask] if np.sum(fit_mask) > 5 else E_grid
    phi_fit = phi_filt[fit_mask] if np.sum(fit_mask) > 5 else phi_filt

    kT_guess = float(E_peak) if E_peak > 0 else 30.0e3
    A_guess = float(np.sum(phi_fit * dE_grid[fit_mask])) if args.bin_mode == 1 else float(np.trapezoid(phi_fit, E_fit))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                maxwell_boltzmann_spectrum, E_fit, phi_fit,
                p0=[kT_guess, A_guess], bounds=((1e-3, 0.0), (1e8, np.inf)), maxfev=20000
            )
            kT, A_fit = popt[0], popt[1]
    except (RuntimeError, ValueError):
        kT, A_fit = kT_guess, A_guess

    phi_mb_fitted = maxwell_boltzmann_spectrum(E_grid, kT, A_fit)
    phi_mb_fitted = np.where(phi_mb_fitted < 1.0e-99, 0.0, phi_mb_fitted)

    mb_mask = (E_grid <= 50.0 * kT) & (E_grid > 0.0)

    if args.bin_mode == 1:
        nn = np.sum(phi_raw * dE_grid)
        fnn = np.sum(phi_filt * dE_grid)
        sacs = np.sum(phi_raw * sigma_react * dE_grid) / nn if nn > 0 else 0.0
        fsacs = np.sum(phi_filt * sigma_react * dE_grid) / fnn if fnn > 0 else 0.0
        ssfsacs = np.sum(phi_ssf * sigma_react * dE_grid) / fnn if fnn > 0 else 0.0
        ms_sacs = np.sum(phi_ssf * sigma_react_corr * dE_grid) / fnn if fnn > 0 else 0.0
        
        msacs_num = np.sum(msacs_fun(kT, sigma_react_corr[mb_mask], E_grid[mb_mask]) * dE_grid[mb_mask])
        msacs_denom = np.sum(msacs_fun(kT, 1.0, E_grid[mb_mask]) * dE_grid[mb_mask])
    else:
        nn = np.trapezoid(phi_raw, E_grid)
        fnn = np.trapezoid(phi_filt, E_grid)
        sacs = np.trapezoid(phi_raw * sigma_react, E_grid) / nn if nn > 0 else 0.0
        fsacs = np.trapezoid(phi_filt * sigma_react, E_grid) / fnn if fnn > 0 else 0.0
        ssfsacs = np.trapezoid(phi_ssf * sigma_react, E_grid) / fnn if fnn > 0 else 0.0
        ms_sacs = np.trapezoid(phi_ssf * sigma_react_corr, E_grid) / fnn if fnn > 0 else 0.0
        
        msacs_num = np.trapezoid(msacs_fun(kT, sigma_react_corr[mb_mask], E_grid[mb_mask]), E_grid[mb_mask])
        msacs_denom = np.trapezoid(msacs_fun(kT, 1.0, E_grid[mb_mask]), E_grid[mb_mask])

    ssfact = ssfsacs / fsacs if fsacs > 0 else 1.0

    # Explicit threshold check for MACS / m-SACS
    if is_threshold_rxn:
        msacs = 0.0
        macs = 0.0
    else:
        msacs = msacs_num / msacs_denom if msacs_denom > 0 else 0.0
        macs = (2.0 / np.sqrt(np.pi)) * msacs

    # Output Formatting & allcols Export
    if args.bin_mode == 0:
        E_mid_bounds = 0.5 * (E_grid[:-1] + E_grid[1:])
        E_low_grid = np.zeros_like(E_grid)
        E_up_grid = np.zeros_like(E_grid)

        E_low_grid[0] = E_grid[0] - (E_mid_bounds[0] - E_grid[0])
        E_low_grid[1:] = E_mid_bounds
        E_up_grid[:-1] = E_mid_bounds
        E_up_grid[-1] = E_grid[-1] + (E_grid[-1] - E_mid_bounds[-1])

        np.savetxt("tmp_raw_flux.dat", np.column_stack((E_low_grid, E_up_grid, phi_raw)))
        np.savetxt("tmp_filt_flux.dat", np.column_stack((E_low_grid, E_up_grid, phi_filt)))
        np.savetxt("tmp_ssf_flux.dat", np.column_stack((E_low_grid, E_up_grid, phi_ssf)))
        np.savetxt("tmp_mb_flux.dat", np.column_stack((E_low_grid, E_up_grid, phi_mb_fitted)))

        binned_raw = run_rebinnerize("tmp_raw_flux.dat", 3, 1, 2, 3, E1, E2, args.bpd, interp_type=args.interp_flag)
        binned_filt = run_rebinnerize("tmp_filt_flux.dat", 3, 1, 2, 3, E1, E2, args.bpd, interp_type=args.interp_flag)
        binned_ssf = run_rebinnerize("tmp_ssf_flux.dat", 3, 1, 2, 3, E1, E2, args.bpd, interp_type=args.interp_flag)
        binned_mb = run_rebinnerize("tmp_mb_flux.dat", 3, 1, 2, 3, E1, E2, args.bpd, interp_type=args.interp_flag)

        binned_mb[:, 2] = np.where(binned_mb[:, 2] < 1.0e-99, 0.0, binned_mb[:, 2])

        e_mid_out = 0.5 * (binned_raw[:, 0] + binned_raw[:, 1])
        xs_vals = interp_trend_extrap(e_mid_out, x_react, y_react, n_pts_fit=5, is_loglog=True, is_threshold_rxn=is_threshold_rxn)
        binned_xs = np.column_stack((binned_raw[:, 0], binned_raw[:, 1], xs_vals))
    else:
        binned_raw = np.column_stack((e_low, e_up, phi_raw))
        binned_filt = np.column_stack((e_low, e_up, phi_filt))
        binned_ssf = np.column_stack((e_low, e_up, phi_ssf))
        binned_mb = np.column_stack((e_low, e_up, phi_mb_fitted))
        binned_xs = np.column_stack((e_low, e_up, sigma_react))

    allcols_data = np.column_stack((
        binned_raw[:, 0], binned_raw[:, 1], binned_raw[:, 2],
        binned_filt[:, 2], binned_ssf[:, 2], binned_mb[:, 2], binned_xs[:, 2]
    ))

    kT_keV = kT * 1.0e-3
    if kT >= 1e6:
        kt_str = f"{kT*1e-6:8.3f} MeV"
    elif kT >= 1e3:
        kt_str = f"{kT_keV:8.3f} keV"
    else:
        kt_str = f"{kT:8.3f} eV"

    mode_str = "Early Rebin/Bin" if args.bin_mode == 1 else "Late Rebin/Bin (Master Grid)"
    header = (
        f"neutron spectrum : {args.nspectrum}\n"
        f"binning mode     : {mode_str}\n"
        f"flux interp flag : {args.interp_flag}\n"
        f"filter thickness : {args.fthick_mm:8.1f} mm\n"
        f"sample thickness : {args.sample_thick_atoms_per_barn:8.3e} atoms/b\n"
        f"kT fitted        : {kt_str} ({kT:10.3e} eV)\n"
        f"m-SACS           : {msacs:10.3e} b\n"
        f"MACS             : {macs:10.3e} b\n"
        f"  elow[eV]     eup[eV]       nflux       f-nflux     ssf-nflux    mb-nflux    iso_xs[b]"
    )
    np.savetxt("allcols", allcols_data, fmt="%.6e", header=header, comments="# ")

    # Clean Up Temp Files
    tmp_files = [
        "tmp_raw_flux.dat", "tmp_filt_flux.dat", "tmp_ssf_flux.dat",
        "tmp_mb_flux.dat", "out_binned", "out_rebinned"
    ]
    for tmp in tmp_files:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"# Mode            : {mode_str}")
    print(f"# nspectrum       : {args.nspectrum}")
    print(f"# Interp Flag     : {args.interp_flag}")
    print(f"# kT-fitted        : {kt_str} ({kT:10.3e} eV)")
    print(f"# m-SACS       [b]: {msacs:10.3e}")
    print(f"# MACS         [b]: {macs:10.3e}")
    print("#     fthick[mm]   kT[keV]      nn/pp    f-nn/pp    SACS[b]  f-SACS[b] m-SACS[b] ssf-SACS[b] ms-SACS[b]  ss-factor")
    print(f"line: {args.fthick_mm:8.2f}  {kT_keV:9.3e}  {nn:8.3e}  {fnn:8.3e}  {sacs:8.3e}  {fsacs:8.3e}  {msacs:8.3e}  {ssfsacs:8.3e}  {ms_sacs:8.3e}   {ssfact:8.3f}")

if __name__ == "__main__":
    main()
