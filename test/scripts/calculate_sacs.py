#!/usr/bin/env python3

import sys
import subprocess
import os
import argparse
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expn

sys.path.insert(0, os.path.dirname(__file__))

# Avogadro Constant [atoms/mol]
NA = 6.02214076e23

def load_element_properties_table(filename="data/element_properties.dat"):
    """Reads elemental properties from a text file in data/."""
    path = resolve_data_file(filename)
    element_properties = {}
    
    with open(path, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            cols = line_str.split()
            if len(cols) >= 5:
                z = int(cols[0])
                sym = cols[1]
                name = cols[2]
                molar_mass = float(cols[3])
                density = float(cols[4])
                element_properties[z] = (sym, name, molar_mass, density)
                
    return element_properties

def lookup_element(identifier):
    """Looks up molar mass and density by chemical symbol or atomic number Z."""
    elem_dict = load_element_properties_table()
    ident_str = str(identifier).strip()
    
    if ident_str.isdigit():
        z = int(ident_str)
        if z in elem_dict:
            _, _, molar_mass, density = elem_dict[z]
            return molar_mass, density

    sym_clean = ident_str.capitalize()
    for z, (sym, name, molar_mass, density) in elem_dict.items():
        if sym.capitalize() == sym_clean:
            return molar_mass, density

    raise KeyError(
        f"Material '{identifier}' is not in the physical property table. "
        f"Provide a valid symbol (e.g. Si, Au) or Z number (1-98)."
    )

def convert_um_to_atoms_per_barn(thick_um, sample_mat):
    molar_mass, density = lookup_element(sample_mat)
    thick_cm = thick_um * 1e-4
    atoms_per_cm2 = (density * thick_cm / molar_mass) * NA
    return atoms_per_cm2 * 1e-24

def convert_mm_to_atoms_per_barn(thick_mm, sample_mat):
    molar_mass, density = lookup_element(sample_mat)
    thick_cm = thick_mm * 1e-1
    atoms_per_cm2 = (density * thick_cm / molar_mass) * NA
    return atoms_per_cm2 * 1e-24

def convert_cm_to_atoms_per_barn(thick_cm, sample_mat):
    molar_mass, density = lookup_element(sample_mat)
    atoms_per_cm2 = (density * thick_cm / molar_mass) * NA
    return atoms_per_cm2 * 1e-24

def convert_areal_density_to_atoms_per_barn(areal_g_cm2, sample_mat):
    molar_mass, _ = lookup_element(sample_mat)
    atoms_per_cm2 = (areal_g_cm2 / molar_mass) * NA
    return atoms_per_cm2 * 1e-24

def convert_atoms_per_barn_to_um(atoms_per_barn, sample_mat):
    try:
        molar_mass, density = lookup_element(sample_mat)
        atoms_per_cm2 = atoms_per_barn * 1e24
        thick_cm = (atoms_per_cm2 * molar_mass) / (density * NA)
        return thick_cm * 1e4
    except KeyError:
        return 0.0

def convert_atoms_per_barn_to_areal_density(atoms_per_barn, sample_mat):
    try:
        molar_mass, _ = lookup_element(sample_mat)
        atoms_per_cm2 = atoms_per_barn * 1e24
        return (atoms_per_cm2 * molar_mass) / NA
    except KeyError:
        return 0.0

def resolve_helper_script(script_name):
    """Searches for helper scripts in 'scripts/' or 'src/' subfolders locally or in home directory."""
    candidate_dirs = [
        os.path.join(os.getcwd(), "src"),
        os.path.join(os.getcwd(), "scripts"),
        os.path.expanduser("~/cernbox/research/src"),
        os.path.expanduser("~/cernbox/research/scripts")
    ]
    
    for d in candidate_dirs:
        path = os.path.join(d, script_name)
        if os.path.exists(path):
            return path
        
    raise FileNotFoundError(
        f"Could not find '{script_name}' in any of the following subfolders:\n" +
        "\n".join([f" - {d}" for d in candidate_dirs])
    )

BINNERIZE_PATH = resolve_helper_script("binnerize.py")
REBINNERIZE_PATH = resolve_helper_script("rebinnerize.py")

def log_trapezoid(y, x):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return 0.0

    dx = np.diff(x)
    y_safe = np.where(y <= 0.0, 1e-300, y)

    log_x = np.log(x)
    log_y = np.log(y_safe)

    d_log_x = np.diff(log_x)
    d_log_y = np.diff(log_y)

    alpha = d_log_y / d_log_x

    mask_singularity = np.abs(alpha + 1.0) < 1e-5
    mask_zero = (y[:-1] <= 0.0) | (y[1:] <= 0.0)
    mask_linear = mask_singularity | mask_zero

    integral_segments = np.zeros(len(x) - 1, dtype=np.float64)

    if np.any(mask_linear):
        integral_segments[mask_linear] = 0.5 * (y[:-1][mask_linear] + y[1:][mask_linear]) * dx[mask_linear]

    mask_power = ~mask_linear
    if np.any(mask_power):
        x1, x2 = x[:-1][mask_power], x[1:][mask_power]
        y1, y2 = y[:-1][mask_power], y[1:][mask_power]
        a = alpha[mask_power]
        integral_segments[mask_power] = (y2 * x2 - y1 * x1) / (a + 1.0)

    return float(np.sum(integral_segments))

def isotropic_ssf_flat_foil(tau):
    tau = np.maximum(np.asarray(tau, dtype=np.float64), 0.0)
    result = np.ones_like(tau)
    
    mask_large = tau >= 1e-4
    tau_l = tau[mask_large]
    
    if np.any(mask_large):
        e3 = expn(3, tau_l)
        result[mask_large] = (1.0 - 2.0 * e3) / (2.0 * tau_l)
        
    mask_small = ~mask_large
    if np.any(mask_small):
        tau_s = tau[mask_small]
        result[mask_small] = 1.0 - 0.75 * tau_s + (1.0 / 3.0) * (tau_s**2) - 0.125 * (tau_s**3)
        
    return np.clip(result, 0.0, 1.0)

def normal_ssf_flat_foil(tau):
    tau = np.maximum(np.asarray(tau, dtype=np.float64), 0.0)
    result = np.ones_like(tau)
    
    mask_large = tau >= 1e-5
    tau_l = tau[mask_large]
    
    if np.any(mask_large):
        result[mask_large] = (1.0 - np.exp(-tau_l)) / tau_l
        
    mask_small = ~mask_large
    if np.any(mask_small):
        tau_s = tau[mask_small]
        result[mask_small] = 1.0 - 0.5 * tau_s + (1.0 / 6.0) * (tau_s**2)
        
    return np.clip(result, 0.0, 1.0)

def interp_trend_extrap(x_new, x_orig, y_orig, n_pts_fit=5, is_loglog=True, is_threshold_rxn=False, extrap_mode=1):
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

    low_mask = x_new < x_valid[0]
    high_mask = x_new > x_valid[-1]

    if not extrap_mode:
        if np.any(low_mask):
            y_new[low_mask] = 0.0
        if np.any(high_mask):
            y_new[high_mask] = 0.0 if not is_threshold_rxn else y_valid[-1]
        return y_new

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

    if np.any(high_mask):
        if is_threshold_rxn:
            y_new[high_mask] = y_valid[-1]
        else:
            n_pts = min(n_pts_fit, len(x_valid))
            if n_pts >= 2:
                p = np.polyfit(log_x_valid[-n_pts:], log_y_valid[-n_pts:], 1)
                slope, intercept = p[0], p[1]
                slope = min(slope, 0.0)
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
    data = np.loadtxt(filename, comments='#', usecols=(0, 1), unpack=True)
    e_vals, xs_vals = data[0], data[1]
    
    if len(e_vals) > 0 and np.max(e_vals) < 1000.0:
        e_vals = e_vals * 1.0e6
        
    return e_vals, xs_vals

def get_file_num_cols(input_file):
    """Detects the number of data columns in a text file."""
    data = np.loadtxt(input_file)
    if data.ndim == 1:
        return 1
    return data.shape[1]

def resolve_data_file(filename):
    """Ensures file is referenced from the 'data/' folder."""
    if not filename:
        return None
    path = filename if filename.startswith("data/") else os.path.join("data", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find required data file: '{path}'")
    return path

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

def run_binnerize(input_file, num_cols, idx_x, idx_y, e_min, e_max, bpd, interp_type="linear", custom_grid_file=None):
    """Calls binnerize.py using standard BPD/Linear mode or custom target grid mode."""
    if custom_grid_file is not None:
        stdin_data = (
            f"{input_file}\n"
            f"{num_cols} {idx_x} {idx_y}\n"
            f"0.0 0.0 1 0 {interp_type} others\n"
            f"{custom_grid_file}\n"
            f"2 1 2\n"
        )
    else:
        file_min, file_max = get_file_energy_bounds(input_file, is_3col=(num_cols >= 3))
        eff_e_min = max(e_min, file_min)
        eff_e_max = min(e_max, file_max)
        stdin_data = f"{input_file}\n{num_cols} {idx_x} {idx_y}\n{eff_e_min:.6e} {eff_e_max:.6e} 1 {bpd:.0f} {interp_type} others\n"

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

def fit_maxwell_boltzmann_native(x_spec, y_spec, E1, E2, filter_file=None, filter_thick_cm=0.0, interp_flag="1/E", extrap_mode=1):
    """Fits Maxwell-Boltzmann distribution directly on the native spectrum grid."""
    mask_range = (x_spec >= E1) & (x_spec <= E2) & (y_spec > 0.0)
    x_fit = x_spec[mask_range]
    y_fit = y_spec[mask_range]

    if filter_file and os.path.exists(filter_file) and filter_thick_cm > 0.0:
        x_filt, y_filt = load_2col(filter_file)
        sigma_filt = interp_trend_extrap(x_fit, x_filt, y_filt, n_pts_fit=5, is_loglog=True, extrap_mode=extrap_mode)
        
        opt_depth_filt = sigma_filt * filter_thick_cm
        attn_filt = np.where(opt_depth_filt < 99.0, np.exp(-opt_depth_filt), 0.0)
        y_fit = y_fit * attn_filt

    peak_idx = np.argmax(y_fit)
    E_peak = x_fit[peak_idx]
    
    fit_mask = (x_fit >= E_peak / 50.0) & (x_fit <= E_peak * 50.0) if E_peak > 0.0 else np.ones_like(x_fit, dtype=bool)
    x_fit_sub = x_fit[fit_mask] if np.sum(fit_mask) > 5 else x_fit
    y_fit_sub = y_fit[fit_mask] if np.sum(fit_mask) > 5 else y_fit

    kT_guess = float(E_peak) if E_peak > 0 else 30.0e3
    A_guess = float(log_trapezoid(y_fit_sub, x_fit_sub))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                maxwell_boltzmann_spectrum, x_fit_sub, y_fit_sub,
                p0=[kT_guess, A_guess], bounds=((1e-3, 0.0), (1e8, np.inf)), maxfev=20000
            )
            return popt[0], popt[1]
    except (RuntimeError, ValueError):
        return kT_guess, A_guess

def print_usage():
    usage_text = """
================================================================================
Spectral-Averaged Cross Section (SACS) Calculator - calculate_sacs.py
================================================================================

Usage:
  python3 calculate_sacs.py -ns nspectrum -samplexs react_xs sample_ntot \\
                            [-samplemat MAT] [-samplemm MM | -sampleum UM] \\
                            [-filter filter_xs fthick_mm] [-bpd BPD] [-bin_mode MODE]

Parameters:
  -ns, --nspectrum     Neutron spectrum filename in data/ (e.g., NEAR-MCecc)
  -samplexs            Reaction cross-section and sample total cross-section files in data/
                       (e.g., au197ng.b81 au197ntot.b81)
  -samplemat           Sample element symbol or Z number (e.g., Au or 79)
  -samplemm            Sample thickness in mm
  -sampleum, -u        Sample thickness in micrometers [um]
  -samplethick, -a     Sample target thickness directly in atoms/barn

Filter Option:
  -filter filter_xs fthick_mm
                       Filter macroscopic cross-section file in data/ [1/cm] and thickness in mm.
                       If omitted, filter attenuation is skipped.

Optional Options:
  -bpd                 Bins per decade (default: 200)
  -bin_mode            Binning mode: 0 = Late Rebin/Bin (default), 1 = Early Rebin/Bin
  -extrap_mode         Extrapolation mode: 1 = Enabled (default), 0 = Disabled
  -interp_flag         Flux interpolation shape: '1/E' (default) or 'linear'
  -fms_file            Path to pointwise Monte Carlo F_ms output file

Examples:
  # Standard run (No Filter):
  python3 calculate_sacs.py -ns NEAR-MCecc -samplemat Au -samplexs au197ng.b81 au197ntot.b81 -sampleum 100

  # Run with a 10 mm B4C filter:
  python3 calculate_sacs.py -ns NEAR-MCecc -samplemat Au -samplexs au197ng.b81 au197ntot.b81 -sampleum 100 -filter B4Cx_Sigma.ntot 10.0
"""
    print(usage_text)

def main():
    if len(sys.argv) == 1:
        print_usage()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Spectral-Averaged Cross Section (SACS) Integration Driver (calculate_sacs.py)",
        add_help=True
    )
    
    # Spectrum input
    parser.add_argument("-ns", "--nspectrum", required=True, help="Neutron spectrum filename in data/ (e.g., NEAR-MCecc)")
    
    # Sample cross section inputs
    parser.add_argument("-samplexs", nargs=2, metavar=("xs_file", "sample_xs_file"), required=True,
                        help="Reaction cross-section and total cross-section filenames in data/")
    
    # Sample material & thickness inputs
    parser.add_argument("-samplemat", "-m", "--sample_mat", type=str, default=None, help="Sample element symbol (e.g., Au) or Z number")
    parser.add_argument("-samplemm", type=float, default=None, help="Sample thickness in millimeters [mm]")
    parser.add_argument("-sampleum", "-u", "--thick_um", type=float, default=None, help="Sample thickness in micrometers [um]")
    parser.add_argument("-samplethick", "-a", type=float, default=0.0, help="Sample thickness in atoms/barn")

    # Filter option
    parser.add_argument("-filter", nargs=2, metavar=("filter_xs_file", "fthick_mm"), default=None,
                        help="Filter macroscopic cross-section file in data/ and thickness in mm")

    # Processing options
    parser.add_argument("-bpd", type=float, default=200.0, help="Bins per decade (default: 200)")
    parser.add_argument("-bin_mode", type=int, choices=[0, 1], default=0, help="Binning mode: 0=Late (default), 1=Early")
    parser.add_argument("-extrap_mode", type=int, default=1, help="Extrapolation mode (1=enabled [default], 0=disabled)")
    parser.add_argument("-interp_flag", default="1/E", help="Flux interpolation shape ('1/E' or 'linear')")
    parser.add_argument("-fms_file", default=None, help="Path to pointwise Monte Carlo F_ms output file")
    
    parser.add_argument(
        "--ssf_model",
        type=str,
        choices=["isotropic", "normal"],
        default="normal",
        help="Self-shielding angular flux model: 'normal' (default) or 'isotropic'"
    )

    args = parser.parse_args()

    # Unpack sample cross section files
    fln_react_raw, fln_sample_raw = args.samplexs

    # Process -filter argument
    if args.filter:
        raw_filter_file, raw_fthick_str = args.filter
        try:
            effective_fthick_mm = float(raw_fthick_str)
        except ValueError:
            raise ValueError(f"Invalid filter thickness passed to -filter: '{raw_fthick_str}'")
        filter_xs_filename = raw_filter_file
    else:
        effective_fthick_mm = 0.0
        filter_xs_filename = None

    filter_thick_cm = 0.1 * effective_fthick_mm

    # Dynamic sample target thickness determination
    if any(opt is not None for opt in [args.samplemm, args.thick_um]):
        if args.sample_mat is None:
            raise ValueError("Parameter -samplemat is required when providing -samplemm or -sampleum for unit conversions.")
        
        if args.samplemm is not None:
            sample_thick_atoms_per_barn = convert_mm_to_atoms_per_barn(args.samplemm, args.sample_mat)
            sample_thick_um = args.samplemm * 1000.0
            sample_areal_g_cm2 = convert_atoms_per_barn_to_areal_density(sample_thick_atoms_per_barn, args.sample_mat)
        elif args.thick_um is not None:
            sample_thick_atoms_per_barn = convert_um_to_atoms_per_barn(args.thick_um, args.sample_mat)
            sample_thick_um = args.thick_um
            sample_areal_g_cm2 = convert_atoms_per_barn_to_areal_density(sample_thick_atoms_per_barn, args.sample_mat)
    else:
        sample_thick_atoms_per_barn = args.samplethick
        if args.sample_mat is not None and sample_thick_atoms_per_barn > 0.0:
            sample_thick_um = convert_atoms_per_barn_to_um(sample_thick_atoms_per_barn, args.sample_mat)
            sample_areal_g_cm2 = convert_atoms_per_barn_to_areal_density(sample_thick_atoms_per_barn, args.sample_mat)
        else:
            sample_thick_um = 0.0
            sample_areal_g_cm2 = 0.0

    # Resolve input files in data/
    fln_spec = resolve_data_file(args.nspectrum)
    fln_react = resolve_data_file(fln_react_raw)
    fln_sample = resolve_data_file(fln_sample_raw) if sample_thick_atoms_per_barn > 0.0 else None
    filter_file = resolve_data_file(filter_xs_filename) if (effective_fthick_mm > 0.0 and filter_xs_filename) else None

    E1, E2 = 1.0e-3, 1.0e9

    is_threshold_rxn = any(ext in fln_react_raw.lower() for ext in ["n2n", "n4n", ".16", ".17", ".41"])

    fms_x, fms_y = load_fms_file(args.fms_file)

    # Load original native spectrum first
    x_spec, y_spec, dE_spec, e_low_spec, e_up_spec = load_spectrum(fln_spec)

    # Fit Maxwell-Boltzmann kT directly on the original native spectrum grid
    kT, A_fit = fit_maxwell_boltzmann_native(
        x_spec, y_spec, E1, E2, filter_file=filter_file,
        filter_thick_cm=filter_thick_cm,
        interp_flag=args.interp_flag, extrap_mode=args.extrap_mode
    )

    if args.bin_mode == 1:
        spec_cols = get_file_num_cols(fln_spec)
        
        # Step 1: Rebin/Bin flux spectrum first to generate the master energy grid
        if spec_cols >= 3:
            rebinned_spec = run_rebinnerize(fln_spec, 3, 1, 2, 3, E1, E2, args.bpd, interp_type=args.interp_flag, iflag="others")
        else:
            rebinned_spec = run_binnerize(fln_spec, 2, 1, 2, E1, E2, args.bpd, interp_type=args.interp_flag)
        
        # Save reference grid (E_low, E_up) for binnerize.py custom grid mode
        tmp_grid_file = "tmp_master_grid.dat"
        np.savetxt(tmp_grid_file, rebinned_spec[:, :2], fmt="%.6e")

        # Step 2: Binnerize pointwise files onto the reference grid via binnerize.py
        binned_react = run_binnerize(fln_react, 2, 1, 2, E1, E2, args.bpd, interp_type="linear", custom_grid_file=tmp_grid_file)

        if filter_file and os.path.exists(filter_file):
            binned_filt = run_binnerize(filter_file, 2, 1, 2, E1, E2, args.bpd, interp_type="linear", custom_grid_file=tmp_grid_file)
            sigma_filt = binned_filt[:, 2]
        else:
            sigma_filt = np.zeros(len(binned_react))

        if sample_thick_atoms_per_barn > 0.0:
            binned_samp = run_binnerize(fln_sample, 2, 1, 2, E1, E2, args.bpd, interp_type="linear", custom_grid_file=tmp_grid_file)
            sigma_samp = binned_samp[:, 2]
        else:
            sigma_samp = np.zeros(len(binned_react))

        e_low = rebinned_spec[:, 0]
        e_up = rebinned_spec[:, 1]
        dE_grid = e_up - e_low
        E_grid = 0.5 * (e_low + e_up)

        phi_raw = rebinned_spec[:, 2]
        sigma_react = binned_react[:, 2]
        if is_threshold_rxn:
            x_r, y_r = load_2col(fln_react)
            e_thresh = np.min(x_r[y_r > 0]) if np.any(y_r > 0) else 0.0
            sigma_react = np.where(E_grid >= e_thresh, sigma_react, 0.0)

    else:
        x_react, y_react = load_2col(fln_react)

        native_grid_list = [x_react[x_react > 0]]
        if sample_thick_atoms_per_barn > 0.0:
            x_samp, y_samp = load_2col(fln_sample)
            native_grid_list.append(x_samp[x_samp > 0])
        if fms_x is not None and len(fms_x) > 0:
            native_grid_list.append(fms_x[fms_x > 0])
        
        native_grid_list.append(x_spec[x_spec > 0])

        if filter_file and os.path.exists(filter_file):
            x_filt, y_filt = load_2col(filter_file)
            native_grid_list.append(x_filt[x_filt > 0])
        else:
            x_filt, y_filt = np.array([E1, E2]), np.array([0.0, 0.0])

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
            phi_raw = interp_trend_extrap(E_grid, x_spec, y_spec_weighted, n_pts_fit=5, is_loglog=True, extrap_mode=args.extrap_mode) / E_grid
        else:
            phi_raw = interp_trend_extrap(E_grid, x_spec, y_spec, n_pts_fit=5, is_loglog=True, extrap_mode=args.extrap_mode)

        if filter_file and os.path.exists(filter_file):
            sigma_filt = interp_trend_extrap(E_grid, x_filt, y_filt, n_pts_fit=5, is_loglog=True, extrap_mode=args.extrap_mode)
        else:
            sigma_filt = np.zeros_like(E_grid)

        sigma_react = interp_trend_extrap(E_grid, x_react, y_react, n_pts_fit=5, is_loglog=True, is_threshold_rxn=is_threshold_rxn, extrap_mode=args.extrap_mode)
        sigma_samp = interp_trend_extrap(E_grid, x_samp, y_samp, n_pts_fit=5, is_loglog=True, extrap_mode=args.extrap_mode) if sample_thick_atoms_per_barn > 0.0 else np.zeros_like(E_grid)

    # Filter attenuation calculation
    opt_depth_filt = sigma_filt * filter_thick_cm
    attn_filt = np.where(opt_depth_filt < 99.0, np.exp(-opt_depth_filt), 0.0) if effective_fthick_mm > 0.0 else np.ones_like(E_grid)
    phi_filt = phi_raw * attn_filt

    samp_opt_depth = sigma_samp * sample_thick_atoms_per_barn

    if sample_thick_atoms_per_barn > 0.0:
        if args.ssf_model == "isotropic":
            ssf_factor = isotropic_ssf_flat_foil(samp_opt_depth)
        else:
            ssf_factor = normal_ssf_flat_foil(samp_opt_depth)
    else:
        ssf_factor = np.ones_like(E_grid)

    phi_ssf = phi_filt * ssf_factor

    if fms_x is not None and len(fms_x) > 0:
        fms_grid = np.exp(np.interp(np.log(E_grid), np.log(fms_x), np.log(fms_y), left=0.0, right=0.0))
        fms_grid = np.where(sigma_react > 1.0e-12, fms_grid, 1.0)
        sigma_react_corr = sigma_react * fms_grid
    else:
        sigma_react_corr = sigma_react

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
        nn = log_trapezoid(phi_raw, E_grid)
        fnn = log_trapezoid(phi_filt, E_grid)
        sacs = log_trapezoid(phi_raw * sigma_react, E_grid) / nn if nn > 0 else 0.0
        fsacs = log_trapezoid(phi_filt * sigma_react, E_grid) / fnn if fnn > 0 else 0.0
        ssfsacs = log_trapezoid(phi_ssf * sigma_react, E_grid) / fnn if fnn > 0 else 0.0
        ms_sacs = log_trapezoid(phi_ssf * sigma_react_corr, E_grid) / fnn if fnn > 0 else 0.0
        
        msacs_num = log_trapezoid(msacs_fun(kT, sigma_react_corr[mb_mask], E_grid[mb_mask]), E_grid[mb_mask])
        msacs_denom = log_trapezoid(msacs_fun(kT, 1.0, E_grid[mb_mask]), E_grid[mb_mask])

    ssfact = ssfsacs / fsacs if fsacs > 0 else 1.0

    if is_threshold_rxn:
        msacs = 0.0
        macs = 0.0
    else:
        msacs = msacs_num / msacs_denom if msacs_denom > 0 else 0.0
        if msacs < 1.0e-10:
            msacs = 0.0
        macs = (2.0 / np.sqrt(np.pi)) * msacs

    if args.bin_mode == 0:
        E_mid_bounds = 0.5 * (E_grid[:-1] + E_grid[1:])
        E_low_grid = np.zeros_like(E_grid)
        E_up_grid = np.zeros_like(E_grid)

        E_low_grid[0] = max(1.0e-3, E_grid[0] - (E_mid_bounds[0] - E_grid[0]))
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
        x_r, y_r = load_2col(fln_react)
        xs_vals = interp_trend_extrap(e_mid_out, x_r, y_r, n_pts_fit=5, is_loglog=True, is_threshold_rxn=is_threshold_rxn, extrap_mode=args.extrap_mode)
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
    kt_str = f"{kT_keV:8.3f} keV ({kT:10.3e} eV)"

    mode_str = "Early Rebin/Bin" if args.bin_mode == 1 else "Late Rebin/Bin (Master Grid)"
    sample_xs_str = fln_sample if fln_sample else "NONE"
    filter_xs_str = filter_file if filter_file else "NONE"
    
    filter_info_str = f"{effective_fthick_mm:8.1f} mm ({filter_thick_cm:8.2f} cm)"
    summary_line_str = f"line: {effective_fthick_mm:8.2f}  {kT_keV:9.3e}  {nn:8.3e}  {fnn:8.3e}  {sacs:8.3e}  {fsacs:8.3e}  {msacs:8.3e}  {ssfsacs:8.3e}  {ms_sacs:8.3e}   {ssfact:8.3f}"

    header = (
        f"neutron spectrum : {args.nspectrum}\n"
        f"reaction data    : {fln_react}\n"
        f"sample ntot data : {sample_xs_str}\n"
        f"filter ntot data : {filter_xs_str}\n"
        f"binning mode     : {mode_str}\n"
        f"ssf model        : {args.ssf_model}\n"
        f"flux interp flag : {args.interp_flag}\n"
        f"extrap mode      : {args.extrap_mode}\n"
        f"filter thickness : {filter_info_str}\n"
        f"sample thickness : {sample_thick_atoms_per_barn:8.3e} atoms/b ({sample_thick_um:8.2f} um, {sample_areal_g_cm2:8.3e} g/cm^2)\n"
        f"kT fitted        : {kt_str}\n"
        f"MB-SACS          : {msacs:10.3e} b\n"
        f"MACS             : {macs:10.3e} b\n"
        f"   fthick[mm]   kT[keV]      nn/pp    f-nn/pp    SACS[b]  f-SACS[b] MB-SACS[b] ssf-SACS[b] ms-SACS[b]  ss-factor\n"
        f"{summary_line_str}\n"
        f"  elow[eV]     eup[eV]       nflux       f-nflux     ssf-nflux    mb-nflux    iso_xs[b]"
    )
    np.savetxt("allcols", allcols_data, fmt="%.6e", header=header, comments="# ")

    tmp_files = [
        "tmp_raw_flux.dat", "tmp_filt_flux.dat", "tmp_ssf_flux.dat",
        "tmp_mb_flux.dat", "out_binned", "out_rebinned", "tmp_master_grid.dat"
    ]
    for tmp in tmp_files:
        if os.path.exists(tmp):
            os.remove(tmp)

    print(f"# Mode            : {mode_str}")
    print(f"# SSF Model       : {args.ssf_model}")
    print(f"# nspectrum       : {args.nspectrum}")
    print(f"# Reaction Data   : {fln_react}")
    print(f"# Sample ntot Data: {sample_xs_str}")
    print(f"# Filter ntot Data: {filter_xs_str}")
    print(f"# Interp Flag     : {args.interp_flag}")
    print(f"# Extrap Mode     : {args.extrap_mode}")
    print(f"# Filter Thick.   : {filter_info_str}")
    print(f"# Sample Thick.   : {sample_thick_atoms_per_barn:10.3e} atoms/b ({sample_thick_um:8.2f} um | {sample_areal_g_cm2:8.3e} g/cm^2)")
    print(f"# kT-fitted       : {kt_str}")
    print(f"# MB-SACS      [b]: {msacs:10.3e}")
    print(f"# MACS         [b]: {macs:10.3e}")
    print("#    fthick[mm]   kT[keV]      nn/pp    f-nn/pp    SACS[b]  f-SACS[b] MB-SACS[b] ssf-SACS[b] ms-SACS[b]  ss-factor")
    print(summary_line_str)

if __name__ == "__main__":
    main()
