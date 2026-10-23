#!/usr/bin/env python3

import sys
import os
import argparse
import subprocess
import numpy as np

# Ensure helper modules or scripts in candidate directories can be imported/resolved
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Physical Constants for Gold (197Au) Reactions
M_AU = 196.966569        # Molar mass [g/mol]
RHO_AU = 19.32           # Density of Gold [g/cm3]
NA = 6.02214076e23       # Avogadro constant [atoms/mol]
A_ISO = 1.0              # Monoisotopic 197Au

# Reaction Product Half-Lives [seconds]
T_HALF_198AU = 232856.6  # 198Au from (n,g)  (~2.6951 days)
T_HALF_196AU = 532790.4  # 196Au from (n,2n) (~6.1665 days)
T_HALF_194AU = 141120.0  # 194Au from (n,4n) (~39.2 hours)

# Standard n_TOF pulse intensity assumed in calculate_sacs.py spectrum files
PROTONS_PER_STANDARD_PULSE = 7.0e12

# Hard-coded SACS parameters passed downstream to calculate_sacs.py
TARGET_ISO = "Au197"
SAMPLE_MAT = "Au"
DEFAULT_BIN_MODE = "0"
DEFAULT_INTERP_FLAG = "1/E"  # Hard-coded to 1/E interpolation
DEFAULT_FILTER_XS_FILE = "B4Cx_Sigma.ntot"  # Macroscopic B4C filter cross-section file

# Cache spectrum integrals to avoid redundant I/O
SPECTRUM_FNN_CACHE = {}

USAGE_TEXT = """
Usage Examples:
  python3 scripts/calculate_counts.py inputs/input_NEAR_BurialDating cm2
  python3 scripts/calculate_counts.py -f NEAR-MCecc inputs/input_NEAR_BurialDating cm2 200 0
  python3 scripts/calculate_counts.py inputs/input_EAR1CC_BurialDating beam 200 1
"""


def log_trapezoid(y, x):
    """Log-log trapezoidal integration matching calculate_sacs.py quadrature."""
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


def interp_trend_extrap(x_new, x_orig, y_orig, n_pts_fit=5, is_loglog=True, extrap_mode=1):
    """Interpolates cross section onto spectrum energy grid matching calculate_sacs.py."""
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
            y_new[high_mask] = 0.0
        return y_new

    if np.any(low_mask):
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


def evaluate_spectrum_fnn(spectrum_file, fthick_mm=0.0):
    """
    Evaluates total incident neutron flux integral f_nn/pulse directly from spectrum file,
    accounting for B4C filter attenuation if fthick_mm > 0.
    """
    cache_key = (spectrum_file, float(fthick_mm))
    if cache_key in SPECTRUM_FNN_CACHE:
        return SPECTRUM_FNN_CACHE[cache_key]

    candidate_dirs = [
        os.path.join(os.getcwd(), "data"),
        os.path.join(script_dir, "data"),
        os.path.join(os.path.dirname(script_dir), "data")
    ]
    
    target_path = None
    for d in candidate_dirs:
        p = os.path.join(d, spectrum_file)
        if os.path.exists(p):
            target_path = p
            break

    if not target_path or not os.path.exists(target_path):
        raise FileNotFoundError(f"Spectrum file '{spectrum_file}' not found for integration.")

    data = np.loadtxt(target_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] >= 3:
        e_low = data[:, 0]
        e_up = data[:, 1]
        flux = data[:, 2]
        
        if np.max(e_up) < 1000.0:
            e_low = e_low * 1.0e6
            e_up = e_up * 1.0e6

        e_mid = 0.5 * (e_low + e_up)
    else:
        e_mid = data[:, 0]
        flux = data[:, 1]
        
        if np.max(e_mid) < 1000.0:
            e_mid = e_mid * 1.0e6

    # Apply macroscopic B4C filter attenuation if fthick_mm > 0
    if fthick_mm > 0.0:
        filter_path = None
        for d in candidate_dirs:
            fp = os.path.join(d, DEFAULT_FILTER_XS_FILE)
            if os.path.exists(fp):
                filter_path = fp
                break

        if filter_path and os.path.exists(filter_path):
            f_data = np.loadtxt(filter_path, comments='#', usecols=(0, 1), unpack=True)
            x_filt, y_filt = f_data[0], f_data[1]
            if np.max(x_filt) < 1000.0:
                x_filt = x_filt * 1.0e6
            
            sigma_filt = interp_trend_extrap(e_mid, x_filt, y_filt, n_pts_fit=5, is_loglog=True)
            filter_thick_cm = 0.1 * fthick_mm
            opt_depth_filt = sigma_filt * filter_thick_cm
            attn_filt = np.where(opt_depth_filt < 99.0, np.exp(-opt_depth_filt), 0.0)
            flux = flux * attn_filt

    fnn_val = float(log_trapezoid(flux, e_mid))
    SPECTRUM_FNN_CACHE[cache_key] = fnn_val
    return fnn_val


def resolve_helper_script(script_name):
    """
    Searches for helper scripts in current directory, same directory as this script,
    or standard research subfolders.
    """
    candidate_dirs = [
        script_dir,
        os.getcwd(),
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
        f"Could not find '{script_name}' in any of the candidate folders:\n" +
        "\n".join([f" - {d}" for d in candidate_dirs])
    )


def resolve_spectrum_file(area=None, explicit_flux=None):
    """
    Resolves neutron spectrum filename inside data/ based strictly on area.
    Defaults to 'NEAR-MCecc' if no area or explicit flux is provided.
    """
    if explicit_flux:
        spectrum_file = explicit_flux.strip()
    else:
        area_clean = str(area).strip() if area else ""
        
        if area_clean in ["EAR1", "EAR1CC"]:
            spectrum_file = "Z21-EAR1"
        elif area_clean == "EAR1FC":
            spectrum_file = "Z21-EAR1FC"
        elif area_clean == "EAR2":
            spectrum_file = "Z22-EAR2"
        elif area_clean == "NEAR":
            spectrum_file = "NEAR-MCecc"
        elif area_clean:
            spectrum_file = f"{area_clean}-MCecc"
        else:
            spectrum_file = "NEAR-MCecc"

    candidate_dirs = [
        os.path.join(os.getcwd(), "data"),
        os.path.join(script_dir, "data"),
        os.path.join(os.path.dirname(script_dir), "data")
    ]
    
    for d in candidate_dirs:
        target_path = os.path.join(d, spectrum_file)
        if os.path.exists(target_path):
            return spectrum_file

    sys.stderr.write(
        f"\n[FATAL ERROR] Spectrum file error:\n"
        f"  Spectrum filename : '{spectrum_file}'\n"
        f"  Status            : FILE NOT FOUND IN DATA DIRECTORIES\n"
        f"Execution terminated.\n\n"
    )
    sys.exit(1)


def check_flux_unit_convention(areas, flux_unit):
    """
    Checks if the chosen flux_unit ('cm2' or 'beam') matches standard conventions:
      - NEAR station -> standard spectrum unit is n/eV/cm2/pp ('cm2')
      - EAR1 / EAR2  -> standard spectrum unit is n/eV/pp ('beam')
    Returns list of warning string messages if a mismatch is detected.
    """
    warnings = []
    for area in set(areas):
        area_upper = str(area).upper().strip()
        if "NEAR" in area_upper and flux_unit == "beam":
            msg = (
                f"[CONVENTION WARNING] Area '{area}' standard spectrum is in n/eV/cm2/pp ('cm2'). "
                f"You selected 'beam' mode. Ensure this is intentional!"
            )
            warnings.append(msg)
        elif ("EAR1" in area_upper or "EAR2" in area_upper) and flux_unit == "cm2":
            msg = (
                f"[CONVENTION WARNING] Area '{area}' standard spectrum is in n/eV/pp ('beam'). "
                f"You selected 'cm2' mode. Ensure this is intentional!"
            )
            warnings.append(msg)
    return warnings


def get_fms_filename(mass_g, diam_mm, line_id):
    """
    Constructs the expected F_ms Monte Carlo filename inside msdatafiles/ based on 
    the geometry and reaction type (ng, n2n, n4n).
    """
    line_str = str(line_id).strip()
    if line_str == '41':
        react = "n4n"
    elif line_str in ['21', '22']:
        react = "n2n"
    else:
        react = "ng"
    
    diam_str = f"{float(diam_mm):.1f}"
    mass_str = f"{float(mass_g):.4f}"
    
    return f"msdatafiles/fms_d{diam_str}_m{mass_str}_{react}.dat"


def get_decay_constant(line_id):
    """Returns decay constant lambda [1/s] according to product isotope."""
    line_str = str(line_id).strip()
    if line_str == '41':
        t_half = T_HALF_194AU
    elif line_str in ['21', '22']:
        t_half = T_HALF_196AU
    else:
        t_half = T_HALF_198AU
    return np.log(2.0) / t_half


def get_xs_filenames(line_id):
    """Resolves standard cross section filenames for reaction and total cross section in data/."""
    line_str = str(line_id).strip()
    if line_str == '41':
        return "Au197.n4n", "Au197.ntot"
    elif line_str in ['21', '22']:
        return "Au197.n2n", "Au197.ntot"
    else:
        return "Au197.ng", "Au197.ntot"


def calculate_counts_from_cols(cols, flux_unit="cm2", bpd=200, extrap_mode=1, explicit_flux=None):
    """
    Computes total calculated counts using zero-indexed positional columns
    via calculate_sacs.py. Automatically detects and applies pointwise 
    Monte Carlo F_ms correction if the corresponding file exists.
    """
    area          = str(cols[0]).strip()
    fthick_mm     = float(cols[1])
    sample_name   = cols[2]
    mass_g        = float(cols[3])
    diam_mm       = float(cols[4])
    thick_mm      = float(cols[5])
    line_id       = str(cols[6]).strip()
    egam_keV      = float(cols[7])
    Igam          = float(cols[8])
    epsi          = float(cols[9])
    c_table       = float(cols[10])
    tirr_s        = float(cols[11])
    twait_s       = float(cols[12])
    tcount_s      = float(cols[13])
    total_protons = float(cols[14])
    bif           = float(cols[15])  # Beam Interception / Correction Factor

    lam = get_decay_constant(line_id)

    # Resolve specific F_ms file generated inside msdatafiles/
    fms_file = get_fms_filename(mass_g, diam_mm, line_id)
    fms_found = os.path.exists(fms_file)

    # 1. Target nuclei (NT) from mass
    NT = (mass_g / M_AU) * NA * A_ISO

    # 2. Sample Area [cm2] and Areal density [atoms/barn] from mass & diameter
    diam_cm = diam_mm * 0.1
    area_cm2 = np.pi * (0.5 * diam_cm)**2
    atoms_per_barn = (NT / area_cm2) * 1e-24

    spectrum_file = resolve_spectrum_file(area, explicit_flux=explicit_flux)
    rxn_xs_file, ntot_xs_file = get_xs_filenames(line_id)

    # Evaluates incident flux integral including B4C filter attenuation
    f_nn_pulse = evaluate_spectrum_fnn(spectrum_file, fthick_mm=fthick_mm)

    sacs_script_path = resolve_helper_script("calculate_sacs.py")

    # 3. Invoke calculate_sacs.py using standard flag interface
    cmd = [
        sys.executable, sacs_script_path,
        "-ns", spectrum_file,
        "-samplexs", rxn_xs_file, ntot_xs_file,
        "-samplemat", SAMPLE_MAT,
        "-samplethick", str(atoms_per_barn),
        "-bpd", str(bpd),
        "-bin_mode", DEFAULT_BIN_MODE,
        "-extrap_mode", str(extrap_mode),
        "-interp_flag", DEFAULT_INTERP_FLAG
    ]

    if fthick_mm > 0.0:
        cmd.extend(["-filter", DEFAULT_FILTER_XS_FILE, str(fthick_mm)])

    if fms_found:
        cmd.extend(["-fms_file", fms_file])

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"SACS calculation command failed for sample '{sample_name}' (line {line_id}).\n"
            f"Command run: {' '.join(cmd)}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        )

    # 4. Parse output summary line for SACS cross sections
    combined_output = res.stdout + "\n" + res.stderr
    matching_lines = [
        line for line in combined_output.splitlines() 
        if line.lstrip("#").strip().startswith("line:")
    ]

    if not matching_lines:
        raise RuntimeError(
            f"calculate_sacs.py executed successfully but did not produce summary output.\n"
            f"Command executed: {' '.join(cmd)}\n"
            f"Full Output Received:\n{combined_output}"
        )

    line_out = matching_lines[0]
    tokens = line_out.lstrip("#").strip().split()
    
    try:
        sacs_b           = float(tokens[5])     # Unattenuated SACS [b]
        f_sacs_b         = float(tokens[6])     # Filtered SACS [b]
        ssf_sacs_b       = float(tokens[8])     # Self-shielded SACS [b]
        ms_sacs_b        = float(tokens[9])     # Multiple-scattering corrected SACS [b]
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"Failed to parse tokens from: '{line_out}'") from e

    # Effective cross section in cm2 (using fully corrected ms_sacs_b)
    sigma_cm2 = ms_sacs_b * 1e-24

    # Equivalent standard 7e12 proton pulses
    equivalent_pulses = total_protons / PROTONS_PER_STANDARD_PULSE

    # Incident neutron flux and total incident neutrons strictly tied to geometry & BIF
    if flux_unit == "beam":
        total_neutrons_sample = f_nn_pulse * equivalent_pulses * bif
        phi_tot = total_neutrons_sample / area_cm2
    else:
        phi_tot = f_nn_pulse * equivalent_pulses * bif
        total_neutrons_sample = phi_tot * area_cm2

    # 5. Activation Calculations
    # Total Reaction Rate R = Production Rate [reactions / s]
    R = (NT * phi_tot * sigma_cm2) / tirr_s

    # Saturation Factor: Fraction of saturation activity reached at end of irradiation
    f_sat = 1.0 - np.exp(-lam * tirr_s)

    # Activity at End of Irradiation [Bq] and Specific Activity [Bq/g]
    A0 = R * f_sat
    a0_per_g = A0 / mass_g if mass_g > 0 else 0.0

    # Specific Activity at Saturation (t_irr -> infinity) [Bq/g]
    asat_per_g = R / mass_g if mass_g > 0 else 0.0

    # Specific Activity at Saturation per incident neutron [Bq/g/neutron]
    asat_per_neutron = asat_per_g / total_neutrons_sample if total_neutrons_sample > 0 else 0.0

    # Total calculated gamma counts
    counts_calc = (A0 / lam) * np.exp(-lam * twait_s) * (1.0 - np.exp(-lam * tcount_s)) * epsi * Igam

    display_react = "n4n" if line_id == '41' else ("n2n" if line_id in ['21', '22'] else "ng")
    return area, sample_name, line_id, display_react, fthick_mm, mass_g, diam_mm, thick_mm, sacs_b, f_sacs_b, ssf_sacs_b, ms_sacs_b, total_neutrons_sample, a0_per_g, asat_per_neutron, f_sat, counts_calc, c_table, ratio_val(counts_calc, c_table), bif, spectrum_file, fms_found


def ratio_val(c_calc, c_table):
    return c_calc / c_table if c_table > 0 else 0.0


def load_raw_data_lines(filename):
    """Loads data lines ignoring comments (#) or blank lines."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file '{filename}' not found.")

    data_rows = []
    with open(filename, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            cols = line_str.split()
            if len(cols) >= 16:
                data_rows.append(cols)

    return data_rows


def find_default_input_file():
    """Dynamically resolves default input file candidates if none is given as an argument."""
    candidate_paths = [
        "input_NEAR",
        "inputs/input_NEAR",
        "input_EAR1",
        "inputs/input_EAR1",
        "input_EAR2",
        "inputs/input_EAR2"
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate activation counts using calculate_sacs.py engine with dynamic Pointwise F_ms correction.",
        epilog=USAGE_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-f", "--flux", default=None,
        help="Explicit flux file inside data/ directory (e.g. XXX-YYYYY). Overrides area-based resolving entirely."
    )
    parser.add_argument("input_file", nargs="?", default=None, help="Input table file")
    parser.add_argument(
        "flux_unit", choices=["cm2", "beam"],
        help="Neutron flux units option: 'cm2' = n/eV/cm2/7E12p (Fluence density), 'beam' = n/eV/7E12p (Integrated beam yield)"
    )
    parser.add_argument(
        "bpd", nargs="?", type=int, default=200,
        help="Bins per decade for external binning (default: 200)"
    )
    parser.add_argument(
        "extrap_mode", nargs="?", type=int, default=1, choices=[0, 1],
        help="Extrapolation mode for calculate_sacs.py: 1 = Enabled (default), 0 = Disabled"
    )

    if len(sys.argv) == 1 and sys.stdin.isatty():
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.input_file is None:
        resolved_input = find_default_input_file()
        if resolved_input is None:
            parser.print_help()
            print("\n[Error] No input file provided, and no standard default input file (e.g., 'input_NEAR' or 'inputs/input_NEAR') was found.")
            sys.exit(1)
        args.input_file = resolved_input

    if not os.path.exists(args.input_file):
        parser.print_help()
        print(f"\n[Error] Specified input file '{args.input_file}' was not found.")
        sys.exit(1)

    data_rows = load_raw_data_lines(args.input_file)

    # Check for area vs flux unit convention mismatches
    input_areas = [cols[0] for cols in data_rows]
    convention_warnings = check_flux_unit_convention(input_areas, args.flux_unit)
    
    # Output warnings to stderr immediately
    for w in convention_warnings:
        sys.stderr.write(f"\n{w}\n")

    resolved_spectrums = sorted(list(set(
        resolve_spectrum_file(cols[0], explicit_flux=args.flux) for cols in data_rows
    )))
    spec_str = ", ".join(resolved_spectrums)

    processed_results = []
    missing_fms_count = 0

    for cols in data_rows:
        res_tuple = calculate_counts_from_cols(
            cols, flux_unit=args.flux_unit, bpd=args.bpd, extrap_mode=args.extrap_mode, explicit_flux=args.flux
        )
        fms_found = res_tuple[-1]
        if not fms_found:
            missing_fms_count += 1
        processed_results.append(res_tuple)

    # Build Header metadata
    fms_status_msg = "Auto-detect per sample line (msdatafiles/fms_*.dat)"
    if missing_fms_count > 0:
        fms_status_msg += f" -- NOTE: File missing for {missing_fms_count} sample(s); F_ms = 1.0 assumed"

    flux_unit_label = "n/eV/cm2/7E12p (Fluence Density)" if args.flux_unit == "cm2" else "n/eV/7E12p (Integrated Beam Spot)"

    meta_headers = [
        f"# Input Table           : {args.input_file}",
        f"# Flux Units Mode       : {flux_unit_label}",
        f"# Resolved Spectrum(s)  : {spec_str}",
        f"# F_ms Correction Mode  : {fms_status_msg}",
        f"# Neutron Flux Shape    : {DEFAULT_INTERP_FLAG}",
        f"# SACS Engine           : Python (calculate_sacs.py)",
        f"# SACS Target Isotope   : {TARGET_ISO}",
        f"# SACS Sample Material  : {SAMPLE_MAT}",
        f"# SACS Bins per Decade  : {args.bpd}",
        f"# SACS Binning Mode     : {DEFAULT_BIN_MODE} (Master grid)",
        f"# SACS Extrapolation    : {args.extrap_mode}",
        f"# SACS Interp Flag      : {DEFAULT_INTERP_FLAG}"
    ]

    # Append convention warnings into header metadata if present
    for w in convention_warnings:
        meta_headers.append(f"# WARNING               : {w.replace('[CONVENTION WARNING] ', '')}")

    for line in meta_headers:
        print(line)

    # Expanded Header Definition including Column 16: A0/Asat (Fraction of saturation)
    header_fmt1 = f"#{'1':<10} {'2':<9} {'3':<4} {'4':<5} {'5':<10} {'6':<8} {'7':<8} {'8':<9} {'9':<10} {'10':<10} {'11':<10} {'12':<10} {'13':<14} {'14':<12} {'15':<14} {'16':<10} {'17':<11} {'18':<11} {'19':<10} {'20':<6}"
    header_fmt2 = f"#{'area':<9} {'sample':<9} {'line':<4} {'react':<5} {'fthick[mm]':<10} {'mass[g]':<8} {'diam[mm]':<8} {'thick[mm]':<9} {'SACS[b]':<10} {'f-SACS[b]':<10} {'ssf-SACS[b]':<10} {'ms-SACS[b]':<10} {'n_tot_sample':<14} {'A0[Bq/g]':<12} {'Asat/n[Bq/g/n]':<14} {'A0/Asat':<10} {'Cgam_calc':<11} {'Cgam_exp':<11} {'Ratio(C/E)':<10} {'BIF':<6}"
    divider_line = "#" * len(header_fmt2)

    table_headers = f"{header_fmt1}\n{header_fmt2}\n{divider_line}"
    print(table_headers)

    output_lines = meta_headers + [table_headers]

    for res_tuple in processed_results:
        area, sample_name, line_id, react, fthick_mm, mass_g, diam_mm, thick_mm, sacs_b, f_sacs, ssf_sacs, ms_sacs, total_neutrons, a0_per_g, asat_per_neutron, f_sat, c_calc, c_table, ratio, bif, spec_used, fms_found = res_tuple

        line_str = f"{area:<10} {sample_name:<9} {line_id:<4} {react:<5} {fthick_mm:<10.2f} {mass_g:<8.4f} {diam_mm:<8.1f} {thick_mm:<9.4f} {sacs_b:<10.3f} {f_sacs:<10.3f} {ssf_sacs:<10.3f} {ms_sacs:<10.3f} {total_neutrons:<14.3e} {a0_per_g:<12.3e} {asat_per_neutron:<14.3e} {f_sat:<10.4f} {c_calc:<11.3e} {c_table:<11.3e} {ratio:<10.3f} {bif:<6.4f}"
        print(line_str)
        output_lines.append(line_str)

    output_filename = "newtable"
    with open(output_filename, "w") as out_file:
        out_file.write("\n".join(output_lines) + "\n")

    print(f"\n# Summary table saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
