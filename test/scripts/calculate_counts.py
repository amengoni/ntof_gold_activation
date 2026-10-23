#!/usr/bin/env python3

import sys
import os
import argparse
import subprocess
import numpy as np

# Ensure helper modules or scripts in candidate directories can be imported/resolved[cite: 3, 4]
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Physical Constants for Gold (197Au) Reactions[cite: 3]
M_AU = 196.966569        # Molar mass [g/mol][cite: 3]
RHO_AU = 19.32           # Density of Gold [g/cm3][cite: 3]
NA = 6.02214076e23       # Avogadro constant [atoms/mol][cite: 3]
A_ISO = 1.0              # Monoisotopic 197Au[cite: 3]

# Reaction Product Half-Lives [seconds][cite: 3]
T_HALF_198AU = 232856.6  # 198Au from (n,g)  (~2.6951 days)[cite: 3]
T_HALF_196AU = 532790.4  # 196Au from (n,2n) (~6.1665 days)[cite: 3]
T_HALF_194AU = 141120.0  # 194Au from (n,4n) (~39.2 hours)[cite: 3]

# Standard n_TOF pulse intensity assumed in calculate_sacs.py spectrum files[cite: 3]
PROTONS_PER_STANDARD_PULSE = 7.0e12

# Hard-coded SACS parameters passed downstream to calculate_sacs.py[cite: 3]
TARGET_ISO = "Au197"
SAMPLE_MAT = "Au"
DEFAULT_BIN_MODE = "0"
DEFAULT_INTERP_FLAG = "1/E"  # Hard-coded to 1/E interpolation[cite: 3]

USAGE_TEXT = """
Usage Examples:
  python3 scripts/calculate_counts.py input_BurialDating cm2
  python3 scripts/calculate_counts.py input_EAR1CC_BurialDating beam 200 1
  python3 scripts/calculate_counts.py -f XXX-YYYYY input_EAR1CC_BurialDating cm2
"""


def resolve_helper_script(script_name):
    """
    Searches for helper scripts in current directory, same directory as this script,
    or standard research subfolders[cite: 4].
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
    Resolves neutron spectrum filename inside data/ based strictly on area[cite: 3].
    Defaults to 'NEAR-MCecc' if no area or explicit flux is provided[cite: 3].
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

    target_path = os.path.join("data", spectrum_file)

    if not os.path.exists(target_path):
        sys.stderr.write(
            f"\n[FATAL ERROR] Spectrum file error:\n"
            f"  Expected file path : '{target_path}'\n"
            f"  Status            : FILE NOT FOUND\n"
            f"Execution terminated.\n\n"
        )
        sys.exit(1)

    return spectrum_file


def get_fms_filename(mass_g, diam_mm, line_id):
    """
    Constructs the expected F_ms Monte Carlo filename inside msdatafiles/ based on 
    the geometry and reaction type (ng, n2n, n4n)[cite: 3].
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
    """Returns decay constant lambda [1/s] according to product isotope[cite: 3]."""
    line_str = str(line_id).strip()
    if line_str == '41':
        t_half = T_HALF_194AU
    elif line_str in ['21', '22']:
        t_half = T_HALF_196AU
    else:
        t_half = T_HALF_198AU
    return np.log(2.0) / t_half


def get_xs_filenames(line_id):
    """Resolves cross section filenames for reaction and total cross section in data/[cite: 4]."""
    line_str = str(line_id).strip()
    if line_str == '41':
        return "au197n4n.b81", "au197ntot.b81"
    elif line_str in ['21', '22']:
        return "au197n2n.b81", "au197ntot.b81"
    else:
        return "au197ng.b81", "au197ntot.b81"


def calculate_counts_from_cols(cols, flux_unit="cm2", bpd=200, extrap_mode=1, explicit_flux=None):
    """
    Computes total calculated counts using zero-indexed positional columns
    via calculate_sacs.py[cite: 3]. Automatically detects and applies pointwise 
    Monte Carlo F_ms correction if the corresponding file exists[cite: 3].
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
    bif           = float(cols[15])  # Beam Interception / Correction Factor[cite: 3]

    lam = get_decay_constant(line_id)

    # Resolve specific F_ms file generated inside msdatafiles/[cite: 3]
    fms_file = get_fms_filename(mass_g, diam_mm, line_id)
    fms_found = os.path.exists(fms_file)

    # 1. Target nuclei (NT) from mass[cite: 3]
    NT = (mass_g / M_AU) * NA * A_ISO

    # 2. Sample Area [cm2] and Areal density [atoms/barn] from mass & diameter[cite: 3]
    diam_cm = diam_mm * 0.1
    area_cm2 = np.pi * (0.5 * diam_cm)**2
    atoms_per_barn = (NT / area_cm2) * 1e-24

    # Derived Thickness Check (in mm)[cite: 3]
    thick_calc_cm = mass_g / (RHO_AU * area_cm2)
    thick_calc_mm = thick_calc_cm * 10.0

    if thick_mm > 0.0:
        rel_diff = abs(thick_calc_mm - thick_mm) / thick_mm
        if rel_diff > 0.01:
            sys.stderr.write(
                f"[WARNING] Sample '{sample_name}' (line {line_id}): "
                f"Thickness mismatch > 1% ({rel_diff * 100.0:.2f}%)! "
                f"Derived from mass & diam: {thick_calc_mm:.4f} mm, "
                f"Input table thick_mm: {thick_calc_mm:.4f} mm\n"
            )

    spectrum_file = resolve_spectrum_file(area, explicit_flux=explicit_flux)
    rxn_xs_file, ntot_xs_file = get_xs_filenames(line_id)

    # Robust path resolution for calculate_sacs.py[cite: 3, 4]
    sacs_script_path = resolve_helper_script("calculate_sacs.py")

    # 3. Invoke calculate_sacs.py using standard flag interface[cite: 3, 4]
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
        # Pass macroscopic filter option if filter thickness > 0[cite: 4]
        cmd.extend(["-filter", "b4c.mac", str(fthick_mm)])

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

    # 4. Parse output summary line[cite: 3]
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
        f_nn_pulse       = float(tokens[4])     # f-nn/pp[cite: 3]
        sacs_b           = float(tokens[5])     # Unattenuated SACS [b][cite: 3]
        f_sacs_b         = float(tokens[6])     # Filtered SACS [b][cite: 3]
        ssf_sacs_b       = float(tokens[8])     # Self-shielded SACS [b][cite: 3]
        ms_sacs_b        = float(tokens[9])     # Multiple-scattering corrected SACS [b][cite: 3]
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"Failed to parse tokens from: '{line_out}'") from e

    # Effective cross section in cm2 (using the fully corrected ms_sacs_b)[cite: 3]
    sigma_cm2 = ms_sacs_b * 1e-24

    # Equivalent standard 7e12 proton pulses[cite: 3]
    equivalent_pulses = total_protons / PROTONS_PER_STANDARD_PULSE

    # Unambiguous Calculation of Total Neutrons & Fluence Density based on explicitly passed flux_unit[cite: 3]
    if flux_unit == "beam":
        # f_nn_pulse is already total integrated neutrons over beam spot [n/eV/7E12p][cite: 3]
        total_neutrons_sample = f_nn_pulse * equivalent_pulses * bif
        phi_tot = total_neutrons_sample / area_cm2
    else:
        # f_nn_pulse is fluence density [n/eV/cm2/7E12p][cite: 3]
        phi_tot = f_nn_pulse * equivalent_pulses * bif
        total_neutrons_sample = phi_tot * area_cm2

    # 5. Activation Calculation using channel-specific decay constant[cite: 3]
    R = (NT * phi_tot * sigma_cm2) / tirr_s
    A0 = R * (1.0 - np.exp(-lam * tirr_s))  # Activity at End of Irradiation [Bq][cite: 3]
    
    # Specific Activity at End of Irradiation [Bq/g][cite: 3]
    a0_per_g = A0 / mass_g if mass_g > 0 else 0.0

    # Total calculated gamma counts[cite: 3]
    counts_calc = (A0 / lam) * np.exp(-lam * twait_s) * (1.0 - np.exp(-lam * tcount_s)) * epsi * Igam

    display_react = "n4n" if line_id == '41' else ("n2n" if line_id in ['21', '22'] else "ng")
    return area, sample_name, line_id, display_react, fthick_mm, mass_g, diam_mm, thick_mm, sacs_b, f_sacs_b, ssf_sacs_b, ms_sacs_b, total_neutrons_sample, a0_per_g, counts_calc, c_table, ratio_val(counts_calc, c_table), bif, spectrum_file, fms_found


def ratio_val(c_calc, c_table):
    return c_calc / c_table if c_table > 0 else 0.0


def load_raw_data_lines(filename):
    """Loads data lines ignoring comments (#) or blank lines[cite: 3]."""
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
    """Dynamically resolves default input file candidates if none is given as an argument[cite: 3]."""
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

    # Display usage if executed interactively without arguments
    if len(sys.argv) == 1 and sys.stdin.isatty():
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Dynamic default input resolution if omitted[cite: 3]
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

    resolved_spectrums = sorted(list(set(
        resolve_spectrum_file(cols[0], explicit_flux=args.flux) for cols in data_rows
    )))
    spec_str = ", ".join(resolved_spectrums)

    # Process all sample rows first to assess F_ms status across the table[cite: 3]
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

    # Build Header metadata[cite: 3]
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

    for line in meta_headers:
        print(line)

    # Tight Header Definition with exactly 1 blank space between heading tags[cite: 3]
    header_fmt1 = f"#{'1':<10} {'2':<9} {'3':<4} {'4':<5} {'5':<10} {'6':<8} {'7':<8} {'8':<9} {'9':<10} {'10':<10} {'11':<10} {'12':<10} {'13':<14} {'14':<12} {'15':<11} {'16':<11} {'17':<10} {'18':<6}"
    header_fmt2 = f"#{'area':<9} {'sample':<9} {'line':<4} {'react':<5} {'fthick[mm]':<10} {'mass[g]':<8} {'diam[mm]':<8} {'thick[mm]':<9} {'SACS[b]':<10} {'f-SACS[b]':<10} {'ssf-SACS[b]':<10} {'ms-SACS[b]':<10} {'n_tot_sample':<14} {'A0[Bq/g]':<12} {'Cgam_calc':<11} {'Cgam_exp':<11} {'Ratio(C/E)':<10} {'BIF':<6}"
    divider_line = "#" * len(header_fmt2)

    table_headers = f"{header_fmt1}\n{header_fmt2}\n{divider_line}"
    print(table_headers)

    output_lines = meta_headers + [table_headers]

    for res_tuple in processed_results:
        area, sample_name, line_id, react, fthick_mm, mass_g, diam_mm, thick_mm, sacs_b, f_sacs, ssf_sacs, ms_sacs, total_neutrons, a0_per_g, c_calc, c_table, ratio, bif, spec_used, fms_found = res_tuple

        # Data formatting matching exact field widths with single-space separators[cite: 3]
        line_str = f"{area:<10} {sample_name:<9} {line_id:<4} {react:<5} {fthick_mm:<10.2f} {mass_g:<8.4f} {diam_mm:<8.1f} {thick_mm:<9.4f} {sacs_b:<10.3f} {f_sacs:<10.3f} {ssf_sacs:<10.3f} {ms_sacs:<10.3f} {total_neutrons:<14.3e} {a0_per_g:<12.3e} {c_calc:<11.3e} {c_table:<11.3e} {ratio:<10.3f} {bif:<6.4f}"
        print(line_str)
        output_lines.append(line_str)

    output_filename = "newtable"
    with open(output_filename, "w") as out_file:
        out_file.write("\n".join(output_lines) + "\n")

    print(f"\n# Summary table saved to '{output_filename}'.")


if __name__ == "__main__":
    main()
