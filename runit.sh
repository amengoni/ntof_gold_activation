#!/usr/bin/env bash

# ===============================================================================
# SAMPLE-SPECIFIC MULTIPLE SCATTERING & ACTIVATION ANALYSIS WORKFLOW
# ===============================================================================

set -e # Exit immediately if a command fails

TEMP_DIR="tempfiles"
SCRIPTS_DIR="scripts"
INPUTS_DIR="inputs"

show_usage() {
    cat << EOF
===============================================================================
USAGE GUIDE: $(basename "$0")
===============================================================================

DESCRIPTION:
  Executes a full neutron activation analysis workflow for foil targets.
  Automates 3D Numba Monte Carlo multiple scattering simulations (scripts/calculate_ms3.py)
  and integrates activation counts (scripts/calculate_counts3.py).

SYNTAX:
  ./$(basename "$0") <input_file> [flux_variant] [bpd] [--no-ms]
  ./$(basename "$0") -h | --help

POSITIONAL ARGUMENTS:
  input_file    Filename or path to input table file in inputs/ (e.g., input_NEAR_BurialDating)
  flux_variant  Neutron spectrum variant tag in data/ (default: MCecc)
  bpd           Bins per decade for SACS integration (default: 200)

FLAGS:
  --no-ms       Disable Multiple Scattering (skips MC simulation, F_ms = 1.0)
  -h, --help    Display this usage guide and exit

EXAMPLES:
  ./$(basename "$0") input_NEAR_BurialDating MCecc 200          # Full 3D MC Workflow
  ./$(basename "$0") inputs/input_NEAR_N14 MCecc 200 --no-ms   # Fast Run (No MC)
===============================================================================
EOF
}

# Print usage guide and exit if no arguments are provided or if help flag is passed
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
    exit 0
fi

# Parse parameters and check for --no-ms flag
USE_MS=true
ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--no-ms" ]; then
        USE_MS=false
    else
        ARGS+=("$arg")
    fi
done

RAW_INPUT_FILE="${ARGS[0]}"
FLUX_VARIANT="${ARGS[1]:-MCecc}"
BPD="${ARGS[2]:-200}"

# Resolve input file path (automatically prepend inputs/ if not already specified)
if [[ "${RAW_INPUT_FILE}" == *"/"* ]]; then
    INPUT_FILE="${RAW_INPUT_FILE}"
else
    INPUT_FILE="${INPUTS_DIR}/${RAW_INPUT_FILE}"
fi

MC_SCRIPT="${SCRIPTS_DIR}/calculate_ms3.py"
COUNTS_SCRIPT="${SCRIPTS_DIR}/calculate_counts3.py"

echo "================================================================="
echo " STARTING ACTIVATION ANALYSIS WORKFLOW"
echo "================================================================="
echo "Input Table     : ${INPUT_FILE}"
echo "Flux Variant    : ${FLUX_VARIANT}"
echo "Bins Per Decade : ${BPD}"
echo "MS Correction   : ${USE_MS}"
echo "-----------------------------------------------------------------"

if [ ! -f "${INPUT_FILE}" ]; then
    echo "[ERROR] Input file '${INPUT_FILE}' not found!"
    exit 1
fi

if [ ! -f "${COUNTS_SCRIPT}" ]; then
    echo "[ERROR] Count calculation script '${COUNTS_SCRIPT}' not found!"
    exit 1
fi

# ===============================================================================
# MULTIPLE SCATTERING SIMULATION PASS (If --no-ms is NOT passed)
# ===============================================================================
if [ "${USE_MS}" = true ]; then
    mkdir -p "${TEMP_DIR}"

    if [ ! -f "${MC_SCRIPT}" ]; then
        echo "[ERROR] Monte Carlo script '${MC_SCRIPT}' not found!"
        exit 1
    fi

    # Extract valid non-header data rows
    DATA_ROWS=$(grep -v '^\s*#' "${INPUT_FILE}" | grep -v '^\s*$')

    # Construct unique geometry + reaction keys (<DIAM_MM>_<MASS_G>_<REACT>)
    UNIQUE_RUNS=$(echo "${DATA_ROWS}" | awk '{
        line = $7;
        if (line == "41") {
            react = "n4n";
        } else if (line == "21" || line == "22") {
            react = "n2n";
        } else {
            react = "ng";
        }
        print $5 "_" $4 "_" react;
    }' | sort -u)

    for RUN_KEY in ${UNIQUE_RUNS}; do
        DIAM_MM=$(echo "${RUN_KEY}" | cut -d'_' -f1)
        MASS_G=$(echo "${RUN_KEY}" | cut -d'_' -f2)
        REACT=$(echo "${RUN_KEY}" | cut -d'_' -f3)

        # Reformat diameter and mass for strict filename alignment with calculate_counts3.py
        DIAM_STR=$(awk "BEGIN {printf \"%.1f\", ${DIAM_MM}}")
        MASS_STR=$(awk "BEGIN {printf \"%.4f\", ${MASS_G}}")

        THICK_CM=$(awk "BEGIN {
            rho = 19.32;
            radius_cm = (${DIAM_MM} / 10.0) / 2.0;
            area_cm2 = 3.141592653589793 * (radius_cm ^ 2);
            printf \"%.6f\", ${MASS_G} / (rho * area_cm2);
        }")
        
        DIAM_CM=$(awk "BEGIN {printf \"%.6f\", ${DIAM_MM} / 10.0}")

        GEOM_TAG="d${DIAM_STR}_m${MASS_STR}_${REACT}"
        PARAM_FILE="params.txt"
        FMS_OUTPUT="${TEMP_DIR}/fms_${GEOM_TAG}.dat"

        # Check if pre-computed MS output file already exists in tempfiles/
        if [ -f "${FMS_OUTPUT}" ]; then
            echo "    [SKIP] Found existing '${FMS_OUTPUT}' for mass ${MASS_STR} g, reaction ${REACT}."
        else
            echo "    [MC RUN] Running 3D Monte Carlo for mass=${MASS_STR} g, diam=${DIAM_STR} mm (${REACT})..."

            # Write params.txt explicitly matching calculate_ms3.py key format
            cat << EOF > "${PARAM_FILE}"
molar_mass     = 196.9665
density_g_cm3  = 19.32
thickness_cm   = ${THICK_CM}
diameter_cm    = ${DIAM_CM}
scat_file      = data/Au197.nel
capt_file      = data/Au197.ng
n2n_file       = data/Au197.n2n
n4n_file       = data/Au197.n4n
channel        = ${REACT}
target_err     = 0.005
min_histories  = 10000
max_histories  = 100000
batch_size     = 10000
random_seed    = 42
EOF

            # Execute calculate_ms3.py (reads params.txt from current directory)
            python3 "${MC_SCRIPT}" "${PARAM_FILE}"

            # Relocate generated output to target tempfiles/ directory
            GENERATED_OUT="ms_results_${REACT}.dat"
            if [ -f "${GENERATED_OUT}" ]; then
                mv "${GENERATED_OUT}" "${FMS_OUTPUT}"
            else
                echo "[ERROR] Expected MC output '${GENERATED_OUT}' was not generated!"
                exit 1
            fi

            rm -f "${PARAM_FILE}"
        fi
    done
else
    echo ">>> [--no-ms FLAG DETECTED] Skipping Monte Carlo simulations."
fi

# ===============================================================================
# FINAL TABLE COMPUTATION PASS
# ===============================================================================
echo "-----------------------------------------------------------------"
echo ">>> Running activation calculation engine across all samples..."
python3 "${COUNTS_SCRIPT}" "${INPUT_FILE}" "${FLUX_VARIANT}" "${BPD}"

echo "================================================================="
echo " WORKFLOW COMPLETED SUCCESSFULLY"
echo " Final aggregated results saved in 'newtable'"
echo "================================================================="
