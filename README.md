# n_TOF Gold Activation Analysis (`ntof_gold_activation`)

Data analysis tools and Python scripts for processing gold sample activation measurements, energy spectrum binnerization, isotope production yields, and specific activity at saturation ($A_{sat}$) calculations at the CERN **n_TOF** facility (NEAR, EAR1, and EAR2 stations).

---

## 📁 Repository Structure

```text
.
├── scripts/
│   ├── binnerize.py         # Binnerizes energy vs. cross-section/flux data onto a target mesh
│   ├── calculate_counts.py  # Calculates saturation activity (Asat) from gamma peak count data
│   ├── calculate_ms3.py     # 3D Monte Carlo multiple scattering correction driver (Fms calculation)
│   ├── calculate_prod.py    # Calculates production yields, fluence, and SACS metrics across samples
│   ├── calculate_sacs.py    # Calculates Asat and self-shielding/multiple scattering factors from SACS tables
│   └── rebinnerize.py      # Handles secondary re-binning and mesh adaptation
├── data/                    # Input count data, reaction cross-section tables, and neutron spectrum files
├── msdatafiles/             # Pointwise Monte Carlo Fms multiple scattering correction files
└── README.md
```

---

## 🚀 Prerequisites & Installation

* **Python 3.8+**
* Required Python Libraries:
  * `numpy`
  * `scipy`
  * `numba`
  * Standard built-in modules (`math`, `re`, `sys`, `os`, `argparse`, `subprocess`)

To install the required third-party libraries, run:
```bash
pip install numpy scipy numba
```

---

## ⚙️ Usage & Execution

All calculations are executed directly using Python scripts located in the `scripts/` directory.

---

### 1. Spectral-Averaged Cross Section (SACS) Calculator (`scripts/calculate_sacs.py`)

Parses Spectrum-Averaged Cross Section (SACS) inputs, extracts self-shielding corrections (`ssf-SACS`) and Monte Carlo multiple-scattering corrections (`ms-SACS`), and calculates integrated spectrum metrics across filter thickness steps (`fthick`).

```text
================================================================================
Spectral-Averaged Cross Section (SACS) Calculator - calculate_sacs.py
================================================================================

Usage:
  python3 calculate_sacs.py -ns nspectrum -samplexs react_xs sample_ntot \
                            [-samplemat MAT] [-samplemm MM | -sampleum UM] \
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
```

---

### 2. 3D Monte Carlo Multiple Scattering Driver (`scripts/calculate_ms3.py`)

Calculates 3D pointwise Monte Carlo multiple scattering correction factors ($F_{ms}$) across a unified native energy grid using Numba JIT-compiled tracking loops and adaptive batch error convergence.

```text
================================================================================
3D Monte Carlo Multiple Scattering Correction Driver - calculate_ms3.py
================================================================================

Usage:
    1. Run directly with a params.txt configuration file:
           $ python3 calculate_ms3.py params.txt

    2. Override specific parameter flags via CLI:
           $ python3 calculate_ms3.py --channel n4n --thickness-cm 0.0017 --diameter-cm 1.3

    3. View complete argument documentation:
           $ python3 calculate_ms3.py -h

Options:
  params_file          Path to configuration text file (default: params.txt if file exists)
  --scat-file          Path to elastic scattering cross section file
  --capt-file          Path to capture (n,g) cross section file
  --n2n-file           Path to (n,2n) cross section file
  --n4n-file           Path to (n,4n) cross section file
  --channel            Target reaction channel (ng, n2n, n4n, 41)
  --target-err         Target relative precision (e.g., 0.005 for 0.5%)
  --min-histories      Minimum MC histories per energy point
  --max-histories      Maximum MC histories cutoff
  --batch-size         Batch size per convergence loop
  --diameter-cm        Sample diameter in cm
  --thickness-cm       Sample thickness in cm
  --density-g-cm3      Sample density in g/cm3
  --molar-mass         Sample molar mass in g/mol

Examples:
  python3 calculate_ms3.py
  python3 calculate_ms3.py params.txt
  python3 calculate_ms3.py --channel n4n --thickness-cm 0.0017
```

---

### 3. Production Yield & Fluence Calculator (`scripts/calculate_prod.py`)

Calculates isotope production yields (e.g., $^{10}\text{Be}$, $^{26}\text{Al}$), total integrated fluence, and SACS metrics (`SACS`, `f-SACS`, `ssf-SACS`, `ms-SACS`) across multiple samples by executing `calculate_sacs.py` via subprocess calls. Checks `msdatafiles/` for matching $F_{ms}$ data files.

```text
==============================================================================
 Usage Block - calculate_prod.py
==============================================================================
Description:
  Calculates isotope production (e.g., Be10, Al26) and SACS metrics 
  (SACS, f-SACS, ssf-SACS, ms-SACS) by delegating spectral integration to 
  calculate_sacs.py via subprocess calls. Checks for Fms data in msdatafiles/.

Usage:
  python3 scripts/calculate_prod.py [input_file] [--no-ms]

Arguments:
  input_file          : Path to input parameters file (e.g., inputs/input_NEAR_BurialDating_prod)
  --no-ms             : Disable multiple scattering correction (forces Fms = 1.0)

Examples:
  python3 scripts/calculate_prod.py inputs/input_NEAR_BurialDating_prod
  python3 scripts/calculate_prod.py inputs/input_NEAR_BurialDating_prod --no-ms
==============================================================================
```

---

### 4. Gamma Peak Counts Analysis (`scripts/calculate_counts.py`)

Calculates the specific activity at saturation ($A_{sat}$ in $\text{Bq/g}$ and normalized $\text{Bq/g/p}$) directly from gamma-ray peak count measurements and experimental irradiation, cooling, and counting times.

#### Usage:
```bash
python scripts/calculate_counts.py <path_to_count_data_file>
```

#### Example:
```bash
python scripts/calculate_counts.py data/input_NEAR_N14
```

---

### 5. Continuous Data Binnerizer (`scripts/binnerize.py`)

Binnerizes pointwise energy vs. cross-section or flux data onto a target energy grid using logarithmic trapezoidal quadrature to preserve native resonance structures without introducing decade-scale interpolation errors.

#### Help / Usage Documentation:
Executing without parameters in an interactive terminal or with `-h` displays full usage documentation:
```bash
python scripts/binnerize.py -h
```

#### Execution with Piped Configuration File:
```bash
python scripts/binnerize.py < config_file.txt
```

---

### 6. Grid Re-binnerization (`scripts/rebinnerize.py`)

Adapts binned energy spectra to secondary target meshes or customized energy binning layouts.

#### Usage:
```bash
python scripts/rebinnerize.py <input_binned_file> <target_mesh_file>
```

---

## 📜 Physics & Calculation Methods

### Gamma Count Saturation Activity
The cumulative gamma counts $C_{gam}$ detected during measurement interval $t_{count}$ after cooling time $t_{wait}$ following irradiation time $t_{irr}$ are given by:

$$C_{gam} = \frac{A_{sat} \cdot m \cdot I_{\gamma} \cdot \varepsilon \cdot \text{BIF}}{\lambda} \cdot \left(1 - e^{-\lambda t_{irr}}\right) \cdot e^{-\lambda t_{wait}} \cdot \left(1 - e^{-\lambda t_{count}}\right)$$

Where:
* $A_{sat}$: Specific Activity at Saturation ($\text{Bq/g}$)
* $m$: Sample mass ($\text{g}$)
* $I_{\gamma}$: Gamma-ray emission intensity
* $\varepsilon$: Detector efficiency at energy $E_{\gamma}$
* $\lambda$: Isotope decay constant ($\frac{\ln 2}{T_{1/2}}$)
* $\text{BIF}$: Beam Intermittency Factor

### Flux & Fluence Units Treatment
* **`NEAR` Spectra**: Flux units are expressed in differential fluence per unit area ($[n/\text{cm}^2/\text{eV}/p]$). Total fluence is calculated directly as:
  $$\Phi = n_{n} \cdot n_{\text{pulses}} \cdot \text{BIF}$$
* **`EAR1` / `EAR2` Spectra**: Flux units are beam-integrated over the full beam profile ($[n/\text{eV}/p]$). Fluence per unit area seen by the sample is derived by dividing by sample area $A = \pi r^2$:
  $$\Phi = \left(\frac{n_{n}}{A}\right) \cdot n_{\text{pulses}} \cdot \text{BIF}$$

---

## 👥 Contact & Acknowledgments

* **Facility**: CERN n_TOF Collaboration (NEAR, EAR1, and EAR2 Stations)
* **Repository**: [amengoni/ntof_gold_activation](https://github.com/amengoni/ntof_gold_activation)