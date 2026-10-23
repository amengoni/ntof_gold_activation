# n_TOF Gold Activation Analysis (`ntof_gold_activation`)

Data analysis tools and Python scripts for processing gold sample activation measurements, energy spectrum binnerization, and specific activity at saturation ($A_{sat}$) calculations at the CERN **n_TOF** facility (NEAR station).

---

## 📁 Repository Structure

```text
.
├── scripts/
│   ├── binnerize.py         # Binnerizes energy vs. cross-section/flux data onto a target mesh
│   ├── calculate_counts.py   # Calculates saturation activity (Asat) from gamma peak count data
│   ├── calculate_sacs.py     # Calculates Asat and self-shielding factors from SACS tables
│   └── rebinnerize.py       # Handles secondary re-binning and mesh adaptation
├── data/                    # Input count data and reaction cross-section tables
└── README.md
```

---

## 🚀 Prerequisites & Installation

* **Python 3.8+**
* Required Python Libraries:
  * `numpy`
  * Standard built-in modules (`math`, `re`, `sys`, `os`)

To install `numpy`, run:
```bash
pip install numpy
```

---

## ⚙️ Usage & Execution

All calculations are executed directly using Python scripts located in the `scripts/` directory.

---

### 1. Gamma Peak Counts Analysis (`scripts/calculate_counts.py`)

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

### 2. SACS Cross-Section Processing (`scripts/calculate_sacs.py`)

Parses Spectrum-Averaged Cross Section (SACS) files (e.g., `sacs_SAM`), extracts self-shielding corrections, and calculates $A_{sat}$ across filter thickness steps (`fthick`).

#### Usage:
```bash
python scripts/calculate_sacs.py <path_to_sacs_table>
```

#### Example:
```bash
python scripts/calculate_sacs.py data/sacs_SAM
```

---

### 3. Continuous Data Binnerizer (`scripts/binnerize.py`)

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

### 4. Grid Re-binnerization (`scripts/rebinnerize.py`)

Adapts binned energy spectra to secondary target meshes or customized energy binning layouts.

#### Usage:
```bash
python scripts/rebinnerize.py <input_binned_file> <target_mesh_file>
```

---

## 📜 Physics & Calculation Methods

The cumulative gamma counts $C_{gam}$ detected during measurement interval $t_{count}$ after cooling time $t_{wait}$ following irradiation time $t_{irr}$ are given by:

$$C_{gam} = \frac{A_{sat} \cdot m \cdot I_{\gamma} \cdot \varepsilon \cdot \text{BIF}}{\lambda} \cdot \left(1 - e^{-\lambda t_{irr}}\right) \cdot e^{-\lambda t_{wait}} \cdot \left(1 - e^{-\lambda t_{count}}\right)$$

Where:
* $A_{sat}$: Specific Activity at Saturation ($\text{Bq/g}$)
* $m$: Sample mass ($\text{g}$)
* $I_{\gamma}$: Gamma-ray emission intensity
* $\varepsilon$: Detector efficiency at energy $E_{\gamma}$
* $\lambda$: Isotope decay constant ($\frac{\ln 2}{T_{1/2}}$)
* $\text{BIF}$: Beam Intermittency Factor

---

## 👥 Contact & Acknowledgments

* **Facility**: CERN n_TOF Collaboration (NEAR Station)
* **Repository**: [amengoni/ntof_gold_activation](https://github.com/amengoni/ntof_gold_activation)