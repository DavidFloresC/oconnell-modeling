# README for GitHub repository

"""
# Phenomenological Modeling of the O'Connell Effect in Eclipsing Binaries

This repository contains the Python code used for the analysis in:

**Flores Cabrera et al. (2025), A&A**  
"Multi-band analysis of the O’Connell effect in 14 eclipsing binaries" 

## Overview

The code implements a novel interactive modeling procedure that allows:
- **Interactive parameter tuning** for initial conditions using `matplotlib` sliders.
- **Composite model fitting** of eclipsing binary light curves using:
  - A truncated third-order **Fourier series**.
  - Three **super-Gaussian functions** to model eclipses and asymmetries.
- **Asymmetry quantification** through OER and LCA metrics.
- **Diagnostic plots** showing data, model components, and combined fits.

The core functionality is provided by the `sp_modeling` function in `modeling.py`.

## Getting Started

### Prerequisites
- Python 3.7+
- Packages: numpy, matplotlib, scipy, astropy, lmfit, symfit

Install dependencies:
```bash
pip install numpy matplotlib scipy astropy lmfit symfit
```

### Usage Example
```python
from modeling import sp_modeling

# Load your light curve data
phase = ...  # folded phase values between 0 and 1
mag = ...    # corresponding magnitudes or normalized flux
error = ...  # associated uncertainties

# Run the interactive model
sp_modeling(phase, mag, error)
```

## Repository Contents
- `modeling.py`: Main code with Fourier + super-Gaussian modeling tools.
- `Multi_band_analysis_of_the_O_Connell_effect_in_14_eclipsing_binaries__CLEAN_RESUBMISSION_.pdf`: Paper describing the model and results.
- Example notebook (optional, to be added): how to preprocess data and run the model.

## Code Highlights
- `interactive_super_gauss_tuner`: Real-time GUI for eclipse shape tuning.
- `fourier_series2`: Fast third-order Fourier series model.
- `super_gauss`: Super-Gaussian model function.
- `gg_fit9_factory`: Builds the full composite model.
- `integrate_mcmc`: Monte Carlo integration for OER/LCA metrics.

## Citing
If you use this code, please cite our paper:
```
Flores Cabrera et al. (2025), A&A, DOI: TBD
```

## License
This project is released under the MIT License.
"""
