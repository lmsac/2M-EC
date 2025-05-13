### 2M-EC Platform Study Documentation

This EC (endometrial cancer) repository includes data preprocessing, model construction/validation, and 2M-EC platform deployment. The repository also incorporates our in-house developed tools for molecular matching and ROMA data collection.


### Repository Structure

# 2M-EC_platform
Stores trained models and data processors. Deployment programs. Includes 4 test samples for platform validation.

# Gridsearch
Used to optimize analyte-algorithm combinations. The model training methodology referenced literature（Osipov, A., et al. The Molecular Twin artificial-intelligence platform integrates multi-omic data to predict outcomes for pancreatic adenocarcinoma patients. Nat Cancer 5, 299-314 (2024).

# modeling
Training, saving and validation of mass spectrometry models and fusion models. 

# ML (machine learning)
New sample data requires spectral preprocessing followed by binning.

# Identification
Protein molecular weight calculation and matching.

# ROMA.py
Web-based clinical cohort input and result collection for ROMA.

# Maldi spectrum preprocessing.R 
Prior to new sample prediction, spectrum data must undergo preprocessing and spectral binning, with the binning methodology following published literature protocols（Weis, C., et al. Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning. Nat Med 28, 164-174 (2022).）
