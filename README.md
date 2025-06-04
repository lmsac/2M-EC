# 2M-EC Platform Study Documentation

This EC (endometrial cancer) repository includes data preprocessing, model construction/validation, and 2M-EC platform deployment. The repository also incorporates our in-house developed tools for molecular matching and ROMA data collection.

# Repository Structure

### 2M-EC_platform
Stores trained models and data processors. Deployment programs. Includes 4 test samples for platform test.

### Gridsearch
Used to optimize analyte-algorithm combinations. The model training methodology referenced literature（Osipov, A., et al. The Molecular Twin artificial-intelligence platform integrates multi-omic data to predict outcomes for pancreatic adenocarcinoma patients. Nat Cancer 5, 299-314 (2024).

### modeling
Training, saving and validation of mass spectrometry models and fusion models. 

### ML (machine learning)
New sample data requires spectral preprocessing followed by binning. Here is a demonstration of the binning format.

### Identification
Protein molecular weight calculation and matching.

### ROMA.py
Web-based clinical cohort input and result collection for ROMA.

### Maldi spectrum preprocessing.R 
Prior to new sample prediction, spectrum data must undergo preprocessing and spectral binning, with the binning methodology following published literature protocols（Weis, C., et al. Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning. Nat Med 28, 164-174 (2022).）

# Open access to data
Direct decompression provides access to all the multi-omics maldi data (plasma-PM.zip, plasma-PP.zip, cervical.zip, and uterine.zip) used in the study, along with downloadable paired clinical metadata (Metadata_CI.csv) for the 531 patients. The processed maldi data matrices are also available in Appendix Table Data S2: Prepared multi-omics MS dataset (.xlsx).

### Dataset of EC

Database of 1160 multi-omics raw profiles from uterine secretions, cervical secretions and plasma with MALDI-TOF Mass spectrometry, processed dataset and clinical metadata

For each site, the data consists of MALDI-TOF mass spectra in the form of `.txt` files and a aggregated meta-data file (Metadata_CI.csv) with clinical information to model and align.


The EC folder structure obtained after download is the following:
```
EC
├── Cervical 
│   ├── raw_192CM
│   ├── M1/2
│   └── TS1
│
├── Uterine
│   ├── raw_246UM
│   ├── M2
│   ├── TS2
│   ├── NSMP
│   └── p53
│
├── Plasma
│   ├── raw_361PM
│   ├── raw_361PP
│   ├── M1/2 (PM)
│   ├── M1/2 (PP)
│   └── TS1
│
├── Metadata_CI.csv
│
├──README.md


```

### Sites where MALDI-TOF MS profiles were collected

  University of Fudan, China

For details on the dataset extraction and preprocessing, please refer to the Methods section in the article corresponding to the publication https://www.nature.com/articles/s41591-021-01619-9. 

### Conversion to Dataset

Raw MALDI-TOF MS spectra were converted to .txt format using flexAnalysis software (Bruker, Germany). Spectral preprocessing was performed with R packages MALDIquant and MALDIquantForeign, implementing: 1) square root transformation for variance stabilization, 2) Savitzky-Golay filtering (15-point window) for smoothing, and 3) SNIP algorithm (20 iterations) for baseline correction.
Following established protocols, we conducted mass-to-charge ratio (m/z) binning with sample-specific resolutions: 9,000 evenly distributed bins for uterine/cervical metabolic profiles (100-1,000 Da), 900 evenly distributed bins for plasma metabolic profiles (100-1,000 Da) and 675 evenly distributed bins for plasma peptidomic profiles (3-30 kDa). The binning approach was adjusted based on the mass spectrometer's resolution. All processed data retained intensity values normalized to total intensity value of the spectrum.

We recommend using Python package for MALDI-TOF MS preprocessing and machine learning analysis, `maldi-learn` (https://github.com/BorgwardtLab/maldi-learn), to load and analyse EC data.

The github package comes with an elaborate `README.md` file, which gives details on installation and usage examples.
The code tools for data processing can be found at https://github.com/lmsac/2M-EC.git.


### A note on structure of EC Dataset

We implemented a stratified cohort design comprising a modeling cohort and an external test cohort. Modeling cohort (n=436) was partitioned into three M sub-cohorts, M1 (cervical and plasma, n=314), M2 (cervical, plasma and uterine, n=436), and M3 (uterine annotated with subtyping, n=77). The raw spectra processed dataset is uploaded in a tabular form as a stratified cohort. The raw data for p53 and nsmp were obtained from Uterine and the processed dataset was uploaded. The raw data files can be aligned according to Samples.

The first column, "Samples," represents the clinical identification numbers of the patients, and "target" indicates the patient grouping, where 1 denotes the endometrial cancer group and 0 denotes the non-endometrial cancer group. Additionally, for the "p53" and "nsmp" groupings, 1 represents the specific molecular subtype group of the cancer, while 0 represents the other three coded molecular subtype groups combined.


