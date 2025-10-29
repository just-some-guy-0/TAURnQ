# TAURnQ Repository Breakdown

This repository documents the full computational workflow developed and applied to analyse relaxation dynamics in single-molecule magnets using a distribution-based approach. It contains all scripts, datasets, and intermediate outputs necessary to reproduce the results reported in the accompanying research report.

## **`archive/`**

This folder contains early developmental work and validation of the algorithm implemented. It includes four subdirectories:

- `algorithm_testing/` – prototype scripts for fitting the total relaxation-rate model to mean-only experimental data and compiling temperature-dependent Monte Carlo parameter estimates.
- `distribution_testing/` – tests to assess whether, under the assumption that parameters are normally distributed, we can recover parameter estimates from experimental relaxation rate data.
- `mcm_testing/` – contains scripts, CSVs, and plots validating the Monte Carlo algorithm; these plots verify that the simulated relaxation rate distributions show similar behaviour to the log-normal approximation to the Fuoss–Kirkwood distribution.
- `sef_testing/` – a preliminary attempt to approximate the stretched exponential function via a Prony-series expansion; this direction was later abandoned due to a conceptual error in understanding the role of the stretched exponential function in experimental distributions. 

These scripts are largely superseded by newer implementations in `src/dev_area` but are still provided for book-keeping.

## **`plots/`**

Contains Python scripts used to generate figures for both the research report and presentation.

Each script (e.g., `figure1.py`, `slide7.py`) reproduces its namesake figure used in the report or presentation respectively. 

## **`src/`**

This directory holds the main implementation of the TAURnQ algorithm and results for all the molecules studies. It is subdivided into two sub-directories:

### **`a. dev_area/` — Algorithm development and execution**

Implements the TAURnQ algorithm in four separate phases. Since, this work did not develop a generalised tool, hard-coded scripts were needed for each molecule. The hard-coded scripts are provided in a subfolder for each moleucle within each the respective folder for each phase; these were also included to show easy reproducibility of the results from this work. 

#### **`phase1/` — Domain Identification**

- `mean_only_fit.py`: performs non-linear least squares fitting of mean relaxation times to obtain preliminary parameters (A, Ueff, R, n, Q).
- `frac_from_means.py`: calculates fractional contributions of Orbach, Raman, and QTM mechanisms, aiding in identifying dominant temperature ranges for each process. 

#### **`phase2/` — Domain Identification**

- `orbach.py`, `raman.py`, `qtm.py`: apply the normal assumption on parameters and use the expression derived in the report to extract parameter distributions under each relaxation process, generating statistical priors for the next phase.

#### **`phase3/` — Domain Identification**

`montecarlo.py`: samples parameter distributions at each temperature, reproducing experimental log-normal approximation on the Fuoss–Kirkwood distributions to refine mean and variance estimates. 

#### **`phase4/` — Domain Identification**

`global_fit.py`: combines temperature-dependent Monte Carlo outputs into global parameter distributions using domain-specific weighting to ensure physical consistency across temperature ranges.


### **`b. main/ ` — Results and Data Representation**

Each subfolder here corresponds to a different single-molecule magnet analysed and contains the following files:
1.	An experimental input (.tsv) with temperature, mean relaxation time, and α values.
2.	A plot of mean-only relaxation vs temperature.
3.	A plot of the fractional contributions for each relaxation process to the total rate at each temperature.
4.	(.csv) file for the fractional contributions for each process at each temperature. 
5.	(.tsv) file with per-temperature Monte-Carlo generated parameter estimates.
6.	(.csv) file with globally fit parameter estimates from each run. The bottom most estimates show the most recent trial and were used for the plot in (7). 
7.	Plot of mean-only and predicted relaxation rate from distribution-based method across experimental temperature range. 

Together, these files document the intermediate and final outputs for the TAURnQ algorithm and provide illustrations of some results for further analysis in the associated report. 