# FORCE: Factorized Observables for Regressing Conditional Expectation

This repository contains the code presented in the paper **Anomaly Detection in Collider Physics via Factorized Observables**.
The data and results are stored in [Dropbox](https://www.dropbox.com/scl/fo/rtbj10e7nvokqjj0d2i0v/h?rlkey=phk4cg4zt1c5y4p0927rn5p2c&dl=0) due to their size.

## Python
This directory contains notebooks and helper functions for the analyses presented in the paper. Most notably, ```FORCE.ipynb``` contains an example for data loading, model creation, and model application, as well as a signal sensitivity sweep. For the ```processing.py``` and ```observables.py``` scripts, the ```events_anomalydetection_v2.h5``` can be pulled from the [LHCO R&D Zenodo Link](https://zenodo.org/records/6466204).

## Gauss
This directory contains all the code for the toy gaussian analysis presented in the paper.

## Figures
This directory contains all the figures in the paper. The code that generates these figures are in ```plots.ipynb```, ```mutual_info.ipynb```, and ```gauss_test.ipynb```, while the data to generate these figures are on the Dropbox.