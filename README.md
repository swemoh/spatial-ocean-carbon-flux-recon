# Modeling Ocean Surface CO₂ Flux with Deep Learning

This repository contains the code for my master's thesis on **Applying deep learning architectures to estimate ocean surface carbon dioxide flux based on spatiotemporal data**.  
It provides tools for **data preprocessing, model training, and evaluation** on the FOCI-MOPS v1 ocean simulation dataset.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#-usage)
- [Acknowledgements](#-acknowledgements)

---

## Overview

- **Problem**: Accurate modeling of air-sea CO₂ fluxes is crucial for understanding the global carbon cycle, but traditional methods face challenges in capturing spatial and temporal variability.
- **Approach**: This project applies various **deep learning architectures** (MLP, U-Net, U-Next and Swin Transformer) to reconstruct CO₂ fluxes.
- **Data**: FOCI-MOPS v1 (1958–2018), downsampled to a **1°×1° global grid** with monthly resolution.
- **Possible Targets**: `fco2`, `fco2_pre`, `co2flux`, and `co2flux_pre`.
---

## Project Structure

```bash
├── src/
│   ├── models/              # Model architectures (MLP, U-Net, UNext, SwinTransformer)
│   ├── utils/               # Preprocessing & evaluation utilities
│
├── notebooks/               
│   ├── optuna/              # Notebooks to run hyperparameter optimization
│   ├── plots/               # Notebooks to generate plots
│   ├── preprocessing/       # Notebook to preprocess datasets
│   ├── training/            # Notebooks to train different models
│   ├── other/               # Preprocessing & evaluation utilities
├── data/                    # (not included) raw & processed datasets
├── outputs/                 # (not included) Experiment outputs
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/jakobmeggendorfer/co2-flux-reconstruction.git
cd co2-flux-reconstruction

# Create environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```
---

## Usage

1. **Preprocessing**
   - Open `notebooks/preprocessing/regrid_and_store_to_numpy.ipynb`
   - Set variable `data_path` to the relative path of the folder that contains the pickle files
   - Set variable `file_prefix` to the prefix that is identical for each pickle file
   - Set variable `dataset_id` to the id that you want to use for this experiment. It will be used as the name of teh folder which contains the preprocessed data for this experiment.
   - It is expected that the source files are matching with the following naming scheme {file_prefix}{year}_df.pkl, where year ranges from 1958 to 2018.
   - Run all cells in this notebook.

2. **Run model training**
   - Go to `notebooks/training` and open the respective model training notebook
   - In the first cell (after the imports):
     - Specify folder names of datasets inside the `data/` folder
     - Optionally specify other variables in that cell
   - Run all cells in this notebook.

3. **Check results**
   - After training, you will find a folder for your run in the `output/` directory
   - This folder contains:
     - The scaler
     - The trained model
     - All evaluations that were performed automatically
---

## Acknowledgements
