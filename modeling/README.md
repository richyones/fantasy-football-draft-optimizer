# Modeling

This directory contains the data science and modeling code for the Fantasy Football Draft Optimizer.

## Structure

- `scripts/` - Python scripts for data processing and feature engineering
- `notebooks/` - Jupyter notebooks for exploratory analysis and variable selection

## Files

- `Fantasy PPR Scoring and Dataframes.py` - Main script for calculating fantasy points and creating dataframes
- `FFVariableSelection.ipynb` - Jupyter notebook for variable selection and feature engineering

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main scoring script:
```bash
python scripts/"Fantasy PPR Scoring and Dataframes.py"
```

Open the variable selection notebook:
```bash
jupyter notebook notebooks/FFVariableSelection.ipynb
```

## Output

The scripts generate feature files that are stored in `../data/processed/`:
- `QB_features.xls`
- `RB_features.xls`
- `WR_features.xls`
- `TE_features.xls`
- `dst_features.csv`
- `kicker_features.csv`

