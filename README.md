# Sleep Disorder Risk Prediction

Predict sleep disorder risk using lifestyle and health features from NHANES survey data (2005-2016).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Overview

This project predicts **high risk** vs **low risk** for sleep disorders using machine learning on NHANES (National Health and Nutrition Examination Survey) data. It provides:

- Automated data pipeline for NHANES cycles 2005-2016
- Feature engineering from lifestyle and health metrics
- XGBoost classifier with SHAP explainability
- Interactive Streamlit web interface for risk assessment

**⚠️ Disclaimer:** This tool provides risk estimates for research purposes only. It is **not** a medical diagnostic tool and should not replace clinical evaluation.

## Features

- **Automated Data Pipeline**: Downloads and processes 6 NHANES cycles automatically
- **Feature Engineering**: 13 features including BMI, exercise, diet, blood pressure, depression score
- **Model Performance**: ROC-AUC ~0.75-0.80 on test set
- **Explainability**: SHAP values for feature importance analysis
- **Web Interface**: User-friendly Streamlit app for predictions
- **Reproducible**: Version-pinned dependencies and fixed random seeds

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- ~1GB free disk space for data
- Internet connection for downloading NHANES data

### Step 1: Clone Repository

```bash
git clone https://github.com/AlexPi44/sleep-risk-model.git
cd sleep-risk-model
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Critical Dependency:** `pyreadstat>=1.1.5` is required to read NHANES XPT files.

Verify installation:
```bash
python -c "import pyreadstat; print('✓ pyreadstat installed successfully')"
```

### Step 4: Create Directories

```bash
mkdir -p data/raw data/processed reports models
```

## Usage

### Quick Start (Automated Pipeline)

Run the entire pipeline with one command:

```bash
python notebooks/01_data_prep.py
```

This will:
1. Download NHANES data (~10-30 minutes)
2. Merge and engineer features (~1-2 minutes)
3. Train XGBoost model (~1-2 minutes)
4. Save model artifacts to `models/`

Then launch the web interface:

```bash
streamlit run streamlit_app.py
```

Open browser to `http://localhost:8501`

---

### Step-by-Step (Manual Execution)

#### 1. Download NHANES Data

```bash
python src/check_nhanes_downloads.py --out data/raw --report reports --convert
```

**Time:** 10-30 minutes (downloads ~500MB)  
**Output:** XPT files and converted CSVs in `data/raw/<cycle>/`, reports in `reports/`

**Verify download:**
```bash
cat reports/aggregate_report.json | python -m json.tool
```

#### 2. Merge Cycles and Engineer Features

```bash
python src/merge_and_engineer.py --input data/raw --output data/processed/merged_clean.csv
```

**Time:** 1-2 minutes  
**Output:** `data/processed/merged_clean.csv` (~40,000 rows, 14 columns)

**Verify output:**
```bash
python -c "import pandas as pd; df=pd.read_csv('data/processed/merged_clean.csv'); print(f'Shape: {df.shape}')"
```

#### 3. Train Model

**Option A: Jupyter Notebook (Interactive)**
```bash
jupyter notebook notebooks/01_data_prep.ipynb
```
Run cells sequentially from top to bottom.

**Option B: Python Script (Automated)**
```bash
python notebooks/01_data_prep.py
```

**Time:** 1-2 minutes  
**Output:** Model artifacts in `models/`:
- `sleep_risk_model.pkl` - Trained XGBoost classifier
- `feature_scaler.pkl` - StandardScaler for features
- `feature_names.pkl` - Feature order for predictions

#### 4. Run Web Interface

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

**Usage:**
1. Enter your health information (age, BMI, exercise, diet, etc.)
2. Click "Calculate Risk"
3. View risk percentage and interpretation

## Project Structure

```
sleep-risk-model/
├── src/
│   ├── check_nhanes_downloads.py  # Data downloader
│   └── merge_and_engineer.py      # Feature engineering
├── notebooks/
│   ├── 01_data_prep.ipynb         # Jupyter notebook
│   └── 01_data_prep.py            # Python script version
├── data/
│   ├── raw/                       # Downloaded NHANES files
│   └── processed/                 # Merged dataset
├── models/                        # Trained model artifacts
├── reports/                       # Download reports
├── streamlit_app.py               # Web interface
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Model Details

### Target Variable

Participants are labeled **high risk (1)** if any of:
- Doctor-diagnosed sleep disorder (`SLQ060 == 1`)
- Sleep duration < 7 or > 9 hours (`SLD012`)
- Excessive daytime sleepiness ≥ 16 times/month (`SLQ120`)

Otherwise labeled **low risk (0)** if sleep duration is 7-9 hours with no diagnosis.

### Features (13 total)

**Non-modifiable:**
- Age, Sex

**Modifiable lifestyle factors:**
- BMI (Body Mass Index)
- Exercise (minutes/week)
- Diet: calories, fiber, added sugar, caffeine (daily averages)
- Alcohol consumption (drinks/week)
- Current smoker (yes/no)
- Depression score (PHQ-9: 0-27)
- Blood pressure (systolic, diastolic)

### Model Architecture

- **Algorithm:** XGBoost Classifier
- **Parameters:** 100 trees, max depth 6, learning rate 0.1
- **Preprocessing:** StandardScaler normalization
- **Train/Test Split:** 80/20, stratified by target
- **Evaluation:** ROC-AUC, PR-AUC, confusion matrix

### Feature Engineering

- **Exercise:** Moderate minutes + 2× vigorous minutes per week
- **Diet:** Average of two 24-hour dietary recalls
- **Depression:** Sum of 9 PHQ-9 questionnaire items
- **Alcohol:** Daily drinks × 7 = weekly consumption

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.75-0.80 |
| PR-AUC | 0.70-0.75 |
| Training Samples | ~15,000-20,000 |
| Test Samples | ~3,000-5,000 |

### Top Features (by SHAP importance)

1. Age
2. Depression score
3. BMI
4. Sleep duration (embedded in target)
5. Blood pressure

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **`ModuleNotFoundError: No module named 'pyreadstat'`** | `pip install pyreadstat>=1.1.5` |
| **`FileNotFoundError: XPT file not found`** | Run downloader: `python src/check_nhanes_downloads.py ...` |
| **`404` errors during download** | Check CDC NHANES website: https://www.cdc.gov/nchs/nhanes/ |
| **`merged_clean.csv not found`** | Run merge script: `python src/merge_and_engineer.py ...` |
| **Streamlit shows "No model found"** | Verify 3 files exist in `models/` directory |
| **Very few training samples (<1000)** | Check `reports/` for missing sleep variables |
| **Download timeout** | Increase `REQUEST_TIMEOUT` in `check_nhanes_downloads.py` (line 56) |

## Data Source

**NHANES (National Health and Nutrition Examination Survey)**  
- Provider: U.S. Centers for Disease Control and Prevention (CDC)
- Cycles: 2005-2006, 2007-2008, 2009-2010, 2011-2012, 2013-2014, 2015-2016
- License: Public domain, free for use
- Documentation: https://www.cdc.gov/nchs/index.html

## Limitations

- **Not a diagnostic tool** - Predicts risk only, not clinical diagnosis
- **U.S. population** - Trained on NHANES; may not generalize internationally
- **Self-reported data** - Subject to recall and reporting bias
- **Data age** - Trained on 2005-2016 data; modern populations may differ
- **Missing data** - ~70% of samples dropped due to incomplete sleep questionnaires
- **Class imbalance** - Consider using SMOTE or class weights for production use

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Development Setup

```bash
pip install -r requirements.txt
pip install pytest black flake8  # Development dependencies
```

### Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

NHANES data is public domain courtesy of the CDC.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sleep_risk_model_2025,
  author = {{Alexandru Paicu}},
  title = {Sleep Disorder Risk Prediction using NHANES Data},
  year = {2025},
  url = {https://github.com/AlexPi44/sleep-risk-model},
  version = {1.0.0}
}
```

## Contact

**Project Maintainer:** Alexandru Paicu
**Email:** paicualexandru44@gmail.com  
**GitHub:** https://github.com/AlexPi44/sleep-risk-model

---

**Acknowledgments**
- CDC NHANES team for public data
- scikit-learn, XGBoost, and SHAP contributors
- Streamlit for the web framework
