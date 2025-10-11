#!/usr/bin/env python3
"""
01_data_prep.py - COMPLETE VERSION
Sleep Disorder Risk Model - Full Training Pipeline

Runs end-to-end: download → merge → train → save model
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
import joblib

print("="*70)
print("SLEEP DISORDER RISK MODEL - TRAINING PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: DOWNLOAD NHANES DATA
# ============================================================================
print("\n[STEP 1/5] Checking NHANES data...")

os.makedirs('data/raw', exist_ok=True)
os.makedirs('reports', exist_ok=True)

if Path('reports/aggregate_report.json').exists():
    print("✓ Data already downloaded (reports/ exists)")
    print("  Delete reports/ to force re-download\n")
else:
    print("Downloading NHANES data (this takes 10-30 minutes)...\n")
    result = subprocess.run(
        ['python', 'src/check_nhanes_downloads.py', 
         '--out', 'data/raw', '--report', 'reports', '--convert'],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"\n✗ Download failed (exit code {result.returncode})")
        sys.exit(1)
    print("\n✓ Download complete")

# ============================================================================
# STEP 2: INSPECT REPORTS
# ============================================================================
print("\n[STEP 2/5] Inspecting download reports...")

report_dir = Path('reports')
if not report_dir.exists():
    print("✗ No reports/ directory found")
    sys.exit(1)

reports = sorted(report_dir.glob('*_report.json'))
if not reports:
    print("✗ No report files found")
    sys.exit(1)

total_missing = 0
for p in reports:
    with open(p, 'r', encoding='utf-8') as fh:
        rep = json.load(fh)
    
    cycle_name = rep.get('cycle', p.stem)
    print(f"\n  {cycle_name}:")
    
    for fname, info in rep.get('files', {}).items():
        downloaded = info.get('downloaded', False)
        missing = info.get('missing_variables', [])
        
        if not downloaded:
            print(f"    ✗ {fname}: NOT DOWNLOADED")
        elif missing:
            print(f"    ⚠ {fname}: missing {len(missing)} vars")
            total_missing += len(missing)
        else:
            print(f"    ✓ {fname}: OK")

if total_missing > 0:
    print(f"\nWARNING: {total_missing} key variables missing across cycles")
    print("Check CDC NHANES documentation if this affects your analysis")

print("\n✓ Reports inspection complete")

# ============================================================================
# STEP 3: MERGE & ENGINEER FEATURES
# ============================================================================
print("\n[STEP 3/5] Merging cycles and engineering features...")

os.makedirs('data/processed', exist_ok=True)
output_path = Path('data/processed/merged_clean.csv')

if output_path.exists():
    print(f"✓ Merged CSV already exists: {output_path}")
    print("  Delete to force re-merge\n")
else:
    result = subprocess.run(
        ['python', 'src/merge_and_engineer.py', 
         '--input', 'data/raw', '--output', str(output_path)],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"\n✗ Merge failed (exit code {result.returncode})")
        sys.exit(1)
    print("\n✓ Merge complete")

# ============================================================================
# STEP 4: LOAD & PREPROCESS
# ============================================================================
print("\n[STEP 4/5] Loading and preprocessing data...")

if not output_path.exists():
    print(f"✗ Merged CSV not found: {output_path}")
    sys.exit(1)

df = pd.read_csv(output_path)
print(f"  Loaded: {df.shape}")

feature_cols = [
    'age', 'sex', 'BMI', 'exercise_min_week',
    'calories_day', 'fiber_g_day', 'added_sugar_g_day', 'caffeine_mg_day',
    'alcohol_drinks_week', 'current_smoker', 'depression_score',
    'systolic_bp', 'diastolic_bp'
]

# Keep only rows with target
df2 = df[df['sleep_disorder'].notna()].copy()
print(f"  With target: {len(df2)}")

# Drop rows with missing features
df_model = df2[feature_cols + ['sleep_disorder']].dropna()
print(f"  After dropping missing: {len(df_model)}")

if len(df_model) < 1000:
    print("\nWARNING: Very few samples. Model may not be reliable.")

# Target distribution
X = df_model[feature_cols]
y = df_model['sleep_disorder'].astype(int)

print(f"\nTarget distribution:")
print(f"  Low risk (0):  {(y==0).sum()}")
print(f"  High risk (1): {(y==1).sum()}")
print(f"  Balance: {100*y.mean():.1f}% high risk")

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================
print("\n[STEP 5/5] Training XGBoost model...")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train
try:
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_s, y_train)
    print("  ✓ Training complete")
    
    # Evaluate
    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)
    
    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)
    
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"ROC-AUC: {roc:.3f}")
    print(f"PR-AUC:  {pr:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0,0]:5d} | FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d} | TP={cm[1,1]:5d}")
    
except ImportError:
    print("\n✗ xgboost not installed. Run: pip install xgboost")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    sys.exit(1)

# ============================================================================
# SAVE MODEL
# ============================================================================
print(f"\n{'='*70}")
print("SAVING MODEL ARTIFACTS")
print(f"{'='*70}")

os.makedirs('models', exist_ok=True)

try:
    joblib.dump(model, 'models/sleep_risk_model.pkl')
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    
    print("✓ Saved to models/:")
    print("  - sleep_risk_model.pkl")
    print("  - feature_scaler.pkl")
    print("  - feature_names.pkl")
except Exception as e:
    print(f"✗ Save failed: {e}")
    sys.exit(1)

# ============================================================================
# DONE
# ============================================================================
print(f"\n{'='*70}")
print("✓ TRAINING COMPLETE")
print(f"{'='*70}")
print("\nNext steps:")
print("  1. Run Streamlit UI: streamlit run streamlit_app.py")
print("  2. Or use model programmatically:")
print("     model = joblib.load('models/sleep_risk_model.pkl')")
print()
