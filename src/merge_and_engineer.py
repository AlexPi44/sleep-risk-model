"""merge_and_engineer.py 

Read converted CSVs from data/raw/<cycle>/csv/, harmonize, merge on SEQN, 
create target and features, and write data/processed/merged_clean.csv
"""

import os
import sys
import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

CYCLES = [
    "2005_2006",
    "2007_2008",
    "2009_2010",
    "2011_2012",
    "2013_2014",
    "2015_2016",
]

CSV_DIR_TEMPLATE = "{root}/{cycle}/csv"

FEATURE_COLS = [
    "age",
    "sex",
    "BMI",
    "exercise_min_week",
    "calories_day",
    "fiber_g_day",
    "added_sugar_g_day",
    "caffeine_mg_day",
    "alcohol_drinks_week",
    "current_smoker",
    "depression_score",
    "systolic_bp",
    "diastolic_bp",
]


def safe_read_csv(path):
    """Safely read CSV file."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"ERROR: Could not read {path}: {e}", file=sys.stderr)
        return None


def harmonize_and_merge(root):
    """Load, harmonize, and merge all cycles on SEQN."""
    dfs = []
    skipped_cycles = []

    for cycle in CYCLES:
        csv_dir = CSV_DIR_TEMPLATE.format(root=root, cycle=cycle)
        if not os.path.isdir(csv_dir):
            print(f"SKIP: Missing cycle folder: {csv_dir}", file=sys.stderr)
            skipped_cycles.append(cycle)
            continue

        files = os.listdir(csv_dir)
        if not files:
            print(f"SKIP: Empty cycle folder: {csv_dir}", file=sys.stderr)
            skipped_cycles.append(cycle)
            continue

        # Load tables by file type
        table_map = {}
        for f in files:
            name = f.upper()
            if "DEMO" in name:
                table_map["DEMO"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "SLQ" in name:
                table_map["SLQ"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "PAQ" in name:
                table_map["PAQ"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "DR1TOT" in name:
                table_map["DR1TOT"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "DR2TOT" in name:
                table_map["DR2TOT"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "BMX" in name:
                table_map["BMX"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "BPX" in name:
                table_map["BPX"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "DPQ" in name:
                table_map["DPQ"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "ALQ" in name:
                table_map["ALQ"] = safe_read_csv(os.path.join(csv_dir, f))
            elif "SMQ" in name:
                table_map["SMQ"] = safe_read_csv(os.path.join(csv_dir, f))

        # Check for DEMO (required)
        if "DEMO" not in table_map or table_map["DEMO"] is None:
            print(f"SKIP: Cycle {cycle} missing DEMO table", file=sys.stderr)
            skipped_cycles.append(cycle)
            continue

        df = table_map["DEMO"]

        # Check for SEQN (unique ID)
        if "SEQN" not in df.columns:
            print(f"SKIP: DEMO in {cycle} missing SEQN column", file=sys.stderr)
            skipped_cycles.append(cycle)
            continue

        # Merge tables on SEQN
        for key in ["SLQ", "PAQ", "DR1TOT", "DR2TOT", "BMX", "BPX", "DPQ", "ALQ", "SMQ"]:
            if key in table_map and table_map[key] is not None:
                t = table_map[key]
                if "SEQN" in t.columns:
                    df = df.merge(t, on="SEQN", how="left")
                else:
                    print(f"WARN: {key} in {cycle} missing SEQN, skipping", file=sys.stderr)

        df["cycle"] = cycle
        dfs.append(df)
        print(f"OK: Loaded cycle {cycle} ({len(df)} rows)")

    if not dfs:
        raise RuntimeError(
            f"No cycles loaded. Checked: {CYCLES}. Skipped: {skipped_cycles}"
        )

    print(f"\nMerging {len(dfs)} cycles...")
    big = pd.concat(dfs, ignore_index=True, sort=False)
    return big


def create_sleep_disorder_label(row):
    """Create binary sleep disorder label based on NHANES questionnaire responses.
    
    HIGH RISK (1) if ANY of:
    - Doctor-diagnosed sleep disorder (SLQ060 == 1)
    - Sleep duration < 7 or > 9 hours (SLD012)
    - Frequent excessive sleepiness >= 16 times/month (SLQ120)
    
    LOW RISK (0) if:
    - Sleep duration 7-9 hours AND no doctor diagnosis
    
    Otherwise: NaN (insufficient data)
    """
    # Check 1: Doctor-diagnosed sleep disorder
    if pd.notna(row.get("SLQ060")):
        try:
            if int(row.get("SLQ060")) == 1:
                return 1
        except (ValueError, TypeError):
            pass

    # Check 2: Sleep duration
    if pd.notna(row.get("SLD012")):
        try:
            sleep_hrs = float(row.get("SLD012"))
            if sleep_hrs < 7 or sleep_hrs > 9:
                return 1  # Short or long sleep = high risk
            elif 7 <= sleep_hrs <= 9:
                return 0  # Normal sleep = low risk (if no diagnosis above)
        except (ValueError, TypeError):
            pass

    # Check 3: Excessive sleepiness (only if sleep hours not informative)
    if pd.notna(row.get("SLQ120")):
        try:
            sleepiness = float(row.get("SLQ120"))
            if sleepiness >= 16:
                return 1
        except (ValueError, TypeError):
            pass

    # Insufficient data
    return np.nan


def mean_col(df, cols):
    """Calculate mean across multiple columns, handling NaN properly.
    
    Returns Series, not scalar.
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    # mean(axis=1) calculates row-wise mean, skipna=True ignores NaN
    return df[present].mean(axis=1, skipna=True)


def engineer_features(df):
    """Engineer all features from raw NHANES variables."""
    out = pd.DataFrame(index=df.index)

    # Demographics (non-modifiable)
    out["age"] = df.get("RIDAGEYR", pd.NA)
    out["sex"] = df.get("RIAGENDR", pd.NA)

    # Anthropometry
    out["BMI"] = df.get("BMXBMI", pd.NA)

    # Exercise (moderate + 2*vigorous minutes per week)
    pad680 = df.get("PAD680", pd.Series([np.nan] * len(df)))
    pad700 = df.get("PAD700", pd.Series([np.nan] * len(df)))
    # Ensure Series type
    if not isinstance(pad680, pd.Series):
        pad680 = pd.Series([pad680] * len(df), index=df.index)
    if not isinstance(pad700, pd.Series):
        pad700 = pd.Series([pad700] * len(df), index=df.index)

    out["exercise_min_week"] = pad680.fillna(0) + (pad700.fillna(0) * 2)

    # Diet (average of 24-hr dietary recall 1 & 2)
    out["calories_day"] = mean_col(df, ["DR1TKCAL", "DR2TKCAL"])
    out["fiber_g_day"] = mean_col(df, ["DR1TFIBE", "DR2TFIBE"])
    out["added_sugar_g_day"] = mean_col(df, ["DR1TSUGR", "DR2TSUGR"])
    out["caffeine_mg_day"] = mean_col(df, ["DR1TCAFF", "DR2TCAFF"])

    # Alcohol (drinks per week)
    if "ALQ130" in df.columns:
        # ALQ130 is drinks/day in NHANES
        out["alcohol_drinks_week"] = df["ALQ130"].astype(float).fillna(0) * 7
    else:
        out["alcohol_drinks_week"] = np.nan

    # Smoking (current smoker)
    if "SMQ040" in df.columns:
        # SMQ040: 1=every day, 2=some days, 3=not at all
        out["current_smoker"] = (
            df["SMQ040"].apply(lambda x: 1 if (pd.notna(x) and (x <= 2)) else 0)
        ).astype("Int64")
    else:
        out["current_smoker"] = np.nan

    # Depression (PHQ-9 score: sum of DPQ001-DPQ009)
    dpq_cols = [c for c in df.columns if c and str(c).upper().startswith("DPQ0")]
    if dpq_cols:
        # Sum across depression items (each 0-3, total 0-27)
        out["depression_score"] = df[dpq_cols].sum(axis=1, min_count=1)
    else:
        out["depression_score"] = np.nan

    # Blood pressure
    out["systolic_bp"] = df.get("BPXSY1", pd.NA)
    out["diastolic_bp"] = df.get("BPXDI1", pd.NA)

    # Target variables (for reference, not used in model)
    out["SLQ060"] = df.get("SLQ060", pd.NA)
    out["SLD012"] = df.get("SLD012", pd.NA)
    out["SLQ120"] = df.get("SLQ120", pd.NA)
    out["cycle"] = df.get("cycle", pd.NA)

    # Create target label
    out["sleep_disorder"] = df.apply(create_sleep_disorder_label, axis=1)

    return out


def main(args):
    root = args.input
    out_path = Path(args.output)

    print(f"Loading from: {root}")
    print(f"Output to: {out_path}")

    big = harmonize_and_merge(root)
    print(f"Merged dataframe: {big.shape}")

    eng = engineer_features(big)
    print(f"Engineered features: {eng.shape}")

    # Select final columns
    keep = ["cycle"] + FEATURE_COLS + ["sleep_disorder"]
    for c in keep:
        if c not in eng.columns:
            print(f"WARN: Column {c} not found in engineered features", file=sys.stderr)
            eng[c] = np.nan

    out_df = eng[keep]

    # Report target distribution
    target_counts = out_df["sleep_disorder"].value_counts(dropna=False)
    print(f"\nTarget distribution:")
    print(target_counts)
    print(
        f"Missing/NaN: {target_counts.get(np.nan, 0)} "
        f"({100*target_counts.get(np.nan, 0)/len(out_df):.1f}%)"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote merged CSV to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge NHANES cycles and engineer features")
    parser.add_argument(
        "--input",
        default="data/raw",
        help="Root raw directory where cycle folders live",
    )
    parser.add_argument(
        "--output",
        default="data/processed/merged_clean.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    main(args)
