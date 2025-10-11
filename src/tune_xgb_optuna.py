#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for XGBoost classifier (sleep risk model).

Workflow / intent (high-level):
  1. Load data:
     - If --train and --eval are provided, use those CSVs (they should be
       feature-engineered and include the target column).
     - Otherwise, load the merged CSV (default: data/processed/merged_clean.csv),
       drop rows missing the target and perform a stratified train/eval split.
  2. Preprocess:
     - Drop rows missing the target.
     - Optionally sample (sample_frac) for faster tuning.
     - Convert all non-numeric columns (object/category/bool) to numeric codes,
       aligning categories between train and eval (so encoding is consistent).
     - Fill numeric NaNs with -1 sentinel (trees handle sentinel well).
  3. Tune with Optuna:
     - Optimize ROC-AUC on the eval set.
     - Use XGBoost's histogram tree method for speed.
     - Log params & metrics for each trial to MLflow (if tracking URI set).
  4. Retrain:
     - Retrain best model (either on train only or on train+eval if --retrain-on-full).
     - Save model to disk and log model + preprocessing mapping to MLflow.
  5. Save artifacts:
     - joblib model file (default models/xgb_best_model.pkl)
     - JSON with categorical mappings (models/xgb_cat_maps.json)

Usage examples:
  python src/tune_xgb_optuna.py --n-trials 20 --sample-frac 0.3
  python src/tune_xgb_optuna.py --merged data/processed/merged_clean.csv --n-trials 40 --retrain-on-full
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost


# ---- configuration defaults ----
DEFAULT_MERGED = Path("data/processed/merged_clean.csv")
DEFAULT_TRAIN = None
DEFAULT_EVAL = None
DEFAULT_OUT = Path("models/xgb_best_model.pkl")

# configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tune_xgb_optuna")


# ---- helpers ----


def _encode_categorical_columns(X_train: pd.DataFrame, X_eval: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, list]]:
    """
    Convert object / category columns to integer category codes for both train and eval.
    - Align categories between train and eval (union of observed categories).
    - Convert booleans to ints.
    - Convert remaining object-like columns to numeric codes.
    - Fill numeric NaNs with -1 sentinel.
    Returns (X_train_enc, X_eval_enc, mappings)
    mappings: dict column -> list(categories_as_strings)
    """
    X_train = X_train.copy()
    X_eval = X_eval.copy()

    # 1) convert bool -> int
    bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()
    for c in bool_cols:
        X_train[c] = X_train[c].astype("int64")
        if c in X_eval.columns:
            X_eval[c] = X_eval[c].astype("int64")

    # 2) determine object/category columns seen in either frame
    obj_cols = set(X_train.select_dtypes(include=["object", "category"]).columns.tolist()) | set(
        X_eval.select_dtypes(include=["object", "category"]).columns.tolist()
    )

    mappings: Dict[str, list] = {}
    for col in sorted(obj_cols):
        # ensure we have series to work with
        train_col = X_train[col] if col in X_train.columns else pd.Series(dtype="string")
        eval_col = X_eval[col] if col in X_eval.columns else pd.Series(dtype="string")

        # union of categories (exclude NA)
        combined = pd.concat([train_col.astype("string"), eval_col.astype("string")], ignore_index=True)
        cats = pd.Categorical(combined.dropna()).categories

        # encode with same categories; pandas codes use -1 for NaN/unseen
        X_train[col] = pd.Categorical(X_train[col].astype("string"), categories=cats).codes
        X_eval[col] = pd.Categorical(X_eval[col].astype("string"), categories=cats).codes

        mappings[col] = [str(c) for c in cats]

    # 3) try to convert other object-like columns to numeric (coerce => NaN)
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_eval = X_eval.apply(pd.to_numeric, errors="coerce")

    # 4) fill remaining NaNs with -1 sentinel
    X_train = X_train.fillna(-1)
    X_eval = X_eval.fillna(-1)

    # final safety: ensure columns align and have numeric dtypes
    # convert any remaining non-numeric columns via categorical codes
    for col in list(X_train.columns):
        if X_train[col].dtype == object:
            X_train[col] = pd.Categorical(X_train[col].astype("string")).codes
    for col in list(X_eval.columns):
        if X_eval[col].dtype == object:
            X_eval[col] = pd.Categorical(X_eval[col].astype("string")).codes

    # safety check - if there are still non-numeric columns, drop them with a warning
    nonnum_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    nonnum_eval = X_eval.select_dtypes(exclude=[np.number]).columns.tolist()
    if nonnum_train:
        logger.warning("Dropping non-numeric columns from X_train after encoding: %s", nonnum_train)
        X_train = X_train.drop(columns=nonnum_train)
    if nonnum_eval:
        logger.warning("Dropping non-numeric columns from X_eval after encoding: %s", nonnum_eval)
        X_eval = X_eval.drop(columns=nonnum_eval)

    return X_train, X_eval, mappings


def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int) -> pd.DataFrame:
    if sample_frac is None:
        return df.reset_index(drop=True)
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df.reset_index(drop=True)
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def _load_from_files(
    train_path: str | Path,
    eval_path: str | Path,
    sample_frac: Optional[float],
    random_state: int,
    target_col: str,
):
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    train_df = _maybe_sample(train_df, sample_frac, random_state)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    eval_df = eval_df.dropna(subset=[target_col]).reset_index(drop=True)

    y_train = train_df[target_col].astype(int)
    X_train = train_df.drop(columns=[target_col])
    y_eval = eval_df[target_col].astype(int)
    X_eval = eval_df.drop(columns=[target_col])

    return X_train, y_train, X_eval, y_eval


def _load_from_merged(
    merged_path: str | Path,
    sample_frac: Optional[float],
    random_state: int,
    target_col: str,
    test_size: float,
):
    merged_df = pd.read_csv(merged_path)
    merged_df = merged_df.dropna(subset=[target_col]).reset_index(drop=True)
    merged_df = _maybe_sample(merged_df, sample_frac, random_state)

    y = merged_df[target_col].astype(int)
    X = merged_df.drop(columns=[target_col])

    # stratified split to preserve class balance when possible
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train.reset_index(drop=True), y_train.reset_index(drop=True), X_eval.reset_index(drop=True), y_eval.reset_index(
        drop=True
    )


# ---- main tuning function ----


def tune_model(
    merged_path: str | Path = DEFAULT_MERGED,
    train_path: Optional[str | Path] = DEFAULT_TRAIN,
    eval_path: Optional[str | Path] = DEFAULT_EVAL,
    model_output: str | Path = DEFAULT_OUT,
    n_trials: int = 20,
    sample_frac: Optional[float] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "xgboost_optuna_sleep_risk",
    random_state: int = 42,
    early_stopping_rounds: int = 50,
    test_size: float = 0.2,
    target_col: str = "sleep_disorder",
    retrain_on_full: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run Optuna tuning; save best model; return (best_params, best_metrics).
    """
    # MLflow configuration
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    if train_path and eval_path:
        logger.info("Loading explicit train/eval CSVs: %s , %s", train_path, eval_path)
        X_train, y_train, X_eval, y_eval = _load_from_files(train_path, eval_path, sample_frac, random_state, target_col)
    else:
        merged_path = Path(merged_path)
        if not merged_path.exists():
            raise FileNotFoundError(
                f"Merged file not found: {merged_path}. Run the pipeline to produce data/processed/merged_clean.csv first."
            )
        logger.info("Loading merged CSV and creating train/eval split: %s", merged_path)
        X_train, y_train, X_eval, y_eval = _load_from_merged(merged_path, sample_frac, random_state, target_col, test_size)

    logger.info("Data shapes (before encoding): X_train=%s, X_eval=%s, y_train=%s", X_train.shape, X_eval.shape, y_train.shape)

    # Encode non-numeric columns once (before Optuna trials)
    X_train, X_eval, cat_maps = _encode_categorical_columns(X_train, X_eval)
    logger.info("After encoding: X_train=%s, X_eval=%s; categorical columns encoded: %s", X_train.shape, X_eval.shape, list(cat_maps.keys()))

    # Quick check - there should be no non-numeric columns remaining
    nonnum_train = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    nonnum_eval = X_eval.select_dtypes(exclude=[np.number]).columns.tolist()
    if nonnum_train or nonnum_eval:
        logger.warning("Unexpected non-numeric columns remain (train=%s, eval=%s). They will be dropped.", nonnum_train, nonnum_eval)
        X_train = X_train.select_dtypes(include=[np.number])
        X_eval = X_eval.select_dtypes(include=[np.number])

    # Save the categorical mapping next to the model output for reproducibility
    model_out_path = Path(model_output)
    mapping_path = model_out_path.parent / "xgb_cat_maps.json"
    try:
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as fh:
            json.dump(cat_maps, fh, indent=2, ensure_ascii=False)
        logger.info("Saved categorical mapping to %s", mapping_path)
    except Exception as e:
        logger.warning("Failed to save categorical mappings: %s", e)

    # Define objective
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "tree_method": "hist",
            # xgboost sklearn API
            "use_label_encoder": False,
            "verbosity": 0,
        }

        # handle class imbalance
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        if pos > 0 and neg > 0:
            params["scale_pos_weight"] = float(neg) / float(pos)

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = XGBClassifier(**params)

            # fit with early stopping on eval set
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_eval, y_eval)],
                eval_metric="auc",
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )

            # evaluate
            y_prob = model.predict_proba(X_eval)[:, 1]
            y_pred = model.predict(X_eval)

            try:
                roc = float(roc_auc_score(y_eval, y_prob))
            except Exception:
                roc = 0.0
            pr_auc = float(average_precision_score(y_eval, y_prob))
            acc = float(accuracy_score(y_eval, y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average="binary", zero_division=0)

            mlflow.log_metrics({"roc_auc": roc, "pr_auc": pr_auc, "accuracy": acc, "precision": float(precision), "recall": float(recall), "f1": float(f1)})

        # Optuna maximizes ROC AUC
        return roc

    # Run optuna study
    study = optuna.create_study(direction="maximize")
    logger.info("Starting Optuna study (n_trials=%s)", n_trials)
    try:
        study.optimize(objective, n_trials=n_trials)
    except Exception as e:
        logger.error("Optuna optimization ended with exception: %s", e)
        # re-raise so caller sees the problem
        raise

    best_params = study.best_trial.params
    logger.info("Best Optuna params: %s", best_params)

    # Retrain final model
    final_params = {**best_params, "random_state": random_state, "n_jobs": -1, "tree_method": "hist", "use_label_encoder": False, "verbosity": 0}
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    if pos > 0 and neg > 0:
        final_params["scale_pos_weight"] = float(neg) / float(pos)

    if retrain_on_full:
        logger.info("Retraining final model on train+eval (retrain_on_full=True)")
        X_full = pd.concat([X_train, X_eval], axis=0).reset_index(drop=True)
        y_full = pd.concat([y_train, y_eval], axis=0).reset_index(drop=True)
        # safety: convert to numeric if needed (shouldn't be necessary)
        X_full = X_full.select_dtypes(include=[np.number])
        final_model = XGBClassifier(**final_params)
        final_model.fit(X_full, y_full)
    else:
        logger.info("Retraining final model on train only")
        final_model = XGBClassifier(**final_params)
        final_model.fit(X_train, y_train)

    # Evaluate on eval set
    y_prob = final_model.predict_proba(X_eval)[:, 1]
    y_pred = final_model.predict(X_eval)
    best_metrics = {
        "roc_auc": float(roc_auc_score(y_eval, y_prob)) if len(np.unique(y_eval)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_eval, y_prob)),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
    }
    logger.info("Best tuned model metrics: %s", best_metrics)

    # Save model and mapping
    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(final_model, out)
    logger.info("Saved final model to %s", out)

    # Log artifacts to MLflow (final run)
    try:
        with mlflow.start_run(run_name="best_xgb_model"):
            mlflow.log_params(best_params)
            mlflow.log_metrics(best_metrics)
            # log the saved joblib model
            mlflow.xgboost.log_model(final_model, "model")
            # log mapping file if exists
            if mapping_path.exists():
                mlflow.log_artifact(str(mapping_path), artifact_path="preprocessing")
            logger.info("Logged final model and artifacts to MLflow under experiment '%s'", experiment_name)
    except Exception as e:
        logger.warning("Failed to log model or artifacts to MLflow: %s", e)

    return best_params, best_metrics


# ---- CLI ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna tuning + MLflow logging for XGBoost classifier (sleep risk)")
    parser.add_argument("--merged", default=str(DEFAULT_MERGED), help="merged CSV (default: data/processed/merged_clean.csv)")
    parser.add_argument("--train", default=None, help="optional train CSV (if provided, will use --train and --eval)")
    parser.add_argument("--eval", default=None, help="optional eval CSV (if provided, will use --train and --eval)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output model path (joblib .pkl)")
    parser.add_argument("--n-trials", type=int, default=20, help="number of Optuna trials")
    parser.add_argument("--sample-frac", type=float, default=None, help="optional sampling fraction for faster tuning (0.0-1.0)")
    parser.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (optional)")
    parser.add_argument("--experiment", default="xgboost_optuna_sleep_risk", help="MLflow experiment name")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--early-stop", type=int, default=50, help="early_stopping_rounds used during tuning")
    parser.add_argument("--test-size", type=float, default=0.2, help="eval split size when using --merged")
    parser.add_argument("--target-col", type=str, default="sleep_disorder", help="target column name in merged CSV")
    parser.add_argument("--retrain-on-full", action="store_true", help="retrain final model on train+eval (useful for production)")
    args = parser.parse_args()

    tune_model(
        merged_path=args.merged,
        train_path=args.train,
        eval_path=args.eval,
        model_output=args.out,
        n_trials=args.n_trials,
        sample_frac=args.sample_frac,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stop,
        test_size=args.test_size,
        target_col=args.target_col,
        retrain_on_full=args.retrain_on_full,
    )
