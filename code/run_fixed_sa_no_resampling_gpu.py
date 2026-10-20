# -*- coding: utf-8 -*-
"""Repeated grouped validation of the fixed SA model without resampling.

This run keeps the previously fixed settings:
* 10 repeated, pot-grouped five-fold outer validation;
* three-fold grouped OOF stacking inside each outer training fold;
* NIR + SWIR SA features;
* logistic regression C=0.3, class_weight=None;
* no grade-0/1 threshold.

The only pipeline change is removal of grade-0 undersampling and SMOTE.  All
three meta-model comparisons use the same logistic-regression settings, so the
NIR+SWIR comparison isolates the SA features.  XGBoost uses CUDA; SVM, RF and
logistic regression use CPU implementations.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost

import run_repeated_grouped_sa_ablation as base
import run_fixed_sa_repeated_validation as fixed


OUT = base.ROOT / "work" / "fixed_sa_no_resampling_gpu"
OUT.mkdir(parents=True, exist_ok=True)
fixed.OUT = OUT


def inner_oof_and_outer_predictions_no_resampling(
    Xtr, ytr, gtr, Xte, wave, seed
):
    """Generate OOF/test base probabilities without any class resampling."""
    inner = StratifiedGroupKFold(
        n_splits=base.INNER_FOLDS, shuffle=True, random_state=seed
    )
    names = ["xgb", "svm", "rf"]
    oof = {name: np.zeros((len(ytr), 6), dtype=float) for name in names}
    sa_oof = np.zeros((len(ytr), 24), dtype=float)

    for inner_fold, (fit_idx, val_idx) in enumerate(
        inner.split(Xtr, ytr, gtr), start=1
    ):
        fit_seed = seed + inner_fold
        Xi, yi, gi = Xtr[fit_idx], ytr[fit_idx], gtr[fit_idx]
        centres = base.group_balanced_centres(Xi, yi, gi)
        sa_oof[val_idx] = base.spectral_angle_features(
            Xtr[val_idx], centres, wave
        )
        for name in names:
            model = base.fit_base(name, Xi, yi, fit_seed)
            oof[name][val_idx] = model.predict_proba(Xtr[val_idx])
            del model
        gc.collect()

    fit_seed = seed + 100
    centres = base.group_balanced_centres(Xtr, ytr, gtr)
    sa_test = base.spectral_angle_features(Xte, centres, wave)
    test = {}
    for name in names:
        model = base.fit_base(name, Xtr, ytr, fit_seed)
        test[name] = model.predict_proba(Xte)
        del model
    gc.collect()
    return oof, test, sa_oof, sa_test


def meta_predict_same_settings(train_features, y_train, test_features, seed):
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        C=0.3,
        max_iter=3000,
        class_weight=None,
        random_state=seed,
    )
    model.fit(scaler.transform(train_features), y_train)
    return model.predict(scaler.transform(test_features))


def evaluate_fold(
    X, y, groups, meta, wave, repeat_seed, fold, train_idx, test_idx
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    model_seed = repeat_seed + fold * 1000

    oof, test, sa_oof, sa_test = (
        inner_oof_and_outer_predictions_no_resampling(
            X_train, y_train, groups_train, X_test, wave, model_seed
        )
    )
    base_train = np.hstack([oof["xgb"], oof["svm"], oof["rf"]])
    base_test = np.hstack([test["xgb"], test["svm"], test["rf"]])
    nir_swir_columns = fixed.NIR_SWIR_COLUMNS

    predictions = {
        "No-SA fixed": meta_predict_same_settings(
            base_train, y_train, base_test, model_seed
        ),
        "All-region SA old": meta_predict_same_settings(
            np.hstack([base_train, sa_oof]),
            y_train,
            np.hstack([base_test, sa_test]),
            model_seed,
        ),
        "NIR+SWIR SA fixed": meta_predict_same_settings(
            np.hstack([base_train, sa_oof[:, nir_swir_columns]]),
            y_train,
            np.hstack([base_test, sa_test[:, nir_swir_columns]]),
            model_seed,
        ),
    }

    metric_rows = []
    prediction_rows = []
    for model_name in fixed.MODEL_ORDER:
        prediction = predictions[model_name]
        metric_rows.append(
            {
                "repeat_seed": repeat_seed,
                "fold": fold,
                "model": model_name,
                "accuracy": accuracy_score(y_test, prediction),
                "macro_f1": f1_score(y_test, prediction, average="macro"),
                "n_test_spectra": len(test_idx),
                "n_test_pots": len(np.unique(groups[test_idx])),
            }
        )
        for local_index, predicted_grade in enumerate(prediction):
            row_index = int(test_idx[local_index])
            prediction_rows.append(
                {
                    "repeat_seed": repeat_seed,
                    "fold": fold,
                    "model": model_name,
                    "row_index": row_index,
                    "true_grade": int(y_test[local_index]),
                    "predicted_grade": int(predicted_grade),
                    "pot_id": str(groups[row_index]),
                    "source_file": str(meta.iloc[row_index]["source_file"]),
                    "date": str(meta.iloc[row_index]["Date"]),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def write_metadata():
    metadata = {
        "experiment": "fixed NIR+SWIR SA without resampling",
        "outer_validation": "10 repeated pot-grouped five-fold CV",
        "inner_stacking": "three-fold pot-grouped OOF predictions",
        "resampling": "none",
        "SA_regions": ["NIR", "SWIR"],
        "meta_C": 0.3,
        "meta_class_weight": None,
        "grade01_threshold": None,
        "base_learners": ["XGBoost", "SVM", "RF"],
        "xgboost_device": "cuda",
        "xgboost_version": xgboost.__version__,
        "xgboost_cuda_build": bool(xgboost.build_info().get("USE_CUDA")),
        "comparison_note": (
            "All three meta-models use identical C=0.3 and class_weight=None; "
            "the No-SA versus NIR+SWIR comparison therefore isolates SA features."
        ),
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    write_metadata()
    base.inner_oof_and_outer_predictions = (
        inner_oof_and_outer_predictions_no_resampling
    )
    fixed.evaluate_fold = evaluate_fold
    fixed.main()

    headline_path = OUT / "headline_results.json"
    if headline_path.exists():
        headline = json.loads(headline_path.read_text(encoding="utf-8"))
        headline["fixed_configuration"].update(
            {
                "resampling": "none",
                "xgboost_device": "cuda",
                "matched_meta_settings_for_all_comparisons": True,
            }
        )
        headline_path.write_text(
            json.dumps(headline, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
