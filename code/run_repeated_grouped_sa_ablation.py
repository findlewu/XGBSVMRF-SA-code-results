# -*- coding: utf-8 -*-
"""Repeated pot-grouped validation and regional-SA ablation.

The experiment deliberately excludes the historical grade-0/1 threshold.
Every learned operation is fitted using training data only:

* outer evaluation: 5-fold StratifiedGroupKFold by pot;
* inner stacking features: 3-fold StratifiedGroupKFold by pot;
* class resampling: grade-0 undersampling followed by SMOTE in each fit;
* spectral-angle centres: group-balanced centres from the corresponding
  training fold only;
* meta learner: scaled logistic regression fitted to inner OOF predictions.

The script is checkpointed by repeat and outer fold, so an interrupted run can
resume without discarding completed work.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.signal import savgol_filter
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw_spectra"
OUT_DIR = ROOT / "work" / "repeated_grouped_sa_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTER_FOLDS = 5
INNER_FOLDS = 3
DEFAULT_SEEDS = list(range(20260929, 20260939))
REGIONS = [
    ("VIS", 350, 680),
    ("Red_edge", 680, 760),
    ("NIR", 760, 1300),
    ("SWIR", 1300, 2500),
]


def load_data():
    files = sorted(
        p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".csv", ".xls"}
    )
    if not files:
        raise FileNotFoundError(f"No source files found in {DATA_DIR}")
    metadata = []
    spectra = []
    wavelengths = None
    for path in files:
        # The single .xls file in the source directory contains CSV text.
        frame = pd.read_csv(path, encoding="utf-8-sig")
        wave = np.asarray([float(c) for c in frame.columns[3:]], dtype=float)
        if wavelengths is None:
            wavelengths = wave
        elif not np.array_equal(wavelengths, wave):
            raise ValueError(f"Wavelength mismatch: {path.name}")
        code = path.stem.split("_")[1]
        meta = frame[["Date", "stage", "Filename"]].copy()
        meta["source_file"] = path.name
        meta["pot_id"] = code[:2]
        meta["position_code"] = code[2:]
        meta["row_in_file"] = np.arange(len(frame))
        metadata.append(meta)
        spectra.append(frame.iloc[:, 3:].to_numpy(dtype=np.float64))
    meta = pd.concat(metadata, ignore_index=True)
    meta["Date"] = meta["Date"].astype(str)
    meta["stage"] = pd.to_numeric(meta["stage"], errors="raise").astype(int)
    return np.vstack(spectra), meta, wavelengths, files


def complete_pots(meta):
    counts = pd.crosstab(meta["pot_id"], meta["stage"])
    for grade in range(6):
        if grade not in counts.columns:
            counts[grade] = 0
    return counts.index[(counts.loc[:, range(6)] > 0).all(axis=1)].astype(str)


def preprocess(raw):
    smooth = savgol_filter(raw, window_length=9, polyorder=2, axis=1)
    return savgol_filter(
        smooth, window_length=9, polyorder=2, deriv=2, delta=1.0, axis=1
    )


def resample_training(X, y, seed):
    rng = np.random.RandomState(seed)
    counts = np.bincount(y, minlength=6)
    target_grade0 = int(max(counts[1:]))
    grade0 = np.where(y == 0)[0]
    keep_grade0 = rng.choice(
        grade0, min(target_grade0, len(grade0)), replace=False
    )
    keep = np.concatenate([keep_grade0, np.where(y != 0)[0]])
    keep.sort()
    return SMOTE(random_state=seed, k_neighbors=5).fit_resample(X[keep], y[keep])


def group_balanced_centres(X, y, groups):
    centres = []
    for grade in range(6):
        pot_means = [
            X[(y == grade) & (groups == pot)].mean(axis=0)
            for pot in np.unique(groups[y == grade])
        ]
        centres.append(np.mean(pot_means, axis=0))
    return np.asarray(centres)


def spectral_angle_features(X, centres, wavelengths):
    blocks = []
    for _, low, high in REGIONS:
        mask = (wavelengths >= low) & (wavelengths <= high)
        A = X[:, mask]
        C = centres[:, mask]
        numerator = A @ C.T
        denominator = (
            np.linalg.norm(A, axis=1)[:, None]
            * np.linalg.norm(C, axis=1)[None, :]
            + 1e-12
        )
        blocks.append(np.arccos(np.clip(numerator / denominator, -1.0, 1.0)))
    return np.hstack(blocks)


def make_model(name, seed):
    if name == "xgb":
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            tree_method="hist",
            device="cuda",
            eval_metric="mlogloss",
            n_jobs=4,
        )
    if name == "svm":
        return SVC(
            C=100,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(name)


def fit_base(name, X, y, seed):
    model = make_model(name, seed)
    model.fit(X, y)
    return model


def inner_oof_and_outer_predictions(Xtr, ytr, gtr, Xte, wave, seed):
    inner = StratifiedGroupKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=seed
    )
    names = ["xgb", "svm", "rf"]
    oof = {name: np.zeros((len(ytr), 6), dtype=float) for name in names}
    sa_oof = np.zeros((len(ytr), 24), dtype=float)
    for inner_fold, (fit_idx, val_idx) in enumerate(
        inner.split(Xtr, ytr, gtr), start=1
    ):
        fit_seed = seed + inner_fold
        Xi, yi, gi = Xtr[fit_idx], ytr[fit_idx], gtr[fit_idx]
        Xb, yb = resample_training(Xi, yi, fit_seed)
        centres = group_balanced_centres(Xi, yi, gi)
        sa_oof[val_idx] = spectral_angle_features(Xtr[val_idx], centres, wave)
        for name in names:
            model = fit_base(name, Xb, yb, fit_seed)
            oof[name][val_idx] = model.predict_proba(Xtr[val_idx])
            del model
        gc.collect()

    fit_seed = seed + 100
    Xb, yb = resample_training(Xtr, ytr, fit_seed)
    centres = group_balanced_centres(Xtr, ytr, gtr)
    sa_test = spectral_angle_features(Xte, centres, wave)
    test = {}
    for name in names:
        model = fit_base(name, Xb, yb, fit_seed)
        test[name] = model.predict_proba(Xte)
        del model
    gc.collect()
    return oof, test, sa_oof, sa_test


def meta_predict(train_features, ytr, test_features, seed):
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        C=0.01,
        max_iter=3000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(scaler.transform(train_features), ytr)
    return model.predict(scaler.transform(test_features))


def sa_configurations():
    region_cols = {
        name: np.arange(index * 6, (index + 1) * 6)
        for index, (name, _, _) in enumerate(REGIONS)
    }
    configs = {
        "SA_all": np.arange(24),
        "SA_VIS_only": region_cols["VIS"],
        "SA_Red_edge_only": region_cols["Red_edge"],
        "SA_NIR_only": region_cols["NIR"],
        "SA_SWIR_only": region_cols["SWIR"],
        "SA_VIS+Red_edge": np.concatenate(
            [region_cols["VIS"], region_cols["Red_edge"]]
        ),
        "SA_VIS+Red_edge+NIR": np.concatenate(
            [region_cols["VIS"], region_cols["Red_edge"], region_cols["NIR"]]
        ),
    }
    for name in region_cols:
        configs[f"SA_all_minus_{name}"] = np.setdiff1d(
            np.arange(24), region_cols[name]
        )
    return configs


def evaluate_fold(X, y, groups, meta, wave, repeat_seed, fold, train_idx, test_idx):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    gtr, gte = groups[train_idx], groups[test_idx]
    model_seed = repeat_seed + fold * 1000
    oof, test, sa_oof, sa_test = inner_oof_and_outer_predictions(
        Xtr, ytr, gtr, Xte, wave, model_seed
    )

    predictions = {
        "SVM": test["svm"].argmax(axis=1),
        "RF": test["rf"].argmax(axis=1),
        "XGBoost": test["xgb"].argmax(axis=1),
    }
    xgb_svm_train = np.hstack([oof["xgb"], oof["svm"]])
    xgb_svm_test = np.hstack([test["xgb"], test["svm"]])
    predictions["XGB-SVM"] = meta_predict(
        xgb_svm_train, ytr, xgb_svm_test, model_seed
    )
    base_train = np.hstack([oof["xgb"], oof["svm"], oof["rf"]])
    base_test = np.hstack([test["xgb"], test["svm"], test["rf"]])
    predictions["XGB-SVM-RF"] = meta_predict(
        base_train, ytr, base_test, model_seed
    )
    for config, columns in sa_configurations().items():
        predictions[config] = meta_predict(
            np.hstack([base_train, sa_oof[:, columns]]),
            ytr,
            np.hstack([base_test, sa_test[:, columns]]),
            model_seed,
        )

    metric_rows = []
    prediction_rows = []
    for model_name, pred in predictions.items():
        metric_rows.append(
            {
                "repeat_seed": repeat_seed,
                "fold": fold,
                "model": model_name,
                "accuracy": accuracy_score(yte, pred),
                "macro_f1": f1_score(yte, pred, average="macro"),
                "n_test_spectra": len(test_idx),
                "n_test_pots": len(np.unique(gte)),
            }
        )
        for local_index, predicted in enumerate(pred):
            source_index = int(test_idx[local_index])
            prediction_rows.append(
                {
                    "repeat_seed": repeat_seed,
                    "fold": fold,
                    "model": model_name,
                    "row_index": source_index,
                    "true_grade": int(yte[local_index]),
                    "predicted_grade": int(predicted),
                    "pot_id": str(groups[source_index]),
                    "source_file": str(meta.iloc[source_index]["source_file"]),
                    "date": str(meta.iloc[source_index]["Date"]),
                    "position_code": str(
                        meta.iloc[source_index]["position_code"]
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def bootstrap_mean_ci(values, seed=20260929, n_boot=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 97.5])


def summarise(metrics, predictions, seeds):
    per_repeat = (
        metrics.groupby(["repeat_seed", "model"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"))
    )
    per_repeat.to_csv(OUT_DIR / "per_repeat_model_metrics.csv", index=False)

    summary_rows = []
    for model_name, group in per_repeat.groupby("model"):
        acc_ci = bootstrap_mean_ci(group["accuracy"], seed=seeds[0])
        f1_ci = bootstrap_mean_ci(group["macro_f1"], seed=seeds[0] + 1)
        summary_rows.append(
            {
                "model": model_name,
                "n_repeats": group["repeat_seed"].nunique(),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_sd_across_repeats": group["accuracy"].std(ddof=1),
                "accuracy_resampling_ci_low": acc_ci[0],
                "accuracy_resampling_ci_high": acc_ci[1],
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_sd_across_repeats": group["macro_f1"].std(ddof=1),
                "macro_f1_resampling_ci_low": f1_ci[0],
                "macro_f1_resampling_ci_high": f1_ci[1],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1_mean", "accuracy_mean"], ascending=False
    )
    summary.to_csv(OUT_DIR / "repeated_grouped_summary.csv", index=False)

    baseline = (
        per_repeat[per_repeat["model"].eq("XGB-SVM-RF")]
        .set_index("repeat_seed")["macro_f1"]
        .sort_index()
    )
    comparisons = []
    for model_name, group in per_repeat.groupby("model"):
        if model_name in {"SVM", "RF", "XGBoost", "XGB-SVM", "XGB-SVM-RF"}:
            continue
        candidate = group.set_index("repeat_seed")["macro_f1"].sort_index()
        candidate, matched_baseline = candidate.align(baseline, join="inner")
        difference = candidate - matched_baseline
        try:
            statistic, pvalue = wilcoxon(
                difference,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
        except ValueError:
            statistic, pvalue = 0.0, 1.0
        ci = bootstrap_mean_ci(difference, seed=seeds[0] + 2)
        comparisons.append(
            {
                "candidate": model_name,
                "reference": "XGB-SVM-RF",
                "n_repeats": len(difference),
                "macro_f1_mean_difference": difference.mean(),
                "macro_f1_sd_difference": difference.std(ddof=1),
                "resampling_ci_low": ci[0],
                "resampling_ci_high": ci[1],
                "wilcoxon_W": statistic,
                "wilcoxon_p": pvalue,
                "repeats_candidate_higher": int((difference > 0).sum()),
                "repeats_equal": int((difference == 0).sum()),
                "repeats_candidate_lower": int((difference < 0).sum()),
            }
        )
    comparison_frame = pd.DataFrame(comparisons).sort_values(
        "macro_f1_mean_difference", ascending=False
    )
    comparison_frame.to_csv(OUT_DIR / "sa_paired_repeat_comparisons.csv", index=False)

    per_class_rows = []
    for (repeat_seed, model_name), group in predictions.groupby(
        ["repeat_seed", "model"]
    ):
        report = classification_report(
            group["true_grade"],
            group["predicted_grade"],
            labels=range(6),
            output_dict=True,
            zero_division=0,
        )
        for grade in range(6):
            per_class_rows.append(
                {
                    "repeat_seed": repeat_seed,
                    "model": model_name,
                    "grade": grade,
                    "precision": report[str(grade)]["precision"],
                    "recall": report[str(grade)]["recall"],
                    "f1": report[str(grade)]["f1-score"],
                    "support": report[str(grade)]["support"],
                }
            )
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(OUT_DIR / "per_repeat_per_class_metrics.csv", index=False)
    (
        per_class.groupby(["model", "grade"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            support_per_repeat=("support", "first"),
        )
        .to_csv(OUT_DIR / "per_class_summary.csv", index=False)
    )

    for model_name in ["XGB-SVM-RF", "SA_all"]:
        aggregate = np.zeros((6, 6), dtype=int)
        for repeat_seed, group in predictions[
            predictions["model"].eq(model_name)
        ].groupby("repeat_seed"):
            aggregate += confusion_matrix(
                group["true_grade"], group["predicted_grade"], labels=range(6)
            )
        matrix = pd.DataFrame(
            aggregate,
            index=[f"true_{grade}" for grade in range(6)],
            columns=[f"pred_{grade}" for grade in range(6)],
        )
        matrix.to_csv(OUT_DIR / f"confusion_sum_{model_name}.csv")

    result = {
        "primary_comparison": comparison_frame[
            comparison_frame["candidate"].eq("SA_all")
        ].to_dict("records")[0],
        "best_configuration": summary.iloc[0].to_dict(),
        "seeds": seeds,
        "note": (
            "Bootstrap intervals describe split-seed stability on the same 40 pots; "
            "they are not confidence intervals for independent biological cohorts."
        ),
    }
    (OUT_DIR / "headline_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated outer-CV random seeds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    start = time.time()
    raw, meta_all, wave, source_files = load_data()
    all_complete_pots = complete_pots(meta_all)
    mask = meta_all["pot_id"].astype(str).isin(all_complete_pots).to_numpy()
    X = preprocess(raw)[mask]
    meta = meta_all.loc[mask].reset_index(drop=True)
    y = meta["stage"].to_numpy(dtype=int)
    groups = meta["pot_id"].astype(str).to_numpy()
    print(
        f"Dataset: {len(y)} spectra, {len(np.unique(groups))} complete pots, "
        f"{len(wave)} wavelengths, grades={np.bincount(y).tolist()}"
    )
    print(f"Source files: {len(source_files)}; repeats: {len(seeds)}")

    all_metrics = []
    all_predictions = []
    for repeat_number, repeat_seed in enumerate(seeds, start=1):
        outer = StratifiedGroupKFold(
            n_splits=OUTER_FOLDS, shuffle=True, random_state=repeat_seed
        )
        for fold, (train_idx, test_idx) in enumerate(
            outer.split(X, y, groups), start=1
        ):
            metric_path = OUT_DIR / f"checkpoint_seed_{repeat_seed}_fold_{fold}_metrics.csv"
            prediction_path = OUT_DIR / f"checkpoint_seed_{repeat_seed}_fold_{fold}_predictions.csv"
            if metric_path.exists() and prediction_path.exists():
                metrics = pd.read_csv(metric_path)
                predictions = pd.read_csv(
                    prediction_path,
                    dtype={"pot_id": str, "position_code": str},
                )
                print(
                    f"repeat {repeat_number}/{len(seeds)} seed={repeat_seed} "
                    f"fold={fold}: loaded checkpoint"
                )
            else:
                tic = time.time()
                metrics, predictions = evaluate_fold(
                    X,
                    y,
                    groups,
                    meta,
                    wave,
                    repeat_seed,
                    fold,
                    train_idx,
                    test_idx,
                )
                metrics.to_csv(metric_path, index=False)
                predictions.to_csv(prediction_path, index=False)
                base_f1 = metrics.loc[
                    metrics["model"].eq("XGB-SVM-RF"), "macro_f1"
                ].iloc[0]
                sa_f1 = metrics.loc[
                    metrics["model"].eq("SA_all"), "macro_f1"
                ].iloc[0]
                print(
                    f"repeat {repeat_number}/{len(seeds)} seed={repeat_seed} "
                    f"fold={fold}: {time.time()-tic:.1f}s, "
                    f"base={base_f1:.4f}, SA_all={sa_f1:.4f}, "
                    f"delta={sa_f1-base_f1:+.4f}"
                )
            all_metrics.append(metrics)
            all_predictions.append(predictions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics.to_csv(OUT_DIR / "all_fold_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "all_out_of_fold_predictions.csv", index=False)
    summarise(metrics, predictions, seeds)
    print(f"Completed in {(time.time()-start)/60:.1f} minutes: {OUT_DIR}")


if __name__ == "__main__":
    main()
