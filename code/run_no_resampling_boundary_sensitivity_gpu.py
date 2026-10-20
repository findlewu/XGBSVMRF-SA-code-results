# -*- coding: utf-8 -*-
"""GPU-assisted repeated validation without class resampling and SA-boundary sensitivity.

Fixed settings inherited from the preceding analysis:
* primary cohort: 40 complete pots / 833 spectra;
* outer validation: 10 repeats x 5 pot-grouped folds;
* stacking OOF generation: 3 pot-grouped inner folds;
* preprocessing: S-G smoothing followed by the second derivative;
* base learners: unchanged XGBoost (CUDA), SVM and RF;
* meta learner: scaled logistic regression, C=0.3, class_weight=None;
* no grade-0/1 threshold;
* no undersampling, SMOTE or class weighting in the base learners.

The boundary variants change only the NIR lower bound and/or the NIR/SWIR
split. Base-learner OOF probabilities are computed once per outer fold and
reused for every boundary configuration. Checkpoints permit safe resume.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import run_repeated_grouped_sa_ablation as base


OUT = base.ROOT / "work" / "no_resampling_boundary_sensitivity_gpu"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(20260929, 20260939))
BOUNDARIES = {
    "NIR740_split1300": (740.0, 1300.0),
    "NIR760_split1280": (760.0, 1280.0),
    "NIR760_split1300": (760.0, 1300.0),
    "NIR760_split1320": (760.0, 1320.0),
    "NIR780_split1300": (780.0, 1300.0),
    "NIR740_split1280": (740.0, 1280.0),
    "NIR780_split1320": (780.0, 1320.0),
}
REFERENCE = "NIR760_split1300"
MODEL_ORDER = ["No-SA no-resampling", *BOUNDARIES.keys()]


def spectral_angle_for_boundary(X, centres, wavelengths, nir_low, split):
    blocks = []
    for low, high in ((nir_low, split), (split, 2500.0)):
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


def fixed_meta_predict(train_features, y_train, test_features, seed):
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        C=0.3,
        max_iter=3000,
        class_weight=None,
        random_state=seed,
    )
    model.fit(scaler.transform(train_features), y_train)
    return model.predict(scaler.transform(test_features))


def inner_oof_no_resampling(X_train, y_train, groups_train, X_test, seed):
    names = ["xgb", "svm", "rf"]
    inner = StratifiedGroupKFold(
        n_splits=base.INNER_FOLDS, shuffle=True, random_state=seed
    )
    oof = {name: np.zeros((len(y_train), 6), dtype=float) for name in names}
    for inner_fold, (fit_idx, val_idx) in enumerate(
        inner.split(X_train, y_train, groups_train), start=1
    ):
        fit_seed = seed + inner_fold
        for name in names:
            model = base.fit_base(name, X_train[fit_idx], y_train[fit_idx], fit_seed)
            oof[name][val_idx] = model.predict_proba(X_train[val_idx])
            del model
        gc.collect()

    test = {}
    fit_seed = seed + 100
    for name in names:
        model = base.fit_base(name, X_train, y_train, fit_seed)
        test[name] = model.predict_proba(X_test)
        del model
    gc.collect()
    return oof, test


def evaluate_fold(X, y, groups, meta, wavelengths, repeat_seed, fold, train_idx, test_idx):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    model_seed = repeat_seed + fold * 1000

    oof, test = inner_oof_no_resampling(
        X_train, y_train, groups_train, X_test, model_seed
    )
    base_train = np.hstack([oof["xgb"], oof["svm"], oof["rf"]])
    base_test = np.hstack([test["xgb"], test["svm"], test["rf"]])
    predictions = {
        "No-SA no-resampling": fixed_meta_predict(
            base_train, y_train, base_test, model_seed
        )
    }

    centres = base.group_balanced_centres(X_train, y_train, groups_train)
    for label, (nir_low, split) in BOUNDARIES.items():
        sa_train = spectral_angle_for_boundary(
            X_train, centres, wavelengths, nir_low, split
        )
        sa_test = spectral_angle_for_boundary(
            X_test, centres, wavelengths, nir_low, split
        )
        predictions[label] = fixed_meta_predict(
            np.hstack([base_train, sa_train]),
            y_train,
            np.hstack([base_test, sa_test]),
            model_seed,
        )

    metric_rows = []
    prediction_rows = []
    for model_name in MODEL_ORDER:
        pred = predictions[model_name]
        metric_rows.append(
            {
                "repeat_seed": repeat_seed,
                "fold": fold,
                "model": model_name,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "n_test_spectra": len(test_idx),
                "n_test_pots": len(np.unique(groups[test_idx])),
            }
        )
        for local_index, predicted_grade in enumerate(pred):
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


def bootstrap_ci(values, seed, n_boot=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(samples, [2.5, 97.5])


def paired_row(left, right, metric, left_name, right_name, seed):
    aligned_left, aligned_right = left.align(right, join="inner")
    difference = aligned_left - aligned_right
    try:
        statistic, pvalue = wilcoxon(difference, zero_method="wilcox")
    except ValueError:
        statistic, pvalue = 0.0, 1.0
    ci = bootstrap_ci(difference.to_numpy(), seed)
    return {
        "metric": metric,
        "left": left_name,
        "right": right_name,
        "n_paired_repeats": len(difference),
        "mean_difference": difference.mean(),
        "sd_difference": difference.std(ddof=1),
        "split_stability_ci_low": ci[0],
        "split_stability_ci_high": ci[1],
        "wilcoxon_W": statistic,
        "two_sided_p": pvalue,
        "repeats_left_higher": int((difference > 0).sum()),
        "repeats_equal": int((difference == 0).sum()),
        "repeats_left_lower": int((difference < 0).sum()),
    }


def summarize(metrics, predictions):
    per_repeat = (
        metrics.groupby(["repeat_seed", "model"], as_index=False)[
            ["accuracy", "macro_f1"]
        ]
        .mean()
    )
    per_repeat.to_csv(OUT / "per_repeat_metrics.csv", index=False)

    summary_rows = []
    for model_name, group in per_repeat.groupby("model"):
        accuracy_ci = bootstrap_ci(group["accuracy"], SEEDS[0])
        macro_ci = bootstrap_ci(group["macro_f1"], SEEDS[0] + 1)
        summary_rows.append(
            {
                "model": model_name,
                "n_repeats": group["repeat_seed"].nunique(),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_sd_across_repeats": group["accuracy"].std(ddof=1),
                "accuracy_split_stability_ci_low": accuracy_ci[0],
                "accuracy_split_stability_ci_high": accuracy_ci[1],
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_sd_across_repeats": group["macro_f1"].std(ddof=1),
                "macro_f1_split_stability_ci_low": macro_ci[0],
                "macro_f1_split_stability_ci_high": macro_ci[1],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(OUT / "repeated_summary.csv", index=False)

    indexed = {
        name: per_repeat[per_repeat["model"].eq(name)].set_index("repeat_seed")
        for name in MODEL_ORDER
    }
    comparisons = []
    for metric_index, metric in enumerate(["accuracy", "macro_f1"]):
        comparisons.append(
            paired_row(
                indexed[REFERENCE][metric],
                indexed["No-SA no-resampling"][metric],
                metric,
                REFERENCE,
                "No-SA no-resampling",
                SEEDS[0] + 10 + metric_index,
            )
        )
        for variant_index, variant in enumerate(BOUNDARIES):
            if variant == REFERENCE:
                continue
            comparisons.append(
                paired_row(
                    indexed[variant][metric],
                    indexed[REFERENCE][metric],
                    metric,
                    variant,
                    REFERENCE,
                    SEEDS[0] + 20 + variant_index * 2 + metric_index,
                )
            )
    comparisons_frame = pd.DataFrame(comparisons)
    comparisons_frame.to_csv(OUT / "paired_comparisons.csv", index=False)

    historical_path = base.ROOT / "work" / "fixed_sa_repeated_validation" / "per_repeat_metrics.csv"
    historical_rows = []
    if historical_path.exists():
        historical = pd.read_csv(historical_path)
        historical = historical[historical["model"].eq("NIR+SWIR SA fixed")].set_index("repeat_seed")
        new_fixed = indexed[REFERENCE]
        for metric_index, metric in enumerate(["accuracy", "macro_f1"]):
            historical_rows.append(
                paired_row(
                    new_fixed[metric],
                    historical[metric],
                    metric,
                    f"{REFERENCE} no-resampling",
                    "NIR+SWIR under+SMOTE historical",
                    SEEDS[0] + 100 + metric_index,
                )
            )
    pd.DataFrame(historical_rows).to_csv(
        OUT / "paired_vs_historical_under_smote.csv", index=False
    )

    class_rows = []
    for (repeat_seed, model_name), group in predictions.groupby(["repeat_seed", "model"]):
        report = classification_report(
            group["true_grade"], group["predicted_grade"],
            labels=range(6), output_dict=True, zero_division=0
        )
        for grade in range(6):
            class_rows.append(
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
    class_frame = pd.DataFrame(class_rows)
    class_frame.to_csv(OUT / "per_repeat_per_class.csv", index=False)
    class_summary = (
        class_frame.groupby(["model", "grade"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            support_per_repeat=("support", "first"),
        )
    )
    class_summary.to_csv(OUT / "per_class_summary.csv", index=False)

    headline = {
        "fixed_settings": {
            "outer_folds": base.OUTER_FOLDS,
            "inner_folds": base.INNER_FOLDS,
            "seeds": SEEDS,
            "resampling": None,
            "meta_C": 0.3,
            "meta_class_weight": None,
            "grade01_threshold": None,
            "xgboost_device": "cuda",
        },
        "boundaries": BOUNDARIES,
        "summary": summary.to_dict("records"),
        "comparisons": comparisons_frame.to_dict("records"),
        "historical_comparisons": historical_rows,
        "note": (
            "Intervals describe split stability across ten random grouped-CV partitions "
            "of the same 40 pots; they are not independent-cohort confidence intervals."
        ),
    }
    (OUT / "headline_results.json").write_text(
        json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    start = time.time()
    raw, meta_all, wavelengths, files = base.load_data()
    pots = base.complete_pots(meta_all)
    mask = meta_all["pot_id"].astype(str).isin(pots).to_numpy()
    X = base.preprocess(raw)[mask]
    meta = meta_all.loc[mask].reset_index(drop=True)
    y = meta["stage"].to_numpy(dtype=int)
    groups = meta["pot_id"].astype(str).to_numpy()
    print(
        f"Dataset: {len(y)} spectra, {len(np.unique(groups))} pots, "
        f"grades={np.bincount(y).tolist()}, source_files={len(files)}",
        flush=True,
    )
    print(
        "Fixed: no resampling; NIR+SWIR SA; meta C=0.3; no class weight; "
        "no grade-0/1 threshold; XGBoost device=cuda",
        flush=True,
    )
    print(f"Boundary variants: {BOUNDARIES}", flush=True)

    metric_frames = []
    prediction_frames = []
    for repeat_number, repeat_seed in enumerate(SEEDS, start=1):
        outer = StratifiedGroupKFold(
            n_splits=base.OUTER_FOLDS, shuffle=True, random_state=repeat_seed
        )
        for fold, (train_idx, test_idx) in enumerate(
            outer.split(X, y, groups), start=1
        ):
            metric_path = OUT / f"checkpoint_seed_{repeat_seed}_fold_{fold}_metrics.csv"
            prediction_path = OUT / f"checkpoint_seed_{repeat_seed}_fold_{fold}_predictions.csv"
            if metric_path.exists() and prediction_path.exists():
                fold_metrics = pd.read_csv(metric_path)
                fold_predictions = pd.read_csv(prediction_path)
                expected_models = set(MODEL_ORDER)
                if set(fold_metrics["model"]) == expected_models:
                    metric_frames.append(fold_metrics)
                    prediction_frames.append(fold_predictions)
                    print(
                        f"[{repeat_number}/10 fold {fold}/5] checkpoint reused",
                        flush=True,
                    )
                    continue

            fold_start = time.time()
            fold_metrics, fold_predictions = evaluate_fold(
                X, y, groups, meta, wavelengths,
                repeat_seed, fold, train_idx, test_idx
            )
            fold_metrics.to_csv(metric_path, index=False)
            fold_predictions.to_csv(prediction_path, index=False)
            metric_frames.append(fold_metrics)
            prediction_frames.append(fold_predictions)
            fixed_f1 = fold_metrics.loc[
                fold_metrics["model"].eq(REFERENCE), "macro_f1"
            ].iloc[0]
            no_sa_f1 = fold_metrics.loc[
                fold_metrics["model"].eq("No-SA no-resampling"), "macro_f1"
            ].iloc[0]
            print(
                f"[{repeat_number}/10 fold {fold}/5] "
                f"fixed-SA F1={fixed_f1:.4f}; no-SA F1={no_sa_f1:.4f}; "
                f"{time.time() - fold_start:.1f}s",
                flush=True,
            )

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(OUT / "all_fold_metrics.csv", index=False)
    predictions.to_csv(OUT / "all_out_of_fold_predictions.csv", index=False)
    summarize(metrics, predictions)

    print(f"Completed in {(time.time() - start) / 60:.1f} min", flush=True)
    print(OUT / "repeated_summary.csv", flush=True)
    print(OUT / "paired_comparisons.csv", flush=True)
    print(OUT / "paired_vs_historical_under_smote.csv", flush=True)


if __name__ == "__main__":
    main()
