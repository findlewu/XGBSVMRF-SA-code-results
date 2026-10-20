# -*- coding: utf-8 -*-
"""Repeated grouped validation of the pre-fixed NIR+SWIR SA configuration.

Configuration fixed before this run from the preceding nested-tuning screen:
regional SA = NIR + SWIR, meta LR C = 0.3, class_weight = None.
No grade-0/1 threshold is applied. XGB, SVM and RF settings are unchanged.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

import run_repeated_grouped_sa_ablation as base


OUT = base.ROOT / "work" / "fixed_sa_repeated_validation"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(20260929, 20260939))
NIR_SWIR_COLUMNS = np.concatenate([np.arange(12, 18), np.arange(18, 24)])
MODEL_ORDER = ["No-SA fixed", "All-region SA old", "NIR+SWIR SA fixed"]


def evaluate_fold(X, y, groups, meta, wave, repeat_seed, fold, train_idx, test_idx):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    model_seed = repeat_seed + fold * 1000
    oof, test, sa_oof, sa_test = base.inner_oof_and_outer_predictions(
        X_train, y_train, groups_train, X_test, wave, model_seed
    )
    base_train = np.hstack([oof["xgb"], oof["svm"], oof["rf"]])
    base_test = np.hstack([test["xgb"], test["svm"], test["rf"]])
    predictions = {
        "No-SA fixed": base.meta_predict(
            base_train, y_train, base_test, model_seed
        ),
        "All-region SA old": base.meta_predict(
            np.hstack([base_train, sa_oof]),
            y_train,
            np.hstack([base_test, sa_test]),
            model_seed,
        ),
    }
    # base.meta_predict uses the historical C=0.01 and balanced class weights.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_fixed = np.hstack([base_train, sa_oof[:, NIR_SWIR_COLUMNS]])
    test_fixed = np.hstack([base_test, sa_test[:, NIR_SWIR_COLUMNS]])
    scaler = StandardScaler().fit(train_fixed)
    meta_model = LogisticRegression(
        C=0.3, max_iter=3000, class_weight=None, random_state=model_seed
    )
    meta_model.fit(scaler.transform(train_fixed), y_train)
    predictions["NIR+SWIR SA fixed"] = meta_model.predict(
        scaler.transform(test_fixed)
    )

    metric_rows = []
    prediction_rows = []
    for model_name in MODEL_ORDER:
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


def bootstrap_ci(values, seed, n_boot=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.percentile(samples, [2.5, 97.5])


def summarise(metrics, predictions):
    per_repeat = (
        metrics.groupby(["repeat_seed", "model"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"))
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
    summary = pd.DataFrame(summary_rows).sort_values(
        "macro_f1_mean", ascending=False
    )
    summary.to_csv(OUT / "repeated_summary.csv", index=False)

    comparison_rows = []
    for reference in ["No-SA fixed", "All-region SA old"]:
        for metric in ["accuracy", "macro_f1"]:
            left = (
                per_repeat[per_repeat["model"].eq("NIR+SWIR SA fixed")]
                .set_index("repeat_seed")[metric]
                .sort_index()
            )
            right = (
                per_repeat[per_repeat["model"].eq(reference)]
                .set_index("repeat_seed")[metric]
                .sort_index()
            )
            left, right = left.align(right, join="inner")
            difference = left - right
            try:
                statistic, pvalue = wilcoxon(difference, zero_method="wilcox")
            except ValueError:
                statistic, pvalue = 0.0, 1.0
            ci = bootstrap_ci(difference, SEEDS[0] + 2)
            comparison_rows.append(
                {
                    "metric": metric,
                    "left": "NIR+SWIR SA fixed",
                    "right": reference,
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
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(OUT / "paired_repeat_comparisons.csv", index=False)

    class_rows = []
    for (repeat_seed, model_name), group in predictions.groupby(
        ["repeat_seed", "model"]
    ):
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
        "fixed_configuration": {
            "SA_regions": ["NIR", "SWIR"],
            "meta_C": 0.3,
            "meta_class_weight": None,
            "grade01_threshold": None,
        },
        "summary": summary.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
        "note": (
            "Split-stability bootstrap intervals use ten random split seeds on "
            "the same 40 pots and are not independent-cohort confidence intervals."
        ),
    }
    (OUT / "headline_results.json").write_text(
        json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    start = time.time()
    raw, meta_all, wave, files = base.load_data()
    pots = base.complete_pots(meta_all)
    mask = meta_all["pot_id"].astype(str).isin(pots).to_numpy()
    X = base.preprocess(raw)[mask]
    meta = meta_all.loc[mask].reset_index(drop=True)
    y = meta["stage"].to_numpy(dtype=int)
    groups = meta["pot_id"].astype(str).to_numpy()
    print(
        f"Dataset: {len(y)} spectra, {len(np.unique(groups))} pots, "
        f"grades={np.bincount(y).tolist()}, source_files={len(files)}"
    )
    print("Fixed candidate: NIR+SWIR SA, meta C=0.3, class_weight=None")

    metric_frames = []
    prediction_frames = []
    for repeat_number, repeat_seed in enumerate(SEEDS, start=1):
        outer = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=repeat_seed
        )
        for fold, (train_idx, test_idx) in enumerate(
            outer.split(X, y, groups), start=1
        ):
            metric_path = OUT / f"checkpoint_seed_{repeat_seed}_fold_{fold}_metrics.csv"
            prediction_path = OUT / f"checkpoint_seed_{repeat_seed}_fold_{fold}_predictions.csv"
            if metric_path.exists() and prediction_path.exists():
                metrics = pd.read_csv(metric_path)
                predictions = pd.read_csv(prediction_path, dtype={"pot_id": str})
                print(
                    f"repeat {repeat_number}/10 seed={repeat_seed} fold={fold}: "
                    "loaded checkpoint"
                )
            else:
                tic = time.time()
                metrics, predictions = evaluate_fold(
                    X, y, groups, meta, wave, repeat_seed, fold,
                    train_idx, test_idx
                )
                metrics.to_csv(metric_path, index=False)
                predictions.to_csv(prediction_path, index=False)
                candidate = metrics[metrics["model"].eq("NIR+SWIR SA fixed")].iloc[0]
                no_sa = metrics[metrics["model"].eq("No-SA fixed")].iloc[0]
                print(
                    f"repeat {repeat_number}/10 seed={repeat_seed} fold={fold}: "
                    f"{time.time()-tic:.1f}s; candidate={candidate['macro_f1']:.4f}, "
                    f"noSA={no_sa['macro_f1']:.4f}, "
                    f"delta={candidate['macro_f1']-no_sa['macro_f1']:+.4f}"
                )
            metric_frames.append(metrics)
            prediction_frames.append(predictions)
            gc.collect()

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(OUT / "all_fold_metrics.csv", index=False)
    predictions.to_csv(OUT / "all_out_of_fold_predictions.csv", index=False)
    summarise(metrics, predictions)
    print(f"Completed in {(time.time()-start)/60:.1f} minutes. Outputs: {OUT}")


if __name__ == "__main__":
    main()
