# -*- coding: utf-8 -*-
"""Frozen, paired group learning curves; no resampling or hyperparameter search.

Run with a Python environment that satisfies requirements.txt:
  python -X utf8 -u <this_file>
Use --validate-only to check cohort, nested subsets, splits and CUDA first.
All new outputs are isolated from historical experiments and the manuscript.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import run_repeated_grouped_sa_ablation as base


OUTER_SEED = 20260929
SUBSET_SEED = 20261006
BUDGETS = (8, 16, 24, 32)
SUBSET_REPEATS = 5
MODELS = ("XGB-SVM-RF", "XGBSVMRF-SA (NIR+SWIR)")
OUT = base.ROOT / "work" / "paired_grouped_learning_curve_gpu_20261006"
DELIVERY = base.ROOT / "work" / "learning_curve_outputs"
REFERENCE = base.ROOT / "work" / "final_common_grouped_fivefold_gpu"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_write(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def metric_values(y, prediction, prefix=""):
    return {
        prefix + "accuracy": float(accuracy_score(y, prediction)),
        prefix + "macro_f1": float(f1_score(y, prediction, labels=range(6), average="macro", zero_division=0)),
    }


def load_cohort():
    raw, meta_all, wave, files = base.load_data()
    complete = base.complete_pots(meta_all)
    mask = meta_all["pot_id"].astype(str).isin(complete).to_numpy()
    X = base.preprocess(raw)[mask]
    meta = meta_all.loc[mask].reset_index(drop=True)
    y = meta["stage"].to_numpy(dtype=int)
    groups = meta["pot_id"].astype(str).to_numpy()
    assert X.shape == (833, 2151), X.shape
    assert len(np.unique(groups)) == 40
    assert np.isfinite(X).all()
    assert np.array_equal(np.bincount(y, minlength=6), [223, 134, 119, 119, 119, 119])
    return X, y, groups, meta, wave, files


def make_plan(X, y, groups):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    folds = {}
    tasks = []
    audit = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), start=1):
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])
        assert len(train_groups) == 32 and len(test_groups) == 8
        assert set(train_groups).isdisjoint(test_groups)
        folds[fold] = (train_idx, test_idx)
        permutations = {
            repeat: np.random.RandomState(SUBSET_SEED + fold * 1000 + repeat).permutation(train_groups)
            for repeat in range(1, SUBSET_REPEATS + 1)
        }
        for budget in BUDGETS:
            repeats = [0] if budget == 32 else range(1, SUBSET_REPEATS + 1)
            for repeat in repeats:
                selected = train_groups if budget == 32 else permutations[repeat][:budget]
                # Preserve original row order to match the established pipeline.
                selected_idx = train_idx[np.isin(groups[train_idx], selected)]
                assert len(np.unique(groups[selected_idx])) == budget
                assert np.array_equal(np.unique(y[selected_idx]), np.arange(6))
                model_seed = OUTER_SEED + fold * 1000
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=model_seed)
                for inner_fold, (fit, val) in enumerate(inner.split(X[selected_idx], y[selected_idx], groups[selected_idx]), start=1):
                    fit_groups = set(groups[selected_idx][fit])
                    val_groups = set(groups[selected_idx][val])
                    assert fit_groups.isdisjoint(val_groups)
                    assert fit_groups.isdisjoint(test_groups)
                    assert val_groups.isdisjoint(test_groups)
                    assert np.array_equal(np.unique(y[selected_idx][fit]), np.arange(6))
                    audit.append({
                        "fold": fold, "train_groups": budget, "subset_repeat": repeat,
                        "inner_fold": inner_fold, "fit_group_ids": sorted(fit_groups),
                        "validation_group_ids": sorted(val_groups), "outer_test_group_ids": test_groups.tolist(),
                        "fit_n_spectra": len(fit), "validation_n_spectra": len(val),
                    })
                tasks.append({
                    "task_id": f"fold_{fold}_groups_{budget}_subset_{repeat}",
                    "fold": fold, "train_groups": budget, "subset_repeat": repeat,
                    "subset_seed": None if repeat == 0 else SUBSET_SEED + fold * 1000 + repeat,
                    "model_seed": model_seed, "train_group_ids": sorted(selected.tolist()),
                    "test_group_ids": test_groups.tolist(), "train_row_indices": selected_idx.tolist(),
                    "test_row_indices": test_idx.tolist(),
                })
        for repeat in range(1, SUBSET_REPEATS + 1):
            subsets = [set(t["train_group_ids"]) for t in tasks if t["fold"] == fold and t["subset_repeat"] == repeat]
            assert len(subsets) == 3 and subsets[0] < subsets[1] < subsets[2] < set(train_groups)
    # Reproduce all full-size endpoints first; then finish each smaller budget.
    tasks.sort(key=lambda t: (0 if t["train_groups"] == 32 else t["train_groups"], t["fold"], t["subset_repeat"]))
    assert len(tasks) == 80
    return tasks, folds, audit


def reference_predictions(meta, y, groups, folds):
    pred = pd.read_csv(REFERENCE / "all_out_of_fold_predictions.csv", dtype={"pot_id": str, "date": str})
    pred = pred[pred["model"].isin(MODELS)].copy()
    assert len(pred) == 2 * len(y)
    for model_name in MODELS:
        for fold, (_, test_idx) in folds.items():
            rows = pred[(pred["model"] == model_name) & (pred["fold"] == fold)].sort_values("row_index")
            assert np.array_equal(rows["row_index"].to_numpy(), test_idx)
            assert np.array_equal(rows["true_grade"].to_numpy(), y[test_idx])
            assert np.array_equal(rows["pot_id"].to_numpy(), groups[test_idx])
            assert np.array_equal(rows["source_file"].to_numpy(), meta.loc[test_idx, "source_file"].to_numpy())
            assert np.array_equal(rows["date"].to_numpy(), meta.loc[test_idx, "Date"].astype(str).to_numpy())
    return pred


def assert_cuda(model):
    device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    if not device.startswith("cuda"):
        raise RuntimeError(f"XGBoost silently fell back from GPU to {device}.")
    return device


def probe_cuda(X, y):
    if not xgboost.build_info().get("USE_CUDA", False):
        raise RuntimeError("The existing XGBoost build does not support CUDA.")
    # GPU probe only: not a candidate, never included in result tables.
    rows = np.concatenate([np.flatnonzero(y == grade)[:12] for grade in range(6)])
    probe = base.make_model("xgb", OUTER_SEED)
    probe.set_params(n_estimators=2)
    probe.fit(X[rows], y[rows])
    device = assert_cuda(probe)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    return {"device_verified_from_fitted_booster": device, "hardware": gpu}


def fit_shared_features(Xtr, ytr, gtr, Xte, wave, seed):
    """Same no-resampling algorithm as the final model, plus apparent train scores."""
    names = ("xgb", "svm", "rf")
    oof = {name: np.zeros((len(ytr), 6), dtype=float) for name in names}
    sa_oof = np.zeros((len(ytr), 24), dtype=float)
    seen = np.zeros(len(ytr), dtype=int)
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for inner_fold, (fit, val) in enumerate(inner.split(Xtr, ytr, gtr), start=1):
        assert set(gtr[fit]).isdisjoint(gtr[val])
        centres = base.group_balanced_centres(Xtr[fit], ytr[fit], gtr[fit])
        sa_oof[val] = base.spectral_angle_features(Xtr[val], centres, wave)
        for name in names:
            model = base.fit_base(name, Xtr[fit], ytr[fit], seed + inner_fold)
            if name == "xgb":
                assert_cuda(model)
            assert np.array_equal(model.classes_, np.arange(6))
            oof[name][val] = model.predict_proba(Xtr[val])
            del model
        seen[val] += 1
        gc.collect()
    assert np.all(seen == 1)
    centres = base.group_balanced_centres(Xtr, ytr, gtr)
    sa_test = base.spectral_angle_features(Xte, centres, wave)
    sa_apparent = base.spectral_angle_features(Xtr, centres, wave)
    test, apparent = {}, {}
    for name in names:
        model = base.fit_base(name, Xtr, ytr, seed + 100)
        if name == "xgb":
            assert_cuda(model)
        test[name] = model.predict_proba(Xte)
        apparent[name] = model.predict_proba(Xtr)
        del model
    gc.collect()
    shared = [np.hstack([block[name] for name in names]) for block in (oof, test, apparent)]
    assert all(np.isfinite(a).all() for a in shared + [sa_oof, sa_test, sa_apparent])
    return shared, (sa_oof[:, 12:24], sa_test[:, 12:24], sa_apparent[:, 12:24])


def evaluate_task(task, X, y, groups, meta, wave, reference):
    train_idx = np.array(task["train_row_indices"])
    test_idx = np.array(task["test_row_indices"])
    shared, sa = fit_shared_features(X[train_idx], y[train_idx], groups[train_idx], X[test_idx], wave, task["model_seed"])
    metric_rows, prediction_rows, comparisons = [], [], []
    for model_name in MODELS:
        features = shared if model_name == MODELS[0] else [np.hstack([b, a]) for b, a in zip(shared, sa)]
        scaler = StandardScaler().fit(features[0])
        model = LogisticRegression(C=0.3, max_iter=3000, class_weight=None, random_state=task["model_seed"])
        model.fit(scaler.transform(features[0]), y[train_idx])
        prediction = model.predict(scaler.transform(features[1]))
        apparent_prediction = model.predict(scaler.transform(features[2]))
        metric_rows.append({
            "task_id": task["task_id"], "fold": task["fold"], "train_groups": task["train_groups"],
            "subset_repeat": task["subset_repeat"], "subset_seed": task["subset_seed"],
            "model_seed": task["model_seed"], "model": model_name,
            "n_train_spectra": len(train_idx), "n_test_spectra": len(test_idx), "n_test_groups": len(np.unique(groups[test_idx])),
            **metric_values(y[test_idx], prediction, "test_"),
            **metric_values(y[train_idx], apparent_prediction, "apparent_train_"),
        })
        for row_idx, predicted in zip(test_idx, prediction):
            prediction_rows.append({
                "task_id": task["task_id"], "fold": task["fold"], "train_groups": task["train_groups"],
                "subset_repeat": task["subset_repeat"], "model": model_name, "row_index": int(row_idx),
                "true_grade": int(y[row_idx]), "predicted_grade": int(predicted), "pot_id": str(groups[row_idx]),
                "source_file": str(meta.iloc[row_idx]["source_file"]), "date": str(meta.iloc[row_idx]["Date"]),
            })
        if task["train_groups"] == 32:
            old = reference[(reference["fold"] == task["fold"]) & (reference["model"] == model_name)].sort_values("row_index")
            differences = int(np.count_nonzero(prediction != old["predicted_grade"].to_numpy()))
            comparisons.append({"fold": task["fold"], "model": model_name, "n_test_spectra": len(test_idx), "different_predictions": differences})
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), comparisons


def summarise(metrics, predictions, completed):
    # Five subset draws from the SAME test fold are not five independent tests.
    fold_means = metrics.groupby(["train_groups", "fold", "model"], as_index=False).agg(
        n_subset_draws=("subset_repeat", "size"), n_train_spectra_mean=("n_train_spectra", "mean"),
        test_accuracy=("test_accuracy", "mean"), test_macro_f1=("test_macro_f1", "mean"),
        subset_test_accuracy_sd=("test_accuracy", "std"), subset_test_macro_f1_sd=("test_macro_f1", "std"),
        apparent_train_accuracy=("apparent_train_accuracy", "mean"), apparent_train_macro_f1=("apparent_train_macro_f1", "mean"),
    )
    summary = fold_means.groupby(["train_groups", "model"], as_index=False).agg(
        n_outer_folds=("fold", "nunique"), n_train_spectra_mean=("n_train_spectra_mean", "mean"),
        test_accuracy_mean=("test_accuracy", "mean"), test_accuracy_sd_across_folds=("test_accuracy", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"), test_macro_f1_sd_across_folds=("test_macro_f1", "std"),
        apparent_train_accuracy_mean=("apparent_train_accuracy", "mean"),
        apparent_train_macro_f1_mean=("apparent_train_macro_f1", "mean"),
        mean_within_fold_subset_accuracy_sd=("subset_test_accuracy_sd", "mean"),
        mean_within_fold_subset_macro_f1_sd=("subset_test_macro_f1_sd", "mean"),
    )
    paired_rows = []
    for (budget, fold), frame in fold_means.groupby(["train_groups", "fold"]):
        matched = frame.set_index("model")
        paired_rows.append({"train_groups": budget, "fold": fold,
            "accuracy_gain_pp": 100 * (matched.loc[MODELS[1], "test_accuracy"] - matched.loc[MODELS[0], "test_accuracy"]),
            "macro_f1_gain_pp": 100 * (matched.loc[MODELS[1], "test_macro_f1"] - matched.loc[MODELS[0], "test_macro_f1"])})
    paired = pd.DataFrame(paired_rows)
    gains = paired.groupby("train_groups", as_index=False).agg(
        n_outer_folds=("fold", "nunique"), accuracy_gain_pp_mean=("accuracy_gain_pp", "mean"),
        accuracy_gain_pp_sd=("accuracy_gain_pp", "std"), macro_f1_gain_pp_mean=("macro_f1_gain_pp", "mean"),
        macro_f1_gain_pp_sd=("macro_f1_gain_pp", "std"),
        folds_macro_f1_gain_positive=("macro_f1_gain_pp", lambda a: int((a > 0).sum())),
        folds_accuracy_gain_positive=("accuracy_gain_pp", lambda a: int((a > 0).sum())),
    )
    for frame, name in [(metrics, "all_task_metrics.csv"), (predictions, "all_test_predictions.csv"),
                        (fold_means, "per_fold_subset_averages.csv"), (summary, "learning_curve_summary.csv"),
                        (paired, "paired_gain_by_fold.csv"), (gains, "paired_sa_gain_summary.csv")]:
        csv_write(frame, OUT / name)
    if completed:
        for frame, name in [(summary, "learning_curve_summary.csv"), (gains, "paired_sa_gain_summary.csv")]:
            csv_write(frame, DELIVERY / name)
    return summary, paired, gains


def per_class_summary(predictions):
    rows = []
    for (budget, fold, repeat, model), frame in predictions.groupby(["train_groups", "fold", "subset_repeat", "model"]):
        report = classification_report(frame["true_grade"], frame["predicted_grade"], labels=range(6), output_dict=True, zero_division=0)
        for grade in range(6):
            rows.append({"train_groups": budget, "fold": fold, "subset_repeat": repeat, "model": model, "grade": grade,
                "precision": report[str(grade)]["precision"], "recall": report[str(grade)]["recall"],
                "f1": report[str(grade)]["f1-score"], "support": report[str(grade)]["support"]})
    detailed = pd.DataFrame(rows)
    by_fold = detailed.groupby(["train_groups", "fold", "model", "grade"], as_index=False)[["precision", "recall", "f1", "support"]].mean()
    csv_write(detailed, OUT / "per_task_per_class_metrics.csv")
    csv_write(by_fold.groupby(["train_groups", "model", "grade"], as_index=False).agg(
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"), f1_sd_across_folds=("f1", "std")), OUT / "learning_curve_per_class_summary.csv")


def make_figures(summary, paired, gains):
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#4B5563", "#0072B2")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7))
    for col, (metric, label) in enumerate((("accuracy", "Accuracy"), ("macro_f1", "Macro F1"))):
        ax = axes[0, col]
        for model, color, marker in zip(MODELS, colors, ("s", "o")):
            rows = summary[summary["model"] == model].sort_values("train_groups")
            x = rows["train_groups"].to_numpy()
            mean = 100 * rows[f"test_{metric}_mean"].to_numpy()
            sd = 100 * rows[f"test_{metric}_sd_across_folds"].to_numpy()
            ax.errorbar(x, mean, yerr=sd, color=color, marker=marker, linewidth=1.8, capsize=4,
                label="No-SA stack" if model == MODELS[0] else "NIR+SWIR SA stack")
        ax.set_title(f"({'a' if col == 0 else 'b'}) Held-out {label}", loc="left")
        ax.set_ylabel(label + " (%)")
        ax.set_ylim(65, 100)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax = axes[1, col]
        x = gains["train_groups"].to_numpy()
        for fold, frame in paired.groupby("fold"):
            ax.plot(frame["train_groups"], frame[f"{metric}_gain_pp"], color="#A5ADB5", linewidth=0.8, alpha=0.7, zorder=1)
        ax.axhline(0, color="#65707A", linestyle="--", linewidth=1)
        ax.errorbar(x, gains[f"{metric}_gain_pp_mean"], yerr=gains[f"{metric}_gain_pp_sd"], color=colors[1], marker="o", linewidth=1.8, capsize=4, zorder=3)
        ax.set_title(f"({'c' if col == 0 else 'd'}) Paired SA contribution", loc="left")
        ax.set_ylabel(f"SA minus no-SA {label} (pp)")
    for ax in axes.flat:
        ax.set_xlabel("Training recorded pot groups")
        ax.set_xticks(BUDGETS)
        ax.grid(axis="y", alpha=0.18)
        ax.set_xlim(6, 34)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12, wspace=0.26, hspace=0.45)
    fig.suptitle("Paired grouped learning curves with frozen model settings", fontsize=14, y=0.98)
    fig.text(0.08, 0.025, "Same five held-out folds. Five nested subset draws at 8/16/24 groups; one fit at 32.\nError bars: SD across five fold averages (not confidence intervals). Grey lines: individual folds.", fontsize=9, color="#4B5563")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(DELIVERY / f"paired_grouped_learning_curve.{extension}", dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    for ax, (metric, label) in zip(axes, (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"))):
        for model, color in zip(MODELS, colors):
            rows = summary[summary["model"] == model].sort_values("train_groups")
            short = "No-SA" if model == MODELS[0] else "SA"
            ax.plot(rows["train_groups"], 100 * rows[f"test_{metric}_mean"], "o-", color=color, label=f"{short}, held-out")
            ax.plot(rows["train_groups"], 100 * rows[f"apparent_train_{metric}_mean"], "s--", color=color, label=f"{short}, apparent train")
        ax.set_xlabel("Training recorded pot groups")
        ax.set_ylabel(label + " (%)")
        ax.set_xticks(BUDGETS)
        ax.set_ylim(65, 101)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.text(0.08, 0.02, "Apparent train scores use full-training base models and training-derived SA centres.\nThey are resubstitution diagnostics, not validation scores; the meta learner is fitted to grouped OOF features.", fontsize=9)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(DELIVERY / f"training_validation_diagnostic.{extension}", dpi=240)
    plt.close(fig)


def write_report(summary, gains, comparisons, elapsed):
    display = []
    for budget in BUDGETS:
        pair = summary[summary["train_groups"] == budget].set_index("model")
        gain = gains[gains["train_groups"] == budget].iloc[0]
        display.append({"training_recorded_groups": budget,
            "mean_training_spectra": pair.loc[MODELS[0], "n_train_spectra_mean"],
            "no_sa_accuracy_percent": 100 * pair.loc[MODELS[0], "test_accuracy_mean"],
            "sa_accuracy_percent": 100 * pair.loc[MODELS[1], "test_accuracy_mean"],
            "accuracy_sa_gain_pp": gain["accuracy_gain_pp_mean"],
            "no_sa_macro_f1_percent": 100 * pair.loc[MODELS[0], "test_macro_f1_mean"],
            "sa_macro_f1_percent": 100 * pair.loc[MODELS[1], "test_macro_f1_mean"],
            "macro_f1_sa_gain_pp": gain["macro_f1_gain_pp_mean"]})
    table = pd.DataFrame(display)
    csv_write(table, DELIVERY / "learning_curve_comparison_percent.csv")
    full = table.iloc[-1]
    start = table.iloc[0]
    sa_change = full["sa_macro_f1_percent"] - start["sa_macro_f1_percent"]
    exact = sum(c["different_predictions"] for c in comparisons) == 0
    lines = [
        "# 配对分组学习曲线实验（2026-10-06）", "",
        "## 设计与计算口径", "",
        "- 固定 833 条光谱、40 个记录 pot_id 组，与最终共同五折完全一致；记录组是否对应真实独立实体仍需作者核实。",
        "- 外层 StratifiedGroupKFold 五折，seed=20260929；每折固定 8 个测试组、32 个候选训练组。测试组不参与子集选择、标准化、SA 中心或模型拟合。",
        "- 训练规模为 8、16、24、32 个完整记录组，不按单条光谱随机抽样。8/16/24 组各做 5 次预设、嵌套、无放回子集抽取；32 组每折只有一项，不重复计数。",
        "- 子集 seed=20261006+1000×fold+repeat；模型 seed=20260929+1000×fold，不随训练规模或子集改变。子集顺序在训练前固定，不选择最佳结果。",
        "- 只比较 XGB-SVM-RF 与 XGBSVMRF-SA (NIR+SWIR)，共享同一组内层三折 OOF 基模型概率及外层预测。所有参数冻结，未再调参。",
        "- 不做欠采样、SMOTE、类别重加权或 0/1 阈值修正。SG 平滑窗口 9、二阶多项式，随后同窗口二阶导数。基模型用全部 2151 个波段。",
        "- XGBoost: 400 trees, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, hist, CUDA, n_jobs=4。SVM: C=100, RBF, gamma=scale, probability=True。RF: 400 trees, max_features=sqrt, min_samples_leaf=1, n_jobs=-1。",
        "- 元模型：训练 OOF 特征上拟合 StandardScaler 与 LogisticRegression(C=0.3, max_iter=3000, class_weight=None)。SA 的 12 维特征为六等级中心在 NIR 760–1300 nm 与 SWIR 1300–2500 nm 的光谱角。中心按训练组等权平均构建，内层验证特征使用对应内层训练中心。",
        "- XGBoost 在 RTX 4060 Laptop GPU 上训练，每次拟合检查实际 booster device 为 CUDA；SVM/RF/逻辑回归在 CPU 上运行。",
        "- 每折先对 5 次子集的指标求均值，再对 5 个外层折求均值与样本 SD (ddof=1)。SA 增量按配对折先做差。误差条是折间描述性 SD，不是置信区间。",
        "- 表中 Accuracy/Macro F1 为百分数，提升为百分点。未将 25 个共享测试集的结果当作独立重复来检验显著性。", "",
        "## 结果", "",
        "|训练记录组数|平均训练光谱数|无 SA Accuracy (%)|SA Accuracy (%)|Accuracy 增量 (pp)|无 SA Macro F1 (%)|SA Macro F1 (%)|Macro F1 增量 (pp)|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in display:
        lines.append("|{}|{:.1f}|{:.4f}|{:.4f}|{:+.4f}|{:.4f}|{:.4f}|{:+.4f}|".format(*row.values()))
    lines.extend(["", f"完整 32 组端点：两模型全部测试预测与既有最终共同五折{'逐条一致' if exact else '存在差异，详见 reproduction_check.json'}。所有 80 个配对训练任务均重新拟合；没有复用旧预测代替本次训练。",
        f"从 8 到 32 个训练组，最终 SA 模型 Macro F1 变化 {sa_change:+.4f} 个百分点。应同时查看无 SA 曲线与配对增量，不能将训练规模效应全归因于 SA。", "",
        "## 如何用于论文", "",
        "这是现有数据上的回顾性训练规模敏感性分析，不是真正的新样本前瞻验证，也不能预测超过 32 个训练组的准确率。",
        "即便学习曲线总体上升，也只能支持在当前抽样范围内模型表现与训练规模有关；不能证明新增独立来源样本必然提高表现，或 SA 贡献必然随样本量增大。",
        "同一记录组内增加光谱条数不等于增加独立组。未来外部验证需冻结模型流程并使用未参与选择/调参的新来源测试数据。",
        "最终方案此前经过现有数据的探索选择，本次冻结参数并不能消除历史模型选择造成的乐观偏倚。",
        "训练诊断图的虚线是完整训练后基模型与训练中心所产生的表观训练分数，不是独立验证分数；SA 中心包含训练观察，因此高训练分数需谨慎解释。",
        "旧学习曲线比例标签与训练样本数存在补集索引问题，且旧划分/重采样口径不同，未混入本次结果；旧文件保持不变。", "",
        "## 保存内容", "",
        "- paired_grouped_learning_curve: 同测试组学习曲线及配对 SA 增量，PNG/SVG/PDF。",
        "- training_validation_diagnostic: 表观训练分数与外层测试分数，PNG/SVG/PDF。",
        "- learning_curve_comparison_percent.csv: 直读百分数和百分点。",
        "- learning_curve_summary.csv: 小数制均值、折间 SD 与子集波动。",
        "- paired_sa_gain_summary.csv: 配对 SA 增量与折间 SD。",
        "详细检查点、所有逐条外层测试预测、逐类指标、嵌套抽样清单、内层组隔离审计与 SHA256 元数据保存在本次 work 目录。", "",
        f"累计训练耗时：{elapsed / 60:.2f} 分钟。未修改原始数据、既有结果或论文文档。", "",
    ])
    (DELIVERY / "学习曲线实验结果与口径说明.md").write_text("\n".join(lines), encoding="utf-8")
    json_write(DELIVERY / "headline_results.json", {"status": "completed", "comparison": display,
        "full_budget_prediction_reproduction_exact": exact, "training_seconds": elapsed,
        "uncertainty": "Descriptive SD across five overlapping CV folds, not independent-cohort confidence intervals.",
        "validation_type": "Retrospective grouped training-size sensitivity; not prospective validation."})
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "run.log", encoding="utf-8")])
    X, y, groups, meta, wave, files = load_cohort()
    tasks, folds, audit = make_plan(X, y, groups)
    reference = reference_predictions(meta, y, groups, folds)
    gpu = probe_cuda(X, y)
    logging.info("Verified cohort: %d spectra, %d recorded groups. %s", len(y), len(np.unique(groups)), gpu)
    definition = {
        "outer_seed": OUTER_SEED, "subset_seed": SUBSET_SEED, "train_group_budgets": BUDGETS,
        "subset_repeats_below_full_budget": SUBSET_REPEATS, "models": MODELS,
        "meta_C": 0.3, "resampling": None, "class_weight": None, "threshold": None,
        "source_sha256": {p.name: sha256(p) for p in files},
        "code_sha256": {"experiment": sha256(__file__), "base_helpers": sha256(base.__file__)},
        "reference_prediction_sha256": sha256(REFERENCE / "all_out_of_fold_predictions.csv"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "matplotlib": matplotlib.__version__},
    }
    fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    metadata_path = OUT / "run_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise RuntimeError("Code/data/protocol changed; old checkpoints cannot be silently reused.")
    json_write(metadata_path, {"fingerprint": fingerprint, "definition": definition, "gpu": gpu,
        "python_executable": sys.executable, "score_aggregation": "Average subsets within each fold, then equal-weight mean/SD across five folds.",
        "status": "validated" if args.validate_only else "running"})
    json_write(OUT / "frozen_subset_plan.json", tasks)
    json_write(OUT / "group_isolation_audit.json", audit)
    if args.validate_only:
        logging.info("PASS: 80 paired tasks; 240 inner group-isolation checks; all subsets nested; CUDA verified. No final training performed.")
        return
    metric_frames, prediction_frames, comparisons = [], [], []
    accumulated_training = 0.0
    for task_number, task in enumerate(tasks, start=1):
        checkpoint = OUT / (task["task_id"] + ".json")
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            assert saved["fingerprint"] == fingerprint
            metrics, predictions = pd.DataFrame(saved["metrics"]), pd.DataFrame(saved["predictions"])
            reproduction = saved["reproduction"]
            accumulated_training += saved["seconds"]
            logging.info("[%d/80] %s: loaded completed checkpoint", task_number, task["task_id"])
        else:
            tic = time.time()
            logging.info("[%d/80] %s: START (%d training spectra)", task_number, task["task_id"], len(task["train_row_indices"]))
            metrics, predictions, reproduction = evaluate_task(task, X, y, groups, meta, wave, reference)
            seconds = time.time() - tic
            accumulated_training += seconds
            json_write(checkpoint, {"fingerprint": fingerprint, "metrics": metrics.to_dict("records"),
                "predictions": predictions.to_dict("records"), "reproduction": reproduction, "seconds": seconds})
            rows = metrics.set_index("model")
            logging.info("[%d/80] DONE %.1fs, no-SA F1=%.4f, SA F1=%.4f, delta=%+.3f pp", task_number, seconds,
                rows.loc[MODELS[0], "test_macro_f1"], rows.loc[MODELS[1], "test_macro_f1"],
                100 * (rows.loc[MODELS[1], "test_macro_f1"] - rows.loc[MODELS[0], "test_macro_f1"]))
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        comparisons.extend(reproduction)
        metrics_all = pd.concat(metric_frames, ignore_index=True)
        predictions_all = pd.concat(prediction_frames, ignore_index=True)
        summary, paired, gains = summarise(metrics_all, predictions_all, completed=task_number == len(tasks))
        json_write(OUT / "progress.json", {"completed_tasks": task_number, "total_tasks": len(tasks),
            "last_task": task["task_id"], "accumulated_training_seconds": accumulated_training,
            "status": "completed" if task_number == len(tasks) else "running"})
        gc.collect()
    assert len(metrics_all) == 160
    assert all(summary["n_outer_folds"] == 5)
    assert len(comparisons) == 10
    json_write(OUT / "reproduction_check.json", comparisons)
    per_class_summary(predictions_all)
    make_figures(summary, paired, gains)
    table = write_report(summary, gains, comparisons, accumulated_training)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "completed"
    metadata["training_seconds"] = accumulated_training
    json_write(metadata_path, metadata)
    logging.info("ALL TASKS COMPLETE.\n%s\nDeliverables: %s", table.to_string(index=False), DELIVERY)


if __name__ == "__main__":
    main()
