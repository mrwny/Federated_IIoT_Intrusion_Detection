import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

MODEL_COLORS = {
    "cnn": "#4C72B0",
    "dnn": "#DD8452",
    "DNN": "#DD8452",
    "random_forest": "#55A868",
    "xgboost": "#C44E52",
}
MODEL_LABELS = {
    "cnn": "CNN",
    "dnn": "DNN",
    "DNN": "DNN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

EVAL_DIR = Path(__file__).resolve().parent.parent / "results"


def _apply_latex_style():
    """Set matplotlib params for LaTeX-matching dissertation figures."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _t(s: str) -> str:
    """Make a string safe for LaTeX rendering."""
    return s.replace("\u2014", "---").replace("&", r"\&")


def _find_newest_run(model_dir: Path) -> Optional[Path]:
    """Return the path of the newest timestamped run inside *model_dir*."""
    runs = sorted(
        [d for d in model_dir.iterdir()
         if d.is_dir() and re.match(r"\d{8}_\d{6}", d.name)],
        key=lambda d: d.name,
    )
    return runs[-1] if runs else None


def _parse_eval_file(eval_path: Path) -> Dict:
    """Parse a nids_evaluation.txt and return a dict of key metrics."""
    text = eval_path.read_text()
    metrics: Dict = {}

    # Overall accuracy
    m = re.search(r"Overall Accuracy:\s+([\d.]+)", text)
    if m:
        metrics["accuracy"] = float(m.group(1))

    # Per-class precision / recall / f1
    for label in ("Benign", "Attack"):
        pattern = rf"{label}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
        m = re.search(pattern, text)
        if m:
            metrics[f"{label.lower()}_precision"] = float(m.group(1))
            metrics[f"{label.lower()}_recall"] = float(m.group(2))
            metrics[f"{label.lower()}_f1"] = float(m.group(3))
            metrics[f"{label.lower()}_support"] = int(m.group(4))

    # Macro avg
    m = re.search(r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    if m:
        metrics["macro_precision"] = float(m.group(1))
        metrics["macro_recall"] = float(m.group(2))
        metrics["macro_f1"] = float(m.group(3))

    # Weighted avg
    m = re.search(r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    if m:
        metrics["weighted_precision"] = float(m.group(1))
        metrics["weighted_recall"] = float(m.group(2))
        metrics["weighted_f1"] = float(m.group(3))

    # Confusion matrix values
    m = re.search(r"True Negatives.*?:\s+(\d+)", text)
    if m:
        metrics["tn"] = int(m.group(1))
    m = re.search(r"False Positives.*?:\s+(\d+)", text)
    if m:
        metrics["fp"] = int(m.group(1))
    m = re.search(r"False Negatives.*?:\s+(\d+)", text)
    if m:
        metrics["fn"] = int(m.group(1))
    m = re.search(r"True Positives.*?:\s+(\d+)", text)
    if m:
        metrics["tp"] = int(m.group(1))

    # Attack Detection Rate & FPR
    m = re.search(r"Attack Detection Rate:\s+([\d.]+)", text)
    if m:
        metrics["attack_detection_rate"] = float(m.group(1))
    m = re.search(r"False Positive Rate:\s+([\d.]+)", text)
    if m:
        metrics["false_positive_rate"] = float(m.group(1))

    # Training date
    m = re.search(r"Training Date:\s+(.+)", text)
    if m:
        metrics["training_date"] = m.group(1).strip()

    # Model type hint
    m = re.search(r"Model (?:Architecture|Type):\s+(.+?)[\(\n]", text)
    if m:
        metrics["model_name"] = m.group(1).strip()

    return metrics


def collect_results(dataset: str) -> Dict[str, Dict]:
    """
    For a given dataset, find the newest eval per model type (single-run).
    Returns {model_type: metrics_dict}.
    """
    dataset_dir = EVAL_DIR / dataset
    if not dataset_dir.exists():
        print(f"  [skip] dataset directory not found: {dataset_dir}")
        return {}

    results = {}
    for model_dir in sorted(dataset_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        newest = _find_newest_run(model_dir)
        if newest is None:
            continue
        eval_file = newest / "nids_evaluation.txt"
        if not eval_file.exists():
            continue
        metrics = _parse_eval_file(eval_file)
        metrics["run_id"] = newest.name
        results[model_dir.name] = metrics

    return results


def collect_results_multi_seed(dataset: str) -> Dict[str, Dict]:
    """
    Collect results across all seeds for each model type.
    Returns {model_type: {metric_mean, metric_std, metric_values, ...}}.
    """
    dataset_dir = EVAL_DIR / "Baseline" / dataset
    if not dataset_dir.exists():
        dataset_dir = EVAL_DIR / dataset
        if not dataset_dir.exists():
            print(f"  [skip] dataset directory not found")
            return {}

    results = {}
    for model_dir in sorted(dataset_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith((".", "_")):
            continue

        model_name = model_dir.name
        seed_dirs = sorted(model_dir.glob("seed_*"))

        all_metrics: List[Dict] = []
        for seed_dir in seed_dirs:
            newest = _find_newest_run(seed_dir)
            if newest is None:
                continue
            eval_file = newest / "nids_evaluation.txt"
            if not eval_file.exists():
                continue
            all_metrics.append(_parse_eval_file(eval_file))

        if not all_metrics:
            # Fall back to single-run (no seed dirs)
            newest = _find_newest_run(model_dir)
            if newest and (newest / "nids_evaluation.txt").exists():
                m = _parse_eval_file(newest / "nids_evaluation.txt")
                m["run_id"] = newest.name
                m["num_seeds"] = 1
                results[model_name] = m
            continue

        # Aggregate across seeds
        agg: Dict = {"num_seeds": len(all_metrics)}
        numeric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
            if values:
                agg[key] = np.mean(values)           # mean value (used by plot funcs)
                agg[f"{key}_mean"] = np.mean(values)
                agg[f"{key}_std"] = np.std(values)
                agg[f"{key}_values"] = values

        results[model_name] = agg

    return results


def _get_std(results, model, key):
    return results[model].get(f"{key}_std", 0)


def plot_overall_metrics(results: Dict[str, Dict], dataset: str, ax: plt.Axes):
    models = list(results.keys())
    metric_keys = ["accuracy", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro F1", "Weighted F1"]

    x = np.arange(len(metric_labels))
    width = 0.8 / len(models)
    has_std = any(_get_std(results, m, metric_keys[0]) > 0 for m in models)

    for i, model in enumerate(models):
        vals = [results[model].get(k, 0) for k in metric_keys]
        errs = [_get_std(results, model, k) for k in metric_keys] if has_std else None
        bars = ax.bar(
            x + i * width, vals, width,
            yerr=errs,
            capsize=3 if has_std else 0,
            error_kw={"linewidth": 1.2, "capthick": 1.2},
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#999999"),
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7,
            )

    n_seeds = max(results[m].get("num_seeds", 1) for m in models)
    title = _t(f"{dataset} --- Overall Metrics")
    if n_seeds > 1:
        title += f" (mean $\\pm$ std, {n_seeds} seeds)"
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    all_vals = [results[m].get(k, 0) for m in models for k in metric_keys]
    ymin = max(0, min(all_vals) - 0.05)
    ax.set_ylim(ymin, 1.02)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_per_class_metrics(results: Dict[str, Dict], dataset: str, axes):
    """Two grouped bar charts: Benign & Attack precision/recall/F1 with error bars."""
    models = list(results.keys())
    sub_metrics = ["precision", "recall", "f1"]
    sub_labels = ["Precision", "Recall", "F1"]
    has_std = any(_get_std(results, m, "benign_precision") > 0 for m in models)

    for idx, cls in enumerate(["benign", "attack"]):
        ax = axes[idx]
        x = np.arange(len(sub_labels))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            keys = [f"{cls}_{s}" for s in sub_metrics]
            vals = [results[model].get(k, 0) for k in keys]
            errs = [_get_std(results, model, k) for k in keys] if has_std else None
            bars = ax.bar(
                x + i * width, vals, width,
                yerr=errs,
                capsize=3 if has_std else 0,
                error_kw={"linewidth": 1.2, "capthick": 1.2},
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, "#999999"),
                edgecolor="white", linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7,
                )

        ax.set_ylabel("Score")
        ax.set_title(_t(f"{dataset} --- {cls.title()} Class Metrics"))
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(sub_labels)
        all_vals = [results[m].get(f"{cls}_{s}", 0) for m in models for s in sub_metrics]
        ymin = max(0, min(all_vals) - 0.05)
        ax.set_ylim(ymin, 1.02)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)


def plot_detection_rates(results: Dict[str, Dict], dataset: str, axes):
    """Horizontal bar charts: Attack Detection Rate and False Positive Rate on separate subplots."""
    models = list(results.keys())
    y = np.arange(len(models))
    has_std = any(_get_std(results, m, "attack_detection_rate") > 0 for m in models)

    adr = [results[m].get("attack_detection_rate", 0) for m in models]
    fpr = [results[m].get("false_positive_rate", 0) for m in models]
    adr_err = [_get_std(results, m, "attack_detection_rate") for m in models] if has_std else None
    fpr_err = [_get_std(results, m, "false_positive_rate") for m in models] if has_std else None
    colors = [MODEL_COLORS.get(m, "#999999") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    ax_adr, ax_fpr = axes

    ax_adr.barh(y, adr, 0.6, label="Attack Detection Rate",
            xerr=adr_err, capsize=3 if has_std else 0,
            error_kw={"linewidth": 1.2, "capthick": 1.2},
            color=colors, edgecolor="white", linewidth=0.5)

    for i in range(len(models)):
        ax_adr.text(adr[i] + 0.005, y[i], f"{adr[i]:.4f}", va="center", fontsize=8)

    ax_adr.set_yticks(y)
    ax_adr.set_yticklabels(labels)
    ax_adr.set_xlabel("Rate")
    ax_adr.set_title(_t(f"{dataset} --- Attack Detection Rate"))
    ax_adr.grid(axis="x", alpha=0.3)
    xmin = max(0, min(adr) - 0.05) if adr else 0
    ax_adr.set_xlim(xmin, 1.05)

    ax_fpr.barh(y, fpr, 0.6, label="False Positive Rate",
            xerr=fpr_err, capsize=3 if has_std else 0,
            error_kw={"linewidth": 1.2, "capthick": 1.2},
            color=colors, edgecolor="white", linewidth=0.5, alpha=0.6, hatch="//")

    fpr_errs_safe = fpr_err if fpr_err else [0] * len(models)
    fpr_tips = [fpr[i] + fpr_errs_safe[i] for i in range(len(models))]
    tip_max = max(fpr_tips) if fpr_tips and max(fpr_tips) > 0 else 0.001
    for i in range(len(models)):
        ax_fpr.text(fpr_tips[i] + tip_max * 0.04, y[i], f"{fpr[i]:.6f}", va="center", fontsize=8)

    ax_fpr.set_yticks(y)
    ax_fpr.set_yticklabels(labels)
    ax_fpr.set_xlabel("Rate")
    ax_fpr.set_title(_t(f"{dataset} --- False Positive Rate"))
    ax_fpr.grid(axis="x", alpha=0.3)
    ax_fpr.set_xlim(0, tip_max * 1.5)


def plot_confusion_matrices(results: Dict[str, Dict], dataset: str, axes):
    """Plot normalised confusion matrices side by side for each model."""
    models = list(results.keys())
    for i, model in enumerate(models):
        ax = axes[i]
        tn = results[model].get("tn", 0)
        fp = results[model].get("fp", 0)
        fn = results[model].get("fn", 0)
        tp = results[model].get("tp", 0)

        cm = np.array([[tn, fp], [fn, tp]])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Attack"], fontsize=8)
        ax.set_yticklabels(["Benign", "Attack"], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=10)

        for row in range(2):
            for col in range(2):
                txt_color = "white" if cm_pct[row, col] > 60 else "black"
                ax.text(
                    col, row,
                    f"{int(round(cm[row, col])):,}\n({cm_pct[row, col]:.1f}%)",
                    ha="center", va="center", fontsize=8, color=txt_color,
                )

    for j in range(len(models), len(axes)):
        axes[j].axis("off")


def _save_or_show(fig, out_dir: Path, name: str, save: bool):
    if save:
        out_path = out_dir / f"{name}.pdf"
        fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"    Saved -> {out_path}")
    else:
        plt.show()
    plt.close(fig)


def generate_report(dataset: str, save: bool = False):
    """Generate separate comparison plots for one dataset."""
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset}")
    print(f"{'=' * 60}")

    _apply_latex_style()

    results = collect_results_multi_seed(dataset)
    if not results:
        results = collect_results(dataset)
    if not results:
        print("  No evaluation results found. Skipping.")
        return

    for model, metrics in results.items():
        n_seeds = metrics.get('num_seeds', 1)
        acc = metrics.get('accuracy', 0)
        acc_std = metrics.get('accuracy_std', 0)
        mf1 = metrics.get('macro_f1', 0)
        if n_seeds > 1:
            print(
                f"  {MODEL_LABELS.get(model, model):15s}  {n_seeds} seeds  "
                f"acc={acc:.10f}+/-{acc_std:.10f}  macro_f1={mf1:.10f}"
            )
        else:
            print(
                f"  {MODEL_LABELS.get(model, model):15s}  run={metrics.get('run_id')}  "
                f"acc={acc:.10f}  macro_f1={mf1:.10f}"
            )

    n_models = len(results)
    out_dir = EVAL_DIR / "_figures" / "baseline" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall metrics (Accuracy, Macro F1, Weighted F1)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_overall_metrics(results, dataset, ax)
    fig.tight_layout()
    _save_or_show(fig, out_dir, "overall_metrics", save)

    # 2. Detection rates
    fig, axes = plt.subplots(2, 1, figsize=(8, max(n_models * 1.6, 5)))
    plot_detection_rates(results, dataset, axes)
    fig.tight_layout()
    _save_or_show(fig, out_dir, "detection_rates", save)

    # 3. Per-class metrics (Benign & Attack side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_per_class_metrics(results, dataset, axes)
    fig.tight_layout()
    _save_or_show(fig, out_dir, "per_class_metrics", save)

    # 4. Confusion matrices (one per model)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    plot_confusion_matrices(results, dataset, axes)
    fig.suptitle(_t(f"{dataset} --- Confusion Matrices"), fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_or_show(fig, out_dir, "confusion_matrices", save)

    # 5. Per-model individual charts
    generate_per_model_report(results, dataset, save)


def generate_per_model_report(results: Dict[str, Dict], dataset: str, save: bool = True):
    """Generate individual charts per model, saved into each model's folder."""
    if not save:
        return
    print(f"\n  Generating per-model charts...")
    for model, metrics in results.items():
        model_dir = EVAL_DIR / "_figures" / "baseline" / dataset / model
        model_dir.mkdir(parents=True, exist_ok=True)
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "#999999")
        n_seeds = metrics.get("num_seeds", 1)
        has_std = n_seeds > 1
        seed_note = f" (mean $\\pm$ std, {n_seeds} seeds)" if has_std else ""

        tn = metrics.get("tn", 0)
        fp = metrics.get("fp", 0)
        fn = metrics.get("fn", 0)
        tp = metrics.get("tp", 0)
        if any(v > 0 for v in [tn, fp, fn, tp]):
            cm = np.array([[tn, fp], [fn, tp]])
            cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Benign", "Attack"], fontsize=10)
            ax.set_yticklabels(["Benign", "Attack"], fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(_t(f"{label} --- {dataset}"))
            for row in range(2):
                for col in range(2):
                    txt_color = "white" if cm_pct[row, col] > 60 else "black"
                    ax.text(
                        col, row,
                        f"{int(round(cm[row, col])):,}\n({cm_pct[row, col]:.1f}\%)",
                        ha="center", va="center", fontsize=10, color=txt_color,
                    )
            fig.tight_layout()
            _save_or_show(fig, model_dir, "confusion_matrix", True)

        adr = metrics.get("attack_detection_rate", 0)
        fpr = metrics.get("false_positive_rate", 0)
        adr_std = metrics.get("attack_detection_rate_std", 0)
        fpr_std = metrics.get("false_positive_rate_std", 0)

        fig, axes = plt.subplots(2, 1, figsize=(6, 4))
        ax_adr, ax_fpr = axes
        adr_err = [adr_std] if has_std else None
        fpr_err = [fpr_std] if has_std else None

        ax_adr.barh([0], [adr], 0.5, xerr=adr_err,
                capsize=4 if has_std else 0,
                error_kw={"linewidth": 1.2, "capthick": 1.2},
                color=color, edgecolor="white", linewidth=0.5, alpha=1.0)
        
        bars_fpr = ax_fpr.barh([0], [fpr], 0.5, xerr=fpr_err,
                           capsize=4 if has_std else 0,
                           error_kw={"linewidth": 1.2, "capthick": 1.2},
                           color=color, edgecolor="white", linewidth=0.5, alpha=0.6)
        bars_fpr[0].set_hatch("//")
        
        ax_adr.text(adr + 0.005, 0, f"{adr:.4f}", va="center", fontsize=9)
        fpr_tip = fpr + (fpr_std if has_std else 0)
        ax_fpr.text(fpr_tip + (fpr_tip * 0.05 + 1e-4), 0, f"{fpr:.6f}", va="center", fontsize=9)

        ax_adr.set_yticks([0])
        ax_adr.set_yticklabels(["Attack Detection"])
        ax_adr.set_xlabel("Rate")
        ax_adr.set_title(_t("Attack Detection Rate"))
        ax_adr.grid(axis="x", alpha=0.3)
        ax_adr.set_xlim(max(0, adr - 0.05), 1.05)

        ax_fpr.set_yticks([0])
        ax_fpr.set_yticklabels(["False Positive"])
        ax_fpr.set_xlabel("Rate")
        ax_fpr.set_title(_t("False Positive Rate"))
        ax_fpr.grid(axis="x", alpha=0.3)
        ax_fpr.set_xlim(0, fpr_tip * 1.5 + 1e-3)

        fig.suptitle(_t(f"{label} --- Detection Rates{seed_note}"), y=1.05)
        fig.tight_layout()
        _save_or_show(fig, model_dir, "detection_rates", True)

        metric_keys = ["accuracy", "macro_f1", "weighted_f1"]
        metric_labels = ["Accuracy", "Macro F1", "Weighted F1"]
        vals = [metrics.get(k, 0) for k in metric_keys]
        errs = [metrics.get(f"{k}_std", 0) for k in metric_keys] if has_std else None

        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(metric_labels))
        ax.bar(x, vals, 0.5, yerr=errs,
               capsize=4 if has_std else 0,
               error_kw={"linewidth": 1.2, "capthick": 1.2},
               color=color, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ymin = max(0, min(vals) - 0.05)
        ax.set_ylim(ymin, 1.05)
        ax.set_title(_t(f"{label} --- Overall Metrics{seed_note}"))
        fig.tight_layout()
        _save_or_show(fig, model_dir, "metrics", True)

        sub_metrics = ["precision", "recall", "f1"]
        sub_labels = ["Precision", "Recall", "F1"]
        classes = ["benign", "attack"]
        class_colors = ["#3498db", "#e74c3c"]

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sub_labels))
        width = 0.35
        for ci, cls in enumerate(classes):
            keys = [f"{cls}_{s}" for s in sub_metrics]
            cvals = [metrics.get(k, 0) for k in keys]
            cerrs = [metrics.get(f"{k}_std", 0) for k in keys] if has_std else None
            ax.bar(x + ci * width, cvals, width, yerr=cerrs,
                   capsize=4 if has_std else 0,
                   error_kw={"linewidth": 1.2, "capthick": 1.2},
                   label=cls.title(), color=class_colors[ci],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(sub_labels)
        ax.set_ylabel("Score")
        all_vals = [metrics.get(f"{c}_{s}", 0) for c in classes for s in sub_metrics]
        ymin = max(0, min(all_vals) - 0.05) if all_vals else 0
        ax.set_ylim(ymin, 1.05)
        ax.legend()
        ax.set_title(_t(f"{label} --- Per-Class Metrics{seed_note}"))
        fig.tight_layout()
        _save_or_show(fig, model_dir, "per_class_metrics", True)

        print(f"    {label}: 4 charts -> {model_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["DataSense", "Edge-IIoT", "all"],
    )
    parser.add_argument(
        "--save", action="store_true",
    )
    args = parser.parse_args()

    datasets = ["DataSense", "Edge-IIoT"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        generate_report(ds, save=args.save)


if __name__ == "__main__":
    main()
