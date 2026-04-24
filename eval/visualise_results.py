import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EVAL_DIR = Path(__file__).resolve().parent.parent / "results"
BASELINE_DIR = EVAL_DIR / "Baseline"
FEDERATED_DIR = EVAL_DIR / "Federated"

DATASETS = ["DataSense", "Edge-IIoT"]

MODEL_COLORS = {
    "DNN":          "#2980b9",
    "dnn":          "#2980b9",
    "CNN":          "#e67e22",
    "cnn":          "#e67e22",
    "XGBoost":      "#c0392b",
    "xgboost":      "#c0392b",
    "RandomForest": "#27ae60",
    "random_forest":"#27ae60",
}
MODEL_LABELS = {
    "dnn": "DNN", "cnn": "CNN", "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "DNN": "DNN", "CNN": "CNN", "XGBoost": "XGBoost",
    "RandomForest": "Random Forest",
}
STRATEGY_COLORS = {
    "FedAvg":       "#3498db",
    "FedProx":      "#e74c3c",
    "FedRF":        "#27ae60",
    "FedXgbBagging":"#8e44ad",
}
STRATEGY_MARKERS = {
    "FedAvg": "o", "FedProx": "s", "FedRF": "^", "FedXgbBagging": "D",
}


def apply_latex_style():
    """Set matplotlib params for clean academic figures with LaTeX fonts."""
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': r'\usepackage{amsmath}',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _save_or_show(fig, out_dir: Path, name: str, save: bool):
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.pdf"
        fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"    Saved → {path}")
    else:
        plt.show()
    plt.close(fig)

def _find_newest_run(seed_dir: Path) -> Optional[Path]:
    runs = sorted(
        [d for d in seed_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.name
    )
    return runs[-1] if runs else None


def _aggregate_metrics(metric_list: List[dict], agg: dict):
    if not metric_list:
        return
    numeric_keys = [k for k in metric_list[0] if isinstance(metric_list[0][k], (int, float))]
    for key in numeric_keys:
        vals = [m[key] for m in metric_list if key in m and isinstance(m[key], (int, float))]
        if vals:
            agg[key] = np.mean(vals)
            agg[f"{key}_std"] = np.std(vals)


def _parse_baseline_eval(eval_path: Path) -> Optional[dict]:
    text = eval_path.read_text()
    metrics = {}
    patterns = {
        'accuracy':             r'Overall Accuracy:\s+([\d.]+)',
        'detection_rate':       r'Attack Detection Rate:\s+([\d.]+)',
        'false_positive_rate':  r'False Positive Rate:\s+([\d.]+)',
        'benign_precision':     r'Benign\s+([\d.]+)\s+[\d.]+\s+[\d.]+',
        'benign_recall':        r'Benign\s+[\d.]+\s+([\d.]+)\s+[\d.]+',
        'benign_f1':            r'Benign\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'attack_precision':     r'Attack\s+([\d.]+)\s+[\d.]+\s+[\d.]+',
        'attack_recall':        r'Attack\s+[\d.]+\s+([\d.]+)\s+[\d.]+',
        'attack_f1':            r'Attack\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'macro_f1':             r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics if metrics else None


def collect_baseline(dataset: str) -> Dict[str, dict]:
    """Collect baseline results: {model: {metric_mean, metric_std, ...}}."""
    base = BASELINE_DIR / dataset
    if not base.exists():
        return {}
    results = {}
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith((".", "_")):
            continue
        model_name = model_dir.name
        seed_dirs = sorted(model_dir.glob("seed_*"))
        all_metrics = []
        for sd in seed_dirs:
            newest = _find_newest_run(sd)
            if newest is None:
                continue
            ef = newest / "nids_evaluation.txt"
            if ef.exists():
                m = _parse_baseline_eval(ef)
                if m:
                    all_metrics.append(m)
        if not all_metrics:
            continue
        agg = {"num_seeds": len(all_metrics)}
        _aggregate_metrics(all_metrics, agg)
        results[model_name] = agg
    return results


def _parse_fed_results(path: Path) -> dict:
    data = {}
    for line in path.read_text().strip().splitlines():
        if ':' not in line:
            continue
        key, val = line.split(':', 1)
        key = key.strip().lower().replace(' ', '_')
        val = val.strip()
        try:
            data[key] = float(val)
        except ValueError:
            data[key] = val
    return data


def collect_federated(dataset: str) -> List[dict]:
    """Collect federated results as a list of aggregated config dicts."""
    base = FEDERATED_DIR / dataset
    if not base.exists():
        return []

    configs = defaultdict(list)
    for results_file in base.rglob("results.txt"):
        parts = results_file.relative_to(base).parts
        metrics = _parse_fed_results(results_file)
        if not metrics:
            continue
        
        comm_cost_file = results_file.parent / "comm_cost.json"
        if comm_cost_file.exists():
            try:
                with open(comm_cost_file) as f:
                    cc_data = json.load(f)
                    metrics["total_mb"] = cc_data.get("total_MB", 0.0)
            except Exception:
                pass

        config_parts = list(parts[:-1])  # remove results.txt
        seed_dir = config_parts[-1] if config_parts and config_parts[-1].startswith("seed_") else None
        if seed_dir:
            config_parts = config_parts[:-1]

        # Skip mu_sweep results (handled separately)
        if config_parts and config_parts[0] == "mu_sweep":
            continue
        # Skip mu subdirs in FedProx (keep only the "best mu" configs)
        # But include them — the mu value becomes part of the config key
        if len(config_parts) < 4:
            continue

        strategy = config_parts[0]
        model = config_parts[1]
        partition = config_parts[2]
        try:
            num_clients = int(config_parts[3])
        except ValueError:
            continue

        # Extract frac if present
        frac = None
        mu = None
        for p in config_parts[4:]:
            if p.startswith("frac_"):
                frac = p.replace("frac_", "")
            elif p.startswith("mu_"):
                mu = p.replace("mu_", "")

        # Skip results not under an explicit frac_ directory (old residuals)
        if frac is None:
            continue

        config_key = (strategy, model, partition, num_clients, frac, mu)
        configs[config_key].append(metrics)

    results = []
    for (strategy, model, partition, num_clients, frac, mu), metric_list in sorted(configs.items()):
        agg = {
            "strategy": strategy, "model": model, "partition": partition,
            "num_clients": num_clients, "frac": frac, "mu": mu,
            "num_seeds": len(metric_list),
        }
        _aggregate_metrics(metric_list, agg)
        results.append(agg)
    return results


def collect_mu_sweep(dataset: str) -> List[dict]:
    """Collect mu_sweep results: list of {mu, accuracy, ...}."""
    base = FEDERATED_DIR / dataset / "mu_sweep"
    if not base.exists():
        return []

    configs = defaultdict(list)
    for rf in base.rglob("results.txt"):
        parts = rf.relative_to(base).parts
        metrics = _parse_fed_results(rf)
        if not metrics:
            continue
        config_parts = list(parts[:-1])
        seed_dir = config_parts[-1] if config_parts and config_parts[-1].startswith("seed_") else None
        if seed_dir:
            config_parts = config_parts[:-1]

        # Structure: {clients}/mu_{val}
        if len(config_parts) < 2:
            continue
        try:
            num_clients = int(config_parts[0])
        except ValueError:
            continue
        mu_str = config_parts[1].replace("mu_", "")
        try:
            mu_val = float(mu_str)
        except ValueError:
            continue

        configs[(num_clients, mu_val)].append(metrics)

    results = []
    for (num_clients, mu_val), metric_list in sorted(configs.items()):
        agg = {"num_clients": num_clients, "mu": mu_val, "num_seeds": len(metric_list)}
        _aggregate_metrics(metric_list, agg)
        results.append(agg)
    return results

def chart_baseline_comparison(datasets: List[str], save: bool, out_dir: Path):
    """Grouped bar chart of Accuracy, DR, FPR per model, side-by-side datasets."""
    print("\n  Baseline Model Comparison")
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)

    for col, ds in enumerate(datasets):
        ax = axes[0, col]
        bl = collect_baseline(ds)
        if not bl:
            ax.set_title(f"{ds} — No baseline data")
            continue

        models = list(bl.keys())
        metrics = ["accuracy", "detection_rate", "false_positive_rate"]
        labels = ["Accuracy", "Detection Rate", "FPR"]
        x = np.arange(len(metrics))
        width = 0.8 / max(len(models), 1)

        for i, model in enumerate(models):
            vals = [bl[model].get(m, 0) for m in metrics]
            errs = [bl[model].get(f"{m}_std", 0) for m in metrics]
            color = MODEL_COLORS.get(model, "#999")
            ax.bar(x + i * width, vals, width, yerr=errs, capsize=3,
                   label=MODEL_LABELS.get(model, model), color=color,
                   edgecolor="white", linewidth=0.5,
                   error_kw={"linewidth": 1, "capthick": 1})

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        n_seeds = max(bl[m].get("num_seeds", 1) for m in models)
        title = f"{ds} — Baseline Models"
        if n_seeds > 1:
            title += f" (mean ± std, {n_seeds} seeds)"
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.08)

    fig.tight_layout()
    _save_or_show(fig, out_dir, "baseline_comparison", save)

def chart_scalability(datasets: List[str], save: bool, out_dir: Path):
    """Line plot: accuracy vs # clients for each model/strategy, IID partition."""
    print("\n  Federated Scalability")
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)

    for col, ds in enumerate(datasets):
        ax = axes[0, col]
        fed = collect_federated(ds)
        # Filter to IID, frac 1.0, no mu override
        iid = [r for r in fed if r["partition"] == "iid" and r["frac"] == "1.0"
               and r.get("mu") is None]
        if not iid:
            ax.set_title(f"{ds} — No IID data")
            continue

        # Group by (strategy, model)
        groups = defaultdict(list)
        for r in iid:
            groups[(r["strategy"], r["model"])].append(r)

        for (strat, model), recs in sorted(groups.items()):
            recs = sorted(recs, key=lambda r: r["num_clients"])
            clients = [r["num_clients"] for r in recs]
            accs = [r.get("accuracy", 0) for r in recs]
            stds = [r.get("accuracy_std", 0) for r in recs]
            color = STRATEGY_COLORS.get(strat, MODEL_COLORS.get(model, "#999"))
            marker = STRATEGY_MARKERS.get(strat, "o")
            label = f"{strat}/{MODEL_LABELS.get(model, model)}"
            line, = ax.plot(clients, accs, marker=marker, linewidth=1.8,
                            markersize=5, label=label, color=color)
            if any(s > 0 for s in stds):
                ax.fill_between(clients,
                                [a - s for a, s in zip(accs, stds)],
                                [a + s for a, s in zip(accs, stds)],
                                alpha=0.15, color=line.get_color())

        ax.set_xlabel("Number of Clients")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds} — Scalability (IID, frac=1.0)")
        ax.legend(fontsize=7, ncol=2, loc="lower left")

    fig.tight_layout()
    _save_or_show(fig, out_dir, "scalability", save)


def chart_iid_vs_noniid(datasets: List[str], save: bool, out_dir: Path):
    """Grouped bars showing IID vs non-IID accuracy per model (at mid client count)."""
    print("\n IID vs Non-IID Degradation")
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)

    for col, ds in enumerate(datasets):
        ax = axes[0, col]
        fed = collect_federated(ds)
        # Use frac 1.0, no mu, pick a middle client count
        filt = [r for r in fed if r["frac"] == "1.0" and r.get("mu") is None]
        if not filt:
            ax.set_title(f"{ds} — No data")
            continue

        all_clients = sorted(set(r["num_clients"] for r in filt))
        mid_c = all_clients[len(all_clients) // 2] if all_clients else 8
        filt = [r for r in filt if r["num_clients"] == mid_c]

        models = sorted(set(r["model"] for r in filt))
        partitions = ["iid", "non-iid"]
        x = np.arange(len(models))
        width = 0.35
        colors = ["#3498db", "#e74c3c"]

        for i, part in enumerate(partitions):
            vals, errs = [], []
            for model in models:
                cands = [r for r in filt if r["model"] == model and r["partition"] == part]
                if cands:
                    best = max(cands, key=lambda r: r.get("accuracy", 0))
                    vals.append(best.get("accuracy", 0))
                    errs.append(best.get("accuracy_std", 0))
                else:
                    vals.append(0)
                    errs.append(0)

            ax.bar(x + i * width, vals, width, yerr=errs, capsize=3,
                   label=('IID' if part == 'iid' else 'Non-IID'), color=colors[i],
                   edgecolor="white", linewidth=0.5,
                   error_kw={"linewidth": 1, "capthick": 1})

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds} — IID vs Non-IID ({mid_c} clients)")
        all_vals = [r.get("accuracy", 0) for r in filt]
        ymin = max(0, min(all_vals) - 0.08) if all_vals else 0
        ax.set_ylim(ymin, 1.05)
        ax.legend()

    fig.tight_layout()
    _save_or_show(fig, out_dir, "iid_vs_noniid", save)


def chart_fedavg_vs_fedprox(datasets: List[str], save: bool, out_dir: Path):
    """Paired bars comparing FedAvg and FedProx for DNN and CNN, IID, mid clients."""
    print("\n  FedAvg vs FedProx")
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)

    for col, ds in enumerate(datasets):
        ax = axes[0, col]
        fed = collect_federated(ds)
        nn_strats = [r for r in fed if r["strategy"] in ("FedAvg", "FedProx")
                     and r["model"] in ("DNN", "CNN")
                     and r["partition"] == "iid" and r["frac"] == "1.0"]
        if not nn_strats:
            ax.set_title(f"{ds} — No FedAvg/FedProx data")
            continue

        all_clients = sorted(set(r["num_clients"] for r in nn_strats))
        mid_c = all_clients[len(all_clients) // 2] if all_clients else 8
        nn_strats = [r for r in nn_strats if r["num_clients"] == mid_c]

        models = sorted(set(r["model"] for r in nn_strats))
        strategies = ["FedAvg", "FedProx"]
        x = np.arange(len(models))
        width = 0.35

        for i, strat in enumerate(strategies):
            vals, errs = [], []
            for model in models:
                cands = [r for r in nn_strats if r["model"] == model and r["strategy"] == strat]
                if cands:
                    best = max(cands, key=lambda r: r.get("accuracy", 0))
                    vals.append(best.get("accuracy", 0))
                    errs.append(best.get("accuracy_std", 0))
                else:
                    vals.append(0)
                    errs.append(0)

            ax.bar(x + i * width, vals, width, yerr=errs, capsize=3,
                   label=strat, color=STRATEGY_COLORS[strat],
                   edgecolor="white", linewidth=0.5,
                   error_kw={"linewidth": 1, "capthick": 1})

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds} — FedAvg vs FedProx (IID, {mid_c} clients)")
        all_vals = [r.get("accuracy", 0) for r in nn_strats]
        ymin = max(0, min(all_vals) - 0.02) if all_vals else 0
        ax.set_ylim(ymin, 1.02)
        ax.legend()

    fig.tight_layout()
    _save_or_show(fig, out_dir, "fedavg_vs_fedprox", save)

def chart_baseline_vs_federated(datasets: List[str], save: bool, out_dir: Path):
    """Grouped bars: baseline vs federated accuracy (20 clients, Non-IID, frac=0.4)."""
    print("\n  Baseline vs Federated (20C, Non-IID, Frac 0.4)")
    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds, 1, figsize=(10, 6 * n_ds), squeeze=False)

    model_map_fed_to_bl = {
        "DNN": "dnn", "CNN": "cnn", "XGBoost": "xgboost",
        "RandomForest": "random_forest",
    }

    strategies_to_plot = [
        ("FedAvg", "DNN", "FedAvg", "#e74c3c"),
        ("FedProx", "DNN", "FedProx", "#e67e22"),
        ("FedXgbBagging", "XGBoost", "FedXGB", "#f1c40f"),
        ("AttentionFedXgb", "XGBoost", "Attention FedXGB", "#f39c12"),
        ("FedRF", "RandomForest", "FedRF", "#2ecc71"),
    ]

    for row, ds in enumerate(datasets):
        ax = axes[row, 0]
        bl = collect_baseline(ds)
        fed = collect_federated(ds)

        labels = []
        bl_vals, bl_errs = [], []
        fed_vals, fed_errs = [], []
        fed_colors = []

        for strat, fm, lbl, col_code in strategies_to_plot:
            bl_key = model_map_fed_to_bl.get(fm)
            if bl_key not in bl:
                continue
            
            # Constrain to 20 clients, non-iid, frac=0.4
            cands = [r for r in fed if r["partition"] == "non-iid" 
                     and r["strategy"] == strat and r["model"] == fm 
                     and r["num_clients"] == 20 and r["frac"] == "0.4"]
            
            if not cands:
                continue
                
            labels.append(lbl)
            fed_colors.append(col_code)
            
            bl_vals.append(bl[bl_key].get("accuracy", 0))
            bl_errs.append(bl[bl_key].get("accuracy_std", 0))

            best = max(cands, key=lambda r: r.get("accuracy", 0))
            fed_vals.append(best.get("accuracy", 0))
            fed_errs.append(best.get("accuracy_std", 0))

        if not labels:
            ax.set_title(f"{ds} — No matching data")
            continue

        x = np.arange(len(labels))
        width = 0.35
        # Move X axis slightly to give space for labels
        ax.bar(x - width/2, bl_vals, width, yerr=bl_errs, capsize=3,
               label="Centralised Baseline", color="#2c3e50",
               edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 1, "capthick": 1})
        bars_fed = ax.bar(x + width/2, fed_vals, width, yerr=fed_errs, capsize=3,
               label="Federated (20c, Non-IID, Frac=0.4)", color=fed_colors,
               edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 1, "capthick": 1})

        # Add single patch for the legend for the multicoloured federated bars
        legend_handles = [
            mpatches.Patch(color='#2c3e50', label='Centralised Baseline'),
            mpatches.Patch(color='#bdc3c7', label='Federated')
        ]

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds} — Baseline vs Federated - 20 clients, Non-IID, frac=0.4")
        all_v = bl_vals + fed_vals
        
        # Add labels to the top of baseline bars
        for idx, (v, e) in enumerate(zip(bl_vals, bl_errs)):
            if v > 0:
                ax.text(x[idx] - width/2, v + e + 0.005, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=7, color="#7f8c8d")

        # Add labels to the top of federated bars
        for idx, (v, e) in enumerate(zip(fed_vals, fed_errs)):
            if v > 0:
                ax.text(x[idx] + width/2, v + e + 0.005, f"{v*100:.1f}%", ha='center', va='bottom', fontsize=8, color="#2c3e50", fontweight='bold')

        ymin = max(0, min(all_v) - 0.05) if all_v else 0
        ax.set_ylim(ymin, 1.05)
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    _save_or_show(fig, out_dir, "baseline_vs_federated", save)

def chart_mu_sweep(datasets: List[str], save: bool, out_dir: Path):
    """Line plot: accuracy vs µ value from mu_sweep results."""
    print("\n  µ-Sweep (FedProx proximal term)")
    has_data = False
    for ds in datasets:
        data = collect_mu_sweep(ds)
        if data:
            has_data = True
            break
    if not has_data:
        print("    No µ-sweep data found, skipping.")
        return

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), squeeze=False)

    for col, ds in enumerate(datasets):
        ax = axes[0, col]
        data = collect_mu_sweep(ds)
        if not data:
            ax.set_title(f"{ds} — No µ-sweep data")
            ax.set_visible(False)
            continue

        # Group by num_clients
        groups = defaultdict(list)
        for r in data:
            groups[r["num_clients"]].append(r)

        for nc, recs in sorted(groups.items()):
            recs = sorted(recs, key=lambda r: r["mu"])
            mus = [r["mu"] for r in recs]
            accs = [r.get("accuracy", 0) for r in recs]
            stds = [r.get("accuracy_std", 0) for r in recs]

            line, = ax.plot(mus, accs, marker="o", linewidth=1.8, markersize=5,
                            label=f"{nc} clients")
            if any(s > 0 for s in stds):
                ax.fill_between(mus,
                                [a - s for a, s in zip(accs, stds)],
                                [a + s for a, s in zip(accs, stds)],
                                alpha=0.15, color=line.get_color())

        ax.set_xscale("log")
        ax.set_xlabel("µ (Proximal Term)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{ds} — FedProx µ-Sweep")
        ax.legend()

    fig.tight_layout()
    _save_or_show(fig, out_dir, "mu_sweep", save)


def chart_communication_cost(datasets: List[str], save: bool, out_dir: Path):
    """Bar chart of total communication cost (MB) per config."""
    print("\n  Communication Cost")

    for ds in datasets:
        base = FEDERATED_DIR / ds
        if not base.exists():
            continue

        json_files = sorted(base.rglob("comm_cost.json"))
        if not json_files:
            print(f"    No comm_cost.json for {ds}, skipping.")
            continue

        records = defaultdict(list)
        for jp in json_files:
            with open(jp) as f:
                data = json.load(f)
            rel = jp.relative_to(base)
            parts = [p for p in rel.parts[:-1]
                     if not p.startswith("seed_") and not p.startswith("mu_")]
            if len(parts) >= 3:
                label = "/".join(parts[:3])  # strategy/model/partition
                records[label].append(data.get("total_MB", 0))

        if not records:
            continue

        labels = sorted(records.keys())
        means = [np.mean(records[l]) for l in labels]
        stds = [np.std(records[l]) for l in labels]

        fig, ax = plt.subplots(figsize=(max(len(labels) * 1.2, 8), 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color="#3498db",
                      edgecolor="white", linewidth=0.5,
                      error_kw={"linewidth": 1, "capthick": 1})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Total Communication Cost (MB)")
        ax.set_title(f"{ds} — Communication Cost")
        fig.tight_layout()
        _save_or_show(fig, out_dir, f"comm_cost_{ds}", save)


_STRATEGY_SNAPSHOT_CLIENTS = 20


def _per_strategy_scalability(strat_data: List[dict], strategy: str,
                              dataset: str, save: bool, out_dir: Path):
    """Line plot: accuracy vs # clients for each model under this strategy.

    frac=1.0 shown as solid lines; frac=0.4 shown as dashed lines (non-IID only).
    """
    # Collect frac 1.0 data
    full = [r for r in strat_data if r["frac"] == "1.0" and r.get("mu") is None]
    if not full:
        # FedProx always has mu
        full = [r for r in strat_data if r["frac"] == "1.0"]
    # Collect frac 0.4 data (non-IID only)
    partial = [r for r in strat_data if r["frac"] == "0.4"]

    if not full:
        return

    partitions = sorted(set(r["partition"] for r in full))
    models = sorted(set(r["model"] for r in full))
    n_parts = len(partitions)

    fig, axes = plt.subplots(1, n_parts, figsize=(7 * n_parts, 5), squeeze=False)

    # Compute global y-axis limits to synchronize panels
    all_global_accs = [r.get("accuracy", 0) for r in full + partial]
    y_min_global = max(0, min(all_global_accs) - 0.02) if all_global_accs else 0
    y_max_global = 1.05

    for col, part in enumerate(partitions):
        ax = axes[0, col]

        # --- frac=1.0 (solid) ---
        part_data = sorted(
            [r for r in full if r["partition"] == part],
            key=lambda r: r["num_clients"],
        )
        for model in models:
            model_data = [r for r in part_data if r["model"] == model]
            if not model_data:
                continue
            clients = [r["num_clients"] for r in model_data]
            accs = [r.get("accuracy", 0) for r in model_data]
            stds = [r.get("accuracy_std", 0) for r in model_data]
            color = MODEL_COLORS.get(model, "#999")
            lbl = MODEL_LABELS.get(model, model)
            line, = ax.plot(clients, accs, marker="o", linewidth=1.8,
                            markersize=5, label=f"{lbl} (frac=1.0)",
                            color=color)
            if any(s > 0 for s in stds):
                ax.fill_between(clients,
                                [a - s for a, s in zip(accs, stds)],
                                [a + s for a, s in zip(accs, stds)],
                                alpha=0.15, color=line.get_color())

        # --- frac=0.4 (dashed, non-IID only) ---
        if part == "non-iid" and partial:
            part_partial = sorted(
                [r for r in partial if r["partition"] == part],
                key=lambda r: r["num_clients"],
            )
            for model in models:
                model_data = [r for r in part_partial if r["model"] == model]
                if not model_data:
                    continue
                clients = [r["num_clients"] for r in model_data]
                accs = [r.get("accuracy", 0) for r in model_data]
                stds = [r.get("accuracy_std", 0) for r in model_data]
                color = MODEL_COLORS.get(model, "#999")
                lbl = MODEL_LABELS.get(model, model)
                line, = ax.plot(clients, accs, marker="^", linewidth=1.8,
                                markersize=5, linestyle="--",
                                label=f"{lbl} (frac=0.4)", color=color,
                                alpha=0.7)
                if any(s > 0 for s in stds):
                    ax.fill_between(clients,
                                    [a - s for a, s in zip(accs, stds)],
                                    [a + s for a, s in zip(accs, stds)],
                                    alpha=0.08, color=line.get_color())

        ax.set_xlabel("Number of Clients")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{dataset} --- {strategy} Scalability ({'IID' if part == 'iid' else 'Non-IID'})")
        ax.legend(fontsize=7, loc="best")
        # Ensure panels share the same y-axis for fair comparison
        ax.set_ylim(y_min_global, y_max_global)

    fig.tight_layout()
    _save_or_show(fig, out_dir, f"scalability_{strategy}", save)


def _per_strategy_iid_noniid(strat_data: List[dict], strategy: str,
                              dataset: str, save: bool, out_dir: Path):
    """Grouped bars: IID vs Non-IID accuracy per model at snapshot client count."""
    nc = _STRATEGY_SNAPSHOT_CLIENTS
    filtered = [r for r in strat_data
                if r["num_clients"] == nc and r["frac"] == "1.0"]
    # Fallback: try without frac filter
    if not filtered:
        filtered = [r for r in strat_data if r["num_clients"] == nc]
    if not filtered:
        return

    models = sorted(set(r["model"] for r in filtered))
    partitions = ["iid", "non-iid"]
    x = np.arange(len(models))
    width = 0.35
    colors = ["#3498db", "#e74c3c"]
    has_std = any(r["num_seeds"] > 1 for r in filtered)

    fig, ax = plt.subplots(figsize=(max(len(models) * 2.5, 6), 5))

    for i, part in enumerate(partitions):
        vals, errs = [], []
        for model in models:
            cands = [r for r in filtered if r["model"] == model and r["partition"] == part]
            if cands:
                best = max(cands, key=lambda r: r.get("accuracy", 0))
                vals.append(best.get("accuracy", 0))
                errs.append(best.get("accuracy_std", 0))
            else:
                vals.append(0)
                errs.append(0)

        ax.bar(x + i * width, vals, width,
               yerr=errs if has_std else None,
               capsize=3 if has_std else 0,
               label=('IID' if part == 'iid' else 'Non-IID'), color=colors[i],
               edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 1, "capthick": 1})

        for j, (v, e) in enumerate(zip(vals, errs)):
            if v > 0:
                tip = v + (e if has_std else 0) + 0.003
                ax.text(x[j] + i * width, tip,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset} --- {strategy}: IID vs Non-IID ({nc} clients)")
    all_vals = [r.get("accuracy", 0) for r in filtered if r.get("accuracy", 0) > 0]
    ymin = max(0, min(all_vals) - 0.05) if all_vals else 0
    ax.set_ylim(ymin, 1.05)
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, out_dir, f"iid_vs_noniid_{strategy}_{nc}c", save)


def _per_strategy_detection_rates(strat_data: List[dict], strategy: str,
                                   dataset: str, save: bool, out_dir: Path):
    """Vertically stacked ADR (top) and FPR (bottom) bar charts at snapshot clients."""
    nc = _STRATEGY_SNAPSHOT_CLIENTS
    filtered = [r for r in strat_data
                if r["num_clients"] == nc and r["partition"] == "iid"
                and r["frac"] == "1.0"]
    if not filtered:
        filtered = [r for r in strat_data
                    if r["num_clients"] == nc and r["partition"] == "iid"]
    if not filtered:
        return

    # Best per model
    models = sorted(set(r["model"] for r in filtered))
    best = {}
    for model in models:
        cands = [r for r in filtered if r["model"] == model]
        best[model] = max(cands, key=lambda r: r.get("accuracy", 0))

    labels = [MODEL_LABELS.get(m, m) for m in models]
    y = np.arange(len(models))
    has_std = any(best[m]["num_seeds"] > 1 for m in models)
    colors = [MODEL_COLORS.get(m, "#999") for m in models]

    adr = [best[m].get("detection_rate", 0) for m in models]
    fpr = [best[m].get("false_positive_rate", 0) for m in models]
    adr_err = [best[m].get("detection_rate_std", 0) for m in models] if has_std else None
    fpr_err = [best[m].get("false_positive_rate_std", 0) for m in models] if has_std else None

    fig, (ax_adr, ax_fpr) = plt.subplots(2, 1, figsize=(9, max(len(models) * 1.5, 4)))

    # ADR subplot
    ax_adr.barh(y, adr, 0.5,
                xerr=adr_err, capsize=3 if has_std else 0,
                error_kw={"linewidth": 1.2, "capthick": 1.2},
                color=colors, edgecolor="white", linewidth=0.5)
    adr_errs_safe = adr_err if adr_err else [0] * len(models)
    for i in range(len(models)):
        adr_tip = adr[i] + adr_errs_safe[i]
        ax_adr.text(adr_tip + 0.005, y[i], f"{adr[i]:.4f}", va="center", fontsize=8)
    ax_adr.set_yticks(y)
    ax_adr.set_yticklabels(labels)
    ax_adr.set_xlabel("Rate")
    ax_adr.set_title(f"{dataset} --- {strategy}: Attack Detection Rate ({nc}c, IID)")
    ax_adr.grid(axis="x", alpha=0.3)
    adr_min = min(adr) if adr else 0
    ax_adr.set_xlim(max(0, adr_min - 0.05), 1.05)

    # FPR subplot
    ax_fpr.barh(y, fpr, 0.5,
                xerr=fpr_err, capsize=3 if has_std else 0,
                error_kw={"linewidth": 1.2, "capthick": 1.2},
                color=colors, edgecolor="white", linewidth=0.5, alpha=0.6, hatch="//")
    fpr_errs_safe = fpr_err if fpr_err else [0] * len(models)
    fpr_tips = [fpr[i] + fpr_errs_safe[i] for i in range(len(models))]
    tip_max = max(fpr_tips) if fpr_tips and max(fpr_tips) > 0 else 0.001
    for i in range(len(models)):
        ax_fpr.text(fpr_tips[i] + tip_max * 0.04, y[i],
                    f"{fpr[i]:.6f}", va="center", fontsize=8)
    ax_fpr.set_yticks(y)
    ax_fpr.set_yticklabels(labels)
    ax_fpr.set_xlabel("Rate")
    ax_fpr.set_title(f"{dataset} --- {strategy}: False Positive Rate ({nc}c, IID)")
    ax_fpr.grid(axis="x", alpha=0.3)
    ax_fpr.set_xlim(0, tip_max * 1.5)

    fig.tight_layout()
    _save_or_show(fig, out_dir, f"detection_rates_{strategy}_{nc}c", save)


def _per_strategy_partial_participation(strat_data: List[dict], strategy: str,
                                         dataset: str, save: bool, out_dir: Path):
    """Grouped bars: frac=1.0 vs frac=0.4 accuracy per model at snapshot clients, non-IID."""
    nc = _STRATEGY_SNAPSHOT_CLIENTS
    # Only non-IID has frac=0.4 data
    filtered = [r for r in strat_data
                if r["num_clients"] == nc and r["partition"] == "non-iid"]
    if not filtered:
        return

    fracs_available = sorted(set(r["frac"] for r in filtered))
    if len(fracs_available) < 2:
        return  # Nothing to compare

    models = sorted(set(r["model"] for r in filtered))
    fracs = ["1.0", "0.4"]
    x = np.arange(len(models))
    width = 0.35
    colors = ["#2c3e50", "#e67e22"]
    frac_labels = ["Full Participation (frac=1.0)", "Partial Participation (frac=0.4)"]
    has_std = any(r["num_seeds"] > 1 for r in filtered)

    fig, ax = plt.subplots(figsize=(max(len(models) * 2.5, 6), 5))

    for i, frac in enumerate(fracs):
        vals, errs = [], []
        for model in models:
            cands = [r for r in filtered if r["model"] == model and r["frac"] == frac]
            if cands:
                best = max(cands, key=lambda r: r.get("accuracy", 0))
                vals.append(best.get("accuracy", 0))
                errs.append(best.get("accuracy_std", 0))
            else:
                vals.append(0)
                errs.append(0)

        ax.bar(x + i * width, vals, width,
               yerr=errs if has_std else None,
               capsize=3 if has_std else 0,
               label=frac_labels[i], color=colors[i],
               edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 1, "capthick": 1})

        for j, (v, e) in enumerate(zip(vals, errs)):
            if v > 0:
                tip = v + (e if has_std else 0) + 0.003
                ax.text(x[j] + i * width, tip,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{dataset} --- {strategy}: Partial Participation ({nc}c, Non-IID)")
    all_vals = [r.get("accuracy", 0) for r in filtered if r.get("accuracy", 0) > 0]
    ymin = max(0, min(all_vals) - 0.05) if all_vals else 0
    ax.set_ylim(ymin, 1.05)
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, out_dir, f"partial_participation_{strategy}_{nc}c", save)


def _export_strategy_latex_table(strat_data: List[dict], strategy: str, dataset: str, out_dir: Path):
    """Exports full numeric federated results for this strategy into a LaTeX table."""
    tex_file = out_dir / f"appendix_table_{strategy}.tex"
    
    # Sort data logically
    sorted_data = sorted(
        strat_data, 
        key=lambda r: (
            r["model"], 
            r["partition"], 
            r["num_clients"], 
            float(r["frac"]), 
            r.get("mu") or 0
        )
    )
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Full Experimental Results: {strategy} on {dataset}}}",
        "\\label{tab:" + f"{dataset.lower()}_{strategy.lower()}_full_results" + "}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccrrrr}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Partition} & \\textbf{Clients} & \\textbf{Frac} & \\textbf{Accuracy} & \\textbf{ADR} & \\textbf{FPR} & \\textbf{Comm (MB)} \\\\",
        "\\midrule"
    ]
    
    curr_model = None
    curr_part = None
    
    for r in sorted_data:
        m = MODEL_LABELS.get(r["model"], r["model"])
        p = "IID" if r["partition"] == "iid" else "Non-IID"
        c = str(r["num_clients"])
        frac = r["frac"]
        
        # Display model/partition only when they change for cleaner tables
        m_str = m if m != curr_model else ""
        p_str = p if p != curr_part or m != curr_model else ""
        curr_model = m
        curr_part = p
        
        acc = f"{r.get('accuracy', 0)*100:.2f}"
        if r.get('accuracy_std'):
            acc += f" {{\\tiny $\\pm${r['accuracy_std']*100:.2f}}}"
            
        adr = f"{r.get('detection_rate', 0)*100:.2f}"
        fpr = f"{r.get('false_positive_rate', 0)*100:.2f}"
        
        comm = f"{r.get('total_mb', 0):.2f}" if r.get('total_mb', 0) > 0 else "-"
        
        lines.append(f"{m_str} & {p_str} & {c} & {frac} & {acc} & {adr} & {fpr} & {comm} \\\\")
        
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}"
    ])
    
    # Save to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(tex_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    Exported table → {tex_file}")


def chart_per_strategy(datasets: List[str], save: bool, out_dir: Path):
    """Generate per-strategy report figures (scalability + IID/Non-IID + detection)."""
    print("\n Per-Strategy Reports")
    for ds in datasets:
        fed = collect_federated(ds)
        if not fed:
            print(f"    No federated data for {ds}, skipping.")
            continue

        strategies = sorted(set(r["strategy"] for r in fed))
        for strategy in strategies:
            strat_data = [r for r in fed if r["strategy"] == strategy]
            strat_out = out_dir / "per_strategy" / ds / strategy
            print(f"    {ds}/{strategy}: {len(strat_data)} configs")

            _per_strategy_scalability(strat_data, strategy, ds, save, strat_out)
            _per_strategy_iid_noniid(strat_data, strategy, ds, save, strat_out)
            _per_strategy_detection_rates(strat_data, strategy, ds, save, strat_out)
            _per_strategy_partial_participation(strat_data, strategy, ds, save, strat_out)
            
            if save:
                _export_strategy_latex_table(strat_data, strategy, ds, strat_out)

CHART_FUNCS = {
    1: chart_baseline_comparison,
    2: chart_scalability,
    3: chart_iid_vs_noniid,
    4: chart_fedavg_vs_fedprox,
    5: chart_baseline_vs_federated,
    6: chart_mu_sweep,
    8: chart_communication_cost,
    10: chart_per_strategy,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate dissertation-quality figures from eval results."
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["DataSense", "Edge-IIoT", "all"],
                        help="Dataset(s) to visualize (default: all)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to results/_figures/")
    parser.add_argument("--charts", type=int, nargs="*", default=None,
                        help="Specific chart numbers to generate (default: all)")
    args = parser.parse_args()

    apply_latex_style()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    out_dir = EVAL_DIR / "_figures"
    chart_nums = args.charts if args.charts else sorted(CHART_FUNCS.keys())

    print(f"\n{'=' * 60}")
    print(f"  Dissertation Visualizations")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Charts: {chart_nums}")
    print(f"  Output: {out_dir if args.save else 'interactive'}")
    print(f"{'=' * 60}")

    for num in chart_nums:
        func = CHART_FUNCS.get(num)
        if func:
            func(datasets, args.save, out_dir)
        else:
            print(f"\n  [!] Unknown chart number: {num}")

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
