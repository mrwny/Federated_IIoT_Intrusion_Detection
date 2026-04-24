import argparse
import re
import csv
from pathlib import Path
import numpy as np
from typing import Optional


EVAL_DIR = Path(__file__).resolve().parent.parent / "results" / "baseline"
MODEL_TYPES = ["dnn", "cnn", "xgboost", "random_forest"]


def _find_newest_run(seed_dir: Path) -> Optional[Path]:
    """Return the path of the newest timestamped run inside a seed dir."""
    runs = sorted(
        [d for d in seed_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.name
    )
    return runs[-1] if runs else None


def _parse_eval_file(eval_path: Path) -> Optional[dict]:
    """Parse nids_evaluation.txt and extract key metrics."""
    text = eval_path.read_text()
    metrics = {}

    patterns = {
        'accuracy': r'Overall Accuracy:\s*([\d.]+)',
        'detection_rate': r'Attack Detection Rate:\s*([\d.]+)',
        'false_positive_rate': r'False Positive Rate:\s*([\d.]+)',
    }

    cr_patterns = {
        'benign_precision': r'Benign\s+([\d.]+)\s+[\d.]+\s+[\d.]+',
        'benign_recall': r'Benign\s+[\d.]+\s+([\d.]+)\s+[\d.]+',
        'benign_f1': r'Benign\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'attack_precision': r'Attack\s+([\d.]+)\s+[\d.]+\s+[\d.]+',
        'attack_recall': r'Attack\s+[\d.]+\s+([\d.]+)\s+[\d.]+',
        'attack_f1': r'Attack\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
    }

    for key, pattern in {**patterns, **cr_patterns}.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))

    return metrics if metrics else None


def aggregate_model(dataset: str, model_type: str) -> Optional[dict]:
    """Aggregate results across all seeds for one model type."""
    baseline_dir = EVAL_DIR / dataset / model_type
    if not baseline_dir.exists():
        return None

    seed_dirs = sorted(baseline_dir.glob("seed_*"))
    if not seed_dirs:
        return None

    all_metrics = []
    seeds_found = []

    for seed_dir in seed_dirs:
        seed_val = seed_dir.name.replace("seed_", "")
        newest_run = _find_newest_run(seed_dir)
        if newest_run is None:
            continue

        eval_file = newest_run / "nids_evaluation.txt"
        if not eval_file.exists():
            continue

        metrics = _parse_eval_file(eval_file)
        if metrics:
            all_metrics.append(metrics)
            seeds_found.append(seed_val)

    if len(all_metrics) < 2:
        print(f"  {model_type}: Only {len(all_metrics)} seed(s) found, skipping aggregation")
        return None

    # Compute mean ± std for each metric
    metric_keys = all_metrics[0].keys()
    aggregated = {
        'model': model_type,
        'num_seeds': len(all_metrics),
        'seeds': seeds_found,
    }

    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_values'] = values

    return aggregated


def save_summary(results: list[dict], dataset: str):
    """Save aggregated results to text and CSV files."""
    output_dir = EVAL_DIR / dataset / "_aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  Baseline Results - {dataset} (Mean +/- Std across seeds)\n")
        f.write(f"{'='*70}\n\n")

        for r in results:
            f.write(f"--- {r['model'].upper()} ({r['num_seeds']} seeds: {', '.join(r['seeds'])}) ---\n")
            for key in ['accuracy', 'detection_rate', 'false_positive_rate',
                        'benign_precision', 'benign_recall', 'benign_f1',
                        'attack_precision', 'attack_recall', 'attack_f1']:
                mean_key = f'{key}_mean'
                std_key = f'{key}_std'
                if mean_key in r:
                    label = key.replace('_', ' ').title()
                    f.write(f"  {label:30s}: {r[mean_key]:.10f} +/- {r[std_key]:.10f}\n")
            f.write("\n")

        f.write(f"{'='*70}\n")

    csv_path = output_dir / "summary.csv"
    metric_keys = ['accuracy', 'detection_rate', 'false_positive_rate',
                   'benign_precision', 'benign_recall', 'benign_f1',
                   'attack_precision', 'attack_recall', 'attack_f1']

    fieldnames = ['model', 'num_seeds']
    for k in metric_keys:
        fieldnames.extend([f'{k}_mean', f'{k}_std'])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {'model': r['model'], 'num_seeds': r['num_seeds']}
            for k in metric_keys:
                row[f'{k}_mean'] = f"{r.get(f'{k}_mean', 0):.10f}"
                row[f'{k}_std'] = f"{r.get(f'{k}_std', 0):.10f}"
            writer.writerow(row)

    print(f"\nAggregated results saved to:")
    print(f"  {txt_path}")
    print(f"  {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate baseline results across seeds')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['DataSense', 'Edge-IIoT', 'all'],
                        help='Dataset to aggregate (default: all)')
    args = parser.parse_args()

    datasets = ['DataSense', 'Edge-IIoT'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        print(f"\nAggregating baseline results for {dataset}...")
        print(f"Looking in: {EVAL_DIR / dataset}\n")

        results = []
        for model_type in MODEL_TYPES:
            agg = aggregate_model(dataset, model_type)
            if agg:
                results.append(agg)

        if results:
            save_summary(results, dataset)
        else:
            print("\nNo multi-seed results found to aggregate.")


if __name__ == "__main__":
    main()
