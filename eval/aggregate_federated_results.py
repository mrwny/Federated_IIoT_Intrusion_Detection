import argparse
import csv
import re
from pathlib import Path
from typing import Optional
import numpy as np


EVAL_DIR = Path(__file__).resolve().parent.parent / "results" / "federated"


def parse_results(results_path: Path) -> Optional[dict]:
    """Parse the results.txt file into a dict of float metrics."""
    text = results_path.read_text()
    data = {}
    for line in text.strip().splitlines():
        if ':' not in line:
            continue
        key, val = line.split(':', 1)
        key = key.strip().lower().replace(' ', '_')
        val = val.strip()
        try:
            data[key] = float(val)
        except ValueError:
            data[key] = val
    return data if data else None


def aggregate_convergence_csvs(seed_dirs: list) -> Optional[list]:
    """Aggregate per-round metrics across seeds (mean ± std per round)."""
    from collections import defaultdict
    rounds_data = defaultdict(lambda: defaultdict(list))

    for sd in seed_dirs:
        csv_file = sd / "round_metrics.csv"
        if not csv_file.exists():
            continue
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rnd = int(row['round'])
                for key, val in row.items():
                    if key == 'round':
                        continue
                    try:
                        rounds_data[rnd][key].append(float(val))
                    except (ValueError, TypeError):
                        pass

    if not rounds_data:
        return None

    aggregated = []
    for rnd in sorted(rounds_data.keys()):
        entry = {'round': rnd}
        for key, values in rounds_data[rnd].items():
            entry[f'{key}_mean'] = np.mean(values)
            entry[f'{key}_std'] = np.std(values)
        aggregated.append(entry)

    return aggregated


def find_and_aggregate(base_dir: Path) -> list[dict]:
    """Walk the directory tree to find seed_* groups and aggregate each."""
    results = []

    # Find all directories that contain seed_* subdirs
    seed_parent_dirs = set()
    for seed_dir in base_dir.rglob("seed_*"):
        if seed_dir.is_dir() and (seed_dir / "results.txt").exists():
            seed_parent_dirs.add(seed_dir.parent)

    for parent in sorted(seed_parent_dirs):
        seed_dirs = sorted(parent.glob("seed_*"))
        seed_dirs_with_results = [sd for sd in seed_dirs if (sd / "results.txt").exists()]

        if len(seed_dirs_with_results) < 2:
            continue

        # Parse all seed results
        all_metrics = []
        seeds_found = []
        for sd in seed_dirs_with_results:
            metrics = parse_results(sd / "results.txt")
            if metrics:
                all_metrics.append(metrics)
                seeds_found.append(sd.name.replace("seed_", ""))

        if len(all_metrics) < 2:
            continue

        # Compute mean ± std
        rel_path = parent.relative_to(base_dir)
        agg = {
            'config': str(rel_path),
            'num_seeds': len(all_metrics),
            'seeds': seeds_found,
        }

        # Only aggregate numeric metrics
        numeric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
            if values:
                agg[f'{key}_mean'] = np.mean(values)
                agg[f'{key}_std'] = np.std(values)

        # Aggregate convergence
        conv = aggregate_convergence_csvs(seed_dirs_with_results)
        if conv:
            agg['convergence'] = conv

        results.append(agg)

    return results


def save_summary(results: list[dict], output_dir: Path):
    """Save aggregated results to text and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "federated_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"{'='*80}\n")
        f.write(f"  Federated Results — Mean ± Std Across Seeds\n")
        f.write(f"{'='*80}\n\n")

        for r in results:
            f.write(f"--- {r['config']} ({r['num_seeds']} seeds: {', '.join(r['seeds'])}) ---\n")
            for key in ['accuracy', 'detection_rate', 'false_positive_rate']:
                mean_key = f'{key}_mean'
                std_key = f'{key}_std'
                if mean_key in r:
                    label = key.replace('_', ' ').title()
                    f.write(f"  {label:30s}: {r[mean_key]:.6f} ± {r[std_key]:.6f}\n")
            f.write("\n")

        f.write(f"{'='*80}\n")

    csv_path = output_dir / "federated_summary.csv"
    metric_keys = ['accuracy', 'detection_rate', 'false_positive_rate']
    fieldnames = ['config', 'num_seeds']
    for k in metric_keys:
        fieldnames.extend([f'{k}_mean', f'{k}_std'])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {'config': r['config'], 'num_seeds': r['num_seeds']}
            for k in metric_keys:
                row[f'{k}_mean'] = f"{r.get(f'{k}_mean', 0):.6f}"
                row[f'{k}_std'] = f"{r.get(f'{k}_std', 0):.6f}"
            writer.writerow(row)
    for r in results:
        if 'convergence' not in r:
            continue
        conf_name = r['config'].replace('/', '_').replace(' ', '_')
        conv_path = output_dir / f"convergence_{conf_name}.csv"
        conv_data = r['convergence']
        if conv_data:
            keys = conv_data[0].keys()
            with open(conv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for entry in conv_data:
                    writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in entry.items()})

    print(f"\nAggregated results saved to:")
    print(f"  {txt_path}")
    print(f"  {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate federated results across seeds')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['DataSense', 'Edge-IIoT', 'all'],
                        help='Dataset to aggregate (default: all)')
    args = parser.parse_args()

    datasets = ['DataSense', 'Edge-IIoT'] if args.dataset == 'all' else [args.dataset]

    for dataset in datasets:
        base_dir = EVAL_DIR / dataset

        if not base_dir.exists():
            print(f"Warning: {base_dir} does not exist, skipping.")
            continue

        print(f"\nAggregating federated results from: {base_dir}\n")

        results = find_and_aggregate(base_dir)

        if not results:
            print("No multi-seed result groups found.")
            continue
        output_dir = base_dir / "_aggregated"
        save_summary(results, output_dir)


if __name__ == "__main__":
    main()
