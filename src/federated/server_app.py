import sys
import csv
import json
import pickle
from pathlib import Path
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx, FedXgbBagging
from flwr.common.record.array import Array
from matplotlib import pyplot as plt
from AttentionWeightedFedXgbBagging import AttentionWeightedFedXgbBagging, attention_weighted_inference
import seaborn as sns
import xgboost as xgb
import logging

LOGGER_NAME = "flwr"
FLOWER_LOGGER = logging.getLogger(LOGGER_NAME)
FLOWER_LOGGER.setLevel(logging.DEBUG)
log = FLOWER_LOGGER.log
matplotlib.use('Agg')

from task import (
    DATASETS_CONFIG,
    set_global_seed,
    load_server_data, load_server_data_numpy,
    get_xgb_params, evaluate_xgb, evaluate_rf, combine_rf_models,
    _find_optimal_threshold,
)
project_root = Path(__file__).resolve().parents[2]
_baseline_models_dir = str(project_root / "src" / "baseline" / "models")
if _baseline_models_dir not in sys.path:
    sys.path.insert(0, _baseline_models_dir)
from dnn import DNN, CNN, evaluate_model_DNN


def set_modern_style():
    sns.set_theme(style="whitegrid", context="talk")
    modern_palette = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#34495e"]
    sns.set_palette(modern_palette)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.titlesize': 18,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.frameon': False,
    })


app = ServerApp()

def _save_round_metrics_csv(round_data: list[dict], output_dir: str):
    """Save per-round metrics to CSV."""
    if not round_data:
        return
    filepath = f"{output_dir}round_metrics.csv"
    keys = round_data[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(round_data)
    print(f"[Metrics] Saved per-round metrics to {filepath}")


def _save_comm_cost(model_size_bytes: int, num_rounds: int,
                    num_clients: int, fraction_train: float,
                    output_dir: str):
    """Compute and save communication cost summary as JSON."""
    clients_per_round = max(1, int(num_clients * fraction_train))
    # Each round: server sends model to clients (download) + clients send back (upload)
    per_round_bytes = model_size_bytes * 2 * clients_per_round
    total_bytes = per_round_bytes * num_rounds

    summary = {
        'model_size_bytes': model_size_bytes,
        'model_size_KB': round(model_size_bytes / 1024, 2),
        'clients_per_round': clients_per_round,
        'num_rounds': num_rounds,
        'per_round_bytes': per_round_bytes,
        'per_round_KB': round(per_round_bytes / 1024, 2),
        'total_bytes': total_bytes,
        'total_MB': round(total_bytes / (1024 * 1024), 4),
    }
    filepath = f"{output_dir}comm_cost.json"
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[CommCost] Model={summary['model_size_KB']} KB, "
          f"Per-round={summary['per_round_KB']} KB, "
          f"Total={summary['total_MB']} MB -> {filepath}")
    return summary


def _plot_convergence(round_data: list[dict], output_dir: str, title_prefix: str = ""):
    """Plot per-round convergence curves (accuracy, detection rate, FPR, loss)."""
    if not round_data:
        return

    rounds = [d['round'] for d in round_data]
    acc = [d.get('accuracy', 0) for d in round_data]
    dr = [d.get('detection_rate', 0) for d in round_data]
    fpr = [d.get('false_positive_rate', 0) for d in round_data]
    loss = [d.get('train_loss', None) for d in round_data]
    has_loss = any(v is not None for v in loss)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Plot A: Training Loss
    if has_loss:
        loss_vals = [v for v in loss if v is not None]
        loss_rnds = [r for r, v in zip(rounds, loss) if v is not None]
        axes[0, 0].plot(loss_rnds, loss_vals, 'purple', marker='o', linewidth=2)
        if len(loss_vals) > 3:
            window = 3
            smoothed = [sum(loss_vals[max(0, i-window):i+1]) /
                        len(loss_vals[max(0, i-window):i+1])
                        for i in range(len(loss_vals))]
            axes[0, 0].plot(loss_rnds, smoothed, 'k--', alpha=0.5, label='Smoothed')
            axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, "No Loss Data", ha='center', va='center',
                        transform=axes[0, 0].transAxes)
    axes[0, 0].set_title(f"{title_prefix}Training Loss", fontweight='bold')
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot B: Accuracy (auto-scaled)
    axes[0, 1].plot(rounds, acc, 'b-', marker='o', label='Accuracy')
    if acc:
        min_a, max_a = min(acc), max(acc)
        buf = (max_a - min_a) * 0.1 if max_a != min_a else 0.01
        axes[0, 1].set_ylim(min_a - buf, max_a + buf)
    axes[0, 1].set_title(f"{title_prefix}Accuracy (Auto-Scaled)", fontweight='bold')
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot C: Detection Rate
    axes[1, 0].plot(rounds, dr, 'g-', marker='s', linewidth=2, label='Detection Rate')
    axes[1, 0].set_title(f"{title_prefix}Detection Rate", fontweight='bold')
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Detection Rate")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot D: Detection Rate vs FPR (dual axis)
    ax_dr = axes[1, 1]
    ax_fpr = ax_dr.twinx()
    ax_dr.plot(rounds, dr, 'g-', marker='s', label='Detection Rate')
    ax_fpr.plot(rounds, fpr, 'r--', marker='^', label='False Positive Rate')
    ax_dr.set_ylabel("Detection Rate", color='g')
    ax_fpr.set_ylabel("False Positive Rate", color='r')
    ax_dr.set_xlabel("Round")
    ax_dr.set_title(f"{title_prefix}DR vs FPR", fontweight='bold')
    lines1, labels1 = ax_dr.get_legend_handles_labels()
    lines2, labels2 = ax_fpr.get_legend_handles_labels()
    ax_dr.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.savefig(f"{output_dir}convergence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Metrics] Saved convergence plot to {output_dir}convergence.png")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    set_modern_style()

    # Read run config
    model_type: str = context.run_config.get("model-type", "DNN")
    num_rounds: int = context.run_config["num-server-rounds"]
    partition_type: str = context.run_config["partition-type"]
    num_partitions: int = len(list(grid.get_node_ids()))
    seed: int = int(context.run_config.get("seed", 42))
    dataset: str = context.run_config.get("dataset", "DataSense")

    set_global_seed(seed)

    if model_type == "XGBoost":
        _run_xgboost(grid, context, num_rounds, num_partitions, partition_type, seed, dataset)
    elif model_type == "AttentionXGBoost":
        _run_attention_xgboost(grid, context, num_rounds, num_partitions, partition_type, seed, dataset)
    elif model_type == "RandomForest":
        _run_random_forest(grid, context, num_partitions, partition_type, seed, dataset)
    else:
        _run_dnn(grid, context, model_type, num_rounds, num_partitions, partition_type, seed, dataset)

def _run_dnn(grid, context, model_type, num_rounds, num_partitions, partition_type, seed, dataset):
    fraction_train: float = context.run_config["fraction-train"]
    min_train_nodes: int = context.run_config.get("min-train-nodes", 2)
    lr: float = context.run_config["lr"]
    strategy_name: str = context.run_config["strategy"]
    proximal_mu: float = context.run_config.get("proximal-mu", 0.1)

    input_size = DATASETS_CONFIG[dataset]['input_size']
    if model_type == "DNN":
        global_model = DNN(input_size, 2)
    else:
        model_type = "CNN"
        global_model = CNN(input_size, 2)

    if torch.cuda.is_available():
        log(logging.INFO, ">>> Using GPU <<<")
    else:
        log(logging.INFO, ">>> Using CPU <<<")

    arrays = ArrayRecord(global_model.state_dict())
    strategy = (
        FedAvg(fraction_train=fraction_train, min_train_nodes=min_train_nodes)
        if strategy_name == "FedAvg"
        else FedProx(fraction_train=fraction_train, min_train_nodes=min_train_nodes, proximal_mu=proximal_mu)
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    state_dict = result.arrays.to_torch_state_dict()
    global_model.load_state_dict(state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model.to(device)

    final_results = evaluate_model_DNN(
        net=global_model,
        testloader=load_server_data(dataset=dataset)[1],
        device=device,
    )

    if strategy_name == "FedProx":
        output_dir = str(project_root / f"results/Federated/{dataset}/{strategy_name}/{model_type}/{partition_type}/{num_partitions}/frac_{fraction_train}/mu_{proximal_mu}/seed_{seed}") + "/"
    else:
        output_dir = str(project_root / f"results/Federated/{dataset}/{strategy_name}/{model_type}/{partition_type}/{num_partitions}/frac_{fraction_train}/seed_{seed}") + "/"

    round_data = []
    for rnd, metrics in sorted(result.evaluate_metrics_clientapp.items()):
        entry = {
            'round': rnd,
            'accuracy': metrics.get('accuracy', 0),
            'detection_rate': metrics.get('detection_rate', 0),
            'false_positive_rate': metrics.get('false_positive_rate', 0),
        }
        if rnd in result.train_metrics_clientapp:
            entry['train_loss'] = result.train_metrics_clientapp[rnd].get('train_loss', None)
        round_data.append(entry)

    # Measure the ArrayRecord that Flower transmits
    model_arrays = ArrayRecord(global_model.state_dict())
    model_size = sum(len(arr.data) if hasattr(arr, 'data') else arr.numpy().nbytes for arr in model_arrays.values())

    _save_dnn_results(
        global_model, output_dir, final_results, num_rounds,
        strategy_name, num_partitions, proximal_mu, device, dataset,
    )
    _save_round_metrics_csv(round_data, output_dir)
    _plot_convergence(round_data, output_dir,
                      title_prefix=f"{strategy_name}/{model_type} - ")
    _save_comm_cost(model_size, num_rounds, num_partitions,
                    fraction_train, output_dir)


def _save_dnn_results(global_model, output_dir, final_results, num_rounds,
                      algorithm, num_partitions, proximal_mu, device, dataset):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}results.txt", "w") as f:
        f.write(f"Final results after {num_rounds} rounds using {algorithm}:\n")
        f.write(f"Number of participating clients: {num_partitions}\n")
        f.write(f"Proximal mu: {proximal_mu}\n")
        f.write(f"Accuracy: {final_results['accuracy']:.10f}\n")
        f.write(f"Detection Rate: {final_results['detection_rate']:.10f}\n")
        f.write(f"False Positive Rate: {final_results['false_positive_rate']:.10f}\n")

    global_model.eval()
    all_preds = []
    all_labels = []
    _, test_loader = load_server_data(dataset=dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = global_model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    _plot_confusion_matrix(all_labels, all_preds, output_dir)
    print(f"Results saved to {output_dir}")

def _run_xgboost(grid, context, num_rounds, num_partitions, partition_type, seed, dataset):
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 0.5)
    min_train_nodes: int = context.run_config.get("min-train-nodes", 2)
    params = get_xgb_params(context.run_config)

    # Init with empty model, XGBooster created on client side in round 1
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Reconstruct final global model
    xgb_params = {k: v for k, v in params.items() if k not in ('n_estimators',)}
    bst = xgb.Booster(params=xgb_params)
    global_model_bytes = bytearray(result.arrays["0"].numpy().tobytes())
    bst.load_model(global_model_bytes)

    # Server-side evaluation
    # Use a held-out validation split for threshold tuning (matching baseline methodology)
    X_train_full, y_train_full, X_test, y_test = load_server_data_numpy(dataset=dataset)
    _, X_val, _, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed, stratify=y_train_full
    )
    final_results = evaluate_xgb(
        bytes(global_model_bytes),
        X_test,
        y_test,
        params,
        X_threshold=X_val,
        y_threshold=y_val,
    )

    # Save results
    output_dir = str(project_root / f"results/Federated/{dataset}/FedXgbBagging/XGBoost/{partition_type}/{num_partitions}/frac_{fraction_train}/seed_{seed}") + "/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}results.txt", "w") as f:
        f.write(f"Final results after {num_rounds} rounds using FedXgbBagging:\n")
        f.write(f"Number of participating clients: {num_partitions}\n")
        f.write(f"XGBoost params: {params}\n")
        f.write(f"Optimal Threshold: {final_results['threshold']:.4f}\n")
        f.write(f"Accuracy: {final_results['accuracy']:.10f}\n")
        f.write(f"Detection Rate: {final_results['detection_rate']:.10f}\n")
        f.write(f"False Positive Rate: {final_results['false_positive_rate']:.10f}\n")

    # Save model
    bst.save_model(f"{output_dir}final_model.json")

    # Confusion matrix — use the same optimal threshold as evaluation
    test_dmatrix = xgb.DMatrix(X_test, label=y_test)
    y_probs = bst.predict(test_dmatrix)
    y_pred = (y_probs > final_results['threshold']).astype(int)
    _plot_confusion_matrix(y_test, y_pred, output_dir)

    round_data = []
    if hasattr(result, 'evaluate_metrics_clientapp'):
        for rnd, metrics in sorted(result.evaluate_metrics_clientapp.items()):
            entry = {
                'round': rnd,
                'accuracy': metrics.get('accuracy', 0),
                'detection_rate': metrics.get('detection_rate', 0),
                'false_positive_rate': metrics.get('false_positive_rate', 0),
            }
            round_data.append(entry)

    if round_data:
        _save_round_metrics_csv(round_data, output_dir)
        _plot_convergence(round_data, output_dir,
                          title_prefix="FedXgbBagging/XGBoost - ")

    # Communication cost
    model_size = len(global_model_bytes)
    _save_comm_cost(model_size, num_rounds, num_partitions,
                    fraction_train, output_dir)

    _plot_federated_feature_importance(global_model_bytes, output_dir, dataset, "FedXgbBagging", is_attention=False)

    print(f"XGBoost federated results saved to {output_dir}")


def _run_attention_xgboost(grid, context, num_rounds, num_partitions,
                           partition_type, seed, dataset):
    """Run Attention-Weighted FedXgbBagging using the subclass strategy.

    At every round the ``AttentionWeightedFedXgbBagging`` strategy intercepts client
    trees, evaluates them on the server holdout, and tags them with attention
    weights.  After training, inference uses the weighted global bag.
    """
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 0.5)
    min_train_nodes: int = context.run_config.get("min-train-nodes", 2)

    X_train_full, y_train_full, X_test, y_test = load_server_data_numpy(
        dataset=dataset
    )
    X_attention, X_threshold, y_attention, y_threshold = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, random_state=seed, stratify=y_train_full,
    )

    strategy = AttentionWeightedFedXgbBagging(
        server_val_X=X_attention,
        server_val_y=y_attention,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
    )

    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    print(f"[AttentionFedXgb] Training complete: {num_rounds} rounds, "
          f"{len(strategy.weighted_global_bag)} total entries in weighted bag")

    global_model_bytes = result.arrays["0"].numpy().tobytes()

    # Use per-client full models from the final round (proper boosters)
    # with that round's attention weights for soft-voting.
    last_round_entries = []
    if strategy.weighted_global_bag:
        last_round_entries = [
            e for e in strategy.weighted_global_bag
            if e['round'] == num_rounds
        ]
        if not last_round_entries:
            # Fallback: use entries from the last available round
            max_round = max(e['round'] for e in strategy.weighted_global_bag)
            last_round_entries = [
                e for e in strategy.weighted_global_bag
                if e['round'] == max_round
            ]

    if last_round_entries:
        val_weighted_probs = attention_weighted_inference(
            last_round_entries, X_threshold
        )
        test_weighted_probs = attention_weighted_inference(
            last_round_entries, X_test
        )
    else:
        print(
            "[AttentionFedXgb] WARNING: Attention bag is empty; "
            "falling back to final aggregated global model for inference"
        )
        final_bst = xgb.Booster()
        final_bst.load_model(bytearray(global_model_bytes))
        val_weighted_probs = final_bst.predict(xgb.DMatrix(X_threshold))
        test_weighted_probs = final_bst.predict(xgb.DMatrix(X_test))

    optimal_threshold = _find_optimal_threshold(val_weighted_probs, y_threshold)
    y_pred = (test_weighted_probs > optimal_threshold).astype(int)

    # Final metrics
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    output_dir = str(project_root / f"results/Federated/{dataset}/AttentionFedXgb/XGBoost/{partition_type}/{num_partitions}/frac_{fraction_train}/seed_{seed}") + "/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}results.txt", "w") as f:
        f.write(f"Final results using Attention-Weighted FedXgbBagging:\n")
        f.write(f"Total rounds: {num_rounds}\n")
        f.write(f"Number of participating clients: {num_partitions}\n")
        f.write(f"XGBoost params: {get_xgb_params(context.run_config)}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"Accuracy: {accuracy:.10f}\n")
        f.write(f"Detection Rate: {detection_rate:.10f}\n")
        f.write(f"False Positive Rate: {false_positive_rate:.10f}\n")
        f.write(f"\n--- Per-Round Attention Weights ---\n")
        for rnd_log in strategy.round_attention_log:
            f.write(f"Round {rnd_log['round']}: α = {rnd_log['alpha']}\n")
            for j, cm_entry in enumerate(rnd_log['client_metrics']):
                f.write(f"  Client {j}: DR={cm_entry['dr']:.4f}, "
                        f"FPR={cm_entry['fpr']:.4f}, "
                        f"Score={cm_entry['score']:.4f}\n")

    # Confusion matrix
    _plot_confusion_matrix(y_test, y_pred, output_dir)

    # Per-round metrics
    round_data = []
    if hasattr(result, 'evaluate_metrics_clientapp'):
        for rnd, metrics in sorted(result.evaluate_metrics_clientapp.items()):
            round_data.append({
                'round': rnd,
                'accuracy': metrics.get('accuracy', 0),
                'detection_rate': metrics.get('detection_rate', 0),
                'false_positive_rate': metrics.get('false_positive_rate', 0),
            })
    if round_data:
        _save_round_metrics_csv(round_data, output_dir)
        _plot_convergence(round_data, output_dir,
                          title_prefix="AttentionFedXgb/XGBoost - ")

    # Communication cost (same as standard FedXgbBagging)
    model_size = len(global_model_bytes)
    _save_comm_cost(model_size, num_rounds, num_partitions,
                    fraction_train, output_dir)

    _plot_federated_feature_importance(last_round_entries, output_dir, dataset, "AttentionFedXgb", is_attention=True)

    # Attention weights JSON
    attention_data = {
        'num_rounds': num_rounds,
        'bag_size': len(strategy.weighted_global_bag),
        'per_round_log': strategy.round_attention_log,
    }
    with open(f"{output_dir}attention_weights.json", 'w') as f:
        json.dump(attention_data, f, indent=2)

    print(f"Attention-weighted XGBoost results saved to {output_dir}")



def _run_random_forest(grid, context, num_partitions, partition_type, seed, dataset):
    """Federated Random Forest via single-round federated bagging.

    Each client trains a subset of trees on local data. The server combines
    all client trees into one global forest.
    """

    # Step 1: Send train messages to all clients
    node_ids = list(grid.get_node_ids())
    placeholder = ArrayRecord([np.array([], dtype=np.uint8)])
    train_messages = []
    for nid in node_ids:
        content = RecordDict({
            "arrays": placeholder,
            "config": ConfigRecord({}),
        })
        msg = grid.create_message(
            content=content,
            message_type="train",
            dst_node_id=nid,
            group_id="rf_train",
        )
        train_messages.append(msg)

    # Step 2: Collect responses
    replies = list(grid.send_and_receive(train_messages))
    print(f"[Server] Received {len(replies)} RF training responses")

    # Step 3: Combine all client forests
    client_model_bytes = []
    for reply in replies:
        mb = reply.content["arrays"]["0"].numpy().tobytes()
        client_model_bytes.append(mb)

    global_model_bytes = combine_rf_models(client_model_bytes)
    global_rf = pickle.loads(global_model_bytes)
    print(f"[Server] Combined forest has {len(global_rf.estimators_)} trees")

    # Step 4: Server-side evaluation
    # Use a held-out validation split for threshold tuning (matching baseline methodology)
    X_train_full, y_train_full, X_test, y_test = load_server_data_numpy(dataset=dataset)
    _, X_val, _, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed, stratify=y_train_full
    )
    final_results = evaluate_rf(
        global_model_bytes,
        X_test,
        y_test,
        X_threshold=X_val,
        y_threshold=y_val,
    )

    # Step 5: Save results
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    output_dir = str(project_root / f"results/Federated/{dataset}/FedRF/RandomForest/{partition_type}/{num_partitions}/frac_{fraction_train}/seed_{seed}") + "/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}results.txt", "w") as f:
        f.write(f"Final results using Federated Random Forest (bagging):\n")
        f.write(f"Number of participating clients: {num_partitions}\n")
        f.write(f"Total trees: {len(global_rf.estimators_)}\n")
        f.write(f"Optimal Threshold: {final_results['threshold']:.10f}\n")
        f.write(f"Accuracy: {final_results['accuracy']:.10f}\n")
        f.write(f"Detection Rate: {final_results['detection_rate']:.10f}\n")
        f.write(f"False Positive Rate: {final_results['false_positive_rate']:.10f}\n")

    # Save model
    with open(f"{output_dir}final_model.pkl", "wb") as f:
        pickle.dump(global_rf, f)

    # Confusion matrix
    y_probs = global_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > final_results['threshold']).astype(int)
    _plot_confusion_matrix(y_test, y_pred, output_dir)

    round_data = [{
        'round': 1,
        'accuracy': final_results['accuracy'],
        'detection_rate': final_results['detection_rate'],
        'false_positive_rate': final_results['false_positive_rate'],
        'train_loss': None
    }]
    _save_round_metrics_csv(round_data, output_dir)

    # Upload: each client sends its local RF model
    total_upload = sum(len(mb) for mb in client_model_bytes)
    # Download: server sends combined model to each client for evaluation
    total_download = len(global_model_bytes) * num_partitions
    comm_summary = {
        'num_rounds': 1,
        'num_clients': num_partitions,
        'total_upload_bytes': total_upload,
        'total_download_bytes': total_download,
        'total_bytes': total_upload + total_download,
        'total_MB': round((total_upload + total_download) / (1024 * 1024), 4),
        'avg_client_model_KB': round(total_upload / num_partitions / 1024, 2),
        'global_model_KB': round(len(global_model_bytes) / 1024, 2),
    }
    with open(f"{output_dir}comm_cost.json", 'w') as f:
        json.dump(comm_summary, f, indent=2)
    print(f"[CommCost] RF total={comm_summary['total_MB']} MB")

    # Send combined model to clients for local evaluation
    global_model_np = np.frombuffer(global_model_bytes, dtype=np.uint8)
    eval_arrays = ArrayRecord([global_model_np])
    eval_messages = []
    for nid in node_ids:
        content = RecordDict({
            "arrays": eval_arrays,
            "config": ConfigRecord({}),
        })
        msg = grid.create_message(
            content=content,
            message_type="evaluate",
            dst_node_id=nid,
            group_id="rf_eval",
        )
        eval_messages.append(msg)

    eval_replies = list(grid.send_and_receive(eval_messages))
    total_acc = 0.0
    total_examples = 0
    for reply in eval_replies:
        m = reply.content["metrics"]
        n = int(m["num-examples"])
        total_acc += float(m["accuracy"]) * n
        total_examples += n
    if total_examples > 0:
        weighted_acc = total_acc / total_examples
        print(f"[Server] Weighted client-side accuracy: {weighted_acc:.4f}")

    print(f"RandomForest federated results saved to {output_dir}")


def _plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title("Final Confusion Matrix", fontsize=14, fontweight='bold')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(f"{output_dir}confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_federated_feature_importance(model_data, output_dir: str, dataset: str, label: str,
                                       is_attention: bool = False, top_n: int = 20):
    """Plot federated feature importance, accommodating attention-weighted bagging ensembles."""
    try:
        importance_dict = {}

        if is_attention:
            # model_data is a list of dicts [{'full_model': bytes, 'alpha': float}]
            total_alpha = 0.0
            for entry in model_data:
                bst = xgb.Booster()
                bst.load_model(bytearray(entry['full_model']))
                gains = bst.get_score(importance_type="gain")
                alpha = entry['alpha']
                total_alpha += alpha
                for feat, gain in gains.items():
                    importance_dict[feat] = importance_dict.get(feat, 0.0) + (gain * alpha)
            if total_alpha > 0:
                for feat in importance_dict:
                    importance_dict[feat] /= total_alpha
        else:
            # model_data is purely bytes of the global reconstructured bag
            bst = xgb.Booster()
            bst.load_model(bytearray(model_data))
            importance_dict = bst.get_score(importance_type="gain")

        if not importance_dict:
            return

        import os, json
        feature_file = f'datasets/{dataset}/processed/feature_names.json'
        cols = []
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                cols = json.load(f)

        mapped_importance = {}
        for key, value in importance_dict.items():
            if key.startswith('f'):
                try:
                    idx = int(key.replace('f', ''))
                    name = cols[idx] if idx < len(cols) else key
                except Exception:
                    name = key
            else:
                name = key

            # Aggregate hashed buckets back to their conceptual parent feature
            if '_hash_' in name:
                base_name = name.split('_hash_')[0]
                mapped_importance[base_name] = mapped_importance.get(base_name, 0.0) + value
            elif '_class_' in name or (name.split('_')[-1].isdigit() and len(name.split('_')) > 2): 
                base_name = '_'.join(name.split('_')[:-1])
                mapped_importance[base_name] = mapped_importance.get(base_name, 0.0) + value
            else:
                mapped_importance[name] = mapped_importance.get(name, 0.0) + value

        sorted_imp = sorted(mapped_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [k for k, _ in sorted_imp]
        gains = [v for _, v in sorted_imp]

        # LaTeX Academic Theme to match dissertation
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })
        def _t(s): return s.replace("—", "---").replace("&", "\\&")

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, gains[::-1], color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features[::-1], fontsize=9)
        ax.set_xlabel("Log Average Gain (Contribution to Accuracy)")
        ax.set_xscale("log")
        ax.set_title(_t(f"{dataset} --- {label} Feature Importance (Top {len(features)} by Gain)"),
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        plt.savefig(f"{output_dir}feature_importance.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    [Feature Importance] Saved -> {output_dir}feature_importance.pdf")
    except Exception as e:
        print(f"    [Feature Importance Error] Could not generate feature importance: {e}")