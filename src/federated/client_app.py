import random
import sys
from pathlib import Path

import numpy as np
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

_baseline_models_dir = str(Path(__file__).resolve().parents[2] / "src" / "baseline" / "models")
if _baseline_models_dir not in sys.path:
    sys.path.insert(0, _baseline_models_dir)
from dnn import DNN, CNN

from task import (
    DATASETS_CONFIG,
    load_partitioned_data, load_partitioned_data_non_iid,
    load_partitioned_data_numpy, load_partitioned_data_non_iid_numpy,
    get_xgb_params, train_xgb, evaluate_xgb,
    get_rf_params, train_rf, evaluate_rf,
)
from task import test as test_fn
from task import train as train_fn

# Flower ClientApp
app = ClientApp()


def _get_model_type(context: Context) -> str:
    return context.run_config.get("model-type", "DNN")


def _load_numpy_partition(context: Context):
    """Load partitioned data as numpy arrays based on config."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition_type = context.run_config["partition-type"]
    seed = int(context.run_config.get("seed", 42))
    dataset = context.run_config.get("dataset", "DataSense")
    alpha_value = float(context.run_config.get("alpha", 0.1)) if partition_type == "non-iid" else None
    if partition_type == "iid":
        return load_partitioned_data_numpy(partition_id, num_partitions, seed=seed, dataset=dataset)
    else:
        return load_partitioned_data_non_iid_numpy(partition_id, num_partitions, alpha=alpha_value, seed=seed, dataset=dataset)


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    model_type = _get_model_type(context)

    if model_type in ("XGBoost", "AttentionXGBoost"):
        return _train_xgboost(msg, context)
    elif model_type == "RandomForest":
        return _train_random_forest(msg, context)
    else:
        return _train_dnn(msg, context)


def _train_dnn(msg: Message, context: Context) -> Message:
    model_type = _get_model_type(context)
    dataset = context.run_config.get("dataset", "DataSense")
    input_size = DATASETS_CONFIG[dataset]['input_size']
    model = DNN(input_size, 2) if model_type == "DNN" else CNN(input_size, 2)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition_type = context.run_config["partition-type"]
    seed = int(context.run_config.get("seed", 42))
    batch_size = int(context.run_config.get("batch-size", 64))
    alpha_value = float(context.run_config.get("alpha", 0.1)) if partition_type == "non-iid" else None
    trainloader = (
        load_partitioned_data(partition_id, num_partitions, seed=seed, dataset=dataset, batch_size=batch_size)[0]
        if partition_type == "iid"
        else load_partitioned_data_non_iid(partition_id, num_partitions, alpha=alpha_value, seed=seed, dataset=dataset)[0]
    )

    # Straggler simulation
    straggler_prob = context.run_config.get("straggler-probability", 0.0)
    max_penalty = context.run_config.get("straggler-max-penalty", 0.7)
    base_epochs = context.run_config["local-epochs"]
    if random.random() < straggler_prob:
        penalty = random.uniform(0.1, max_penalty)
        epochs = max(1, int(base_epochs * (1 - penalty)))
        print(f"[Client {partition_id}] Straggler: {epochs}/{base_epochs} epochs (penalty={penalty:.2f})")
    else:
        epochs = base_epochs

    train_loss = train_fn(
        model, trainloader, epochs, msg.content["config"]["lr"], device, seed=seed,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _train_xgboost(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    X_train, y_train, _, _ = _load_numpy_partition(context)
    
    params = get_xgb_params(context.run_config)
    
    # Calculate exact local rounds needed to match baseline's n_estimators
    total_trees = params.get('n_estimators', 500)
    num_server_rounds = context.run_config.get("num-server-rounds", 20)
    num_partitions = context.node_config["num-partitions"]
    fraction_train = context.run_config.get("fraction-train", 1.0)
    clients_per_round = max(1, int(num_partitions * fraction_train))
    
    num_local_round = max(1, total_trees // (clients_per_round * num_server_rounds))
    
    server_round = int(msg.content["config"].get("server-round", 1))

    # Deserialise global model bytes from ArrayRecord
    global_model_bytes = msg.content["arrays"]["0"].numpy().tobytes()

    local_model_raw = train_xgb(
        X_train, y_train, params, num_local_round,
        global_model_bytes=global_model_bytes,
        server_round=server_round,
    )

    model_np = np.frombuffer(local_model_raw, dtype=np.uint8)
    model_record = ArrayRecord([model_np])
    metrics = {"num-examples": len(X_train)}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    print(f"[Client {partition_id}] XGBoost trained {num_local_round} boosting rounds (round {server_round})")
    return Message(content=content, reply_to=msg)


def _train_random_forest(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, y_train, _, _ = _load_numpy_partition(context)

    params = get_rf_params(context.run_config)
    total_estimators = params['n_estimators']
    n_estimators_local = max(1, total_estimators // num_partitions)
    params['n_estimators'] = n_estimators_local

    model_bytes = train_rf(X_train, y_train, params)

    model_np = np.frombuffer(model_bytes, dtype=np.uint8)
    model_record = ArrayRecord([model_np])
    metrics = {"num-examples": len(X_train)}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    print(f"[Client {partition_id}] RandomForest trained {n_estimators_local} trees")
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model_type = _get_model_type(context)

    if model_type in ("XGBoost", "AttentionXGBoost"):
        return _evaluate_xgboost(msg, context)
    elif model_type == "RandomForest":
        return _evaluate_random_forest(msg, context)
    else:
        return _evaluate_dnn(msg, context)


def _evaluate_dnn(msg: Message, context: Context) -> Message:
    model_type = _get_model_type(context)
    dataset = context.run_config.get("dataset", "DataSense")
    input_size = DATASETS_CONFIG[dataset]['input_size']
    model = DNN(input_size, 2) if model_type == "DNN" else CNN(input_size, 2)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    partition_type = context.run_config["partition-type"]
    seed = int(context.run_config.get("seed", 42))
    batch_size = int(context.run_config.get("batch-size", 64))
    alpha_value = float(context.run_config.get("alpha", 0.1)) if partition_type == "non-iid" else None
    _, valloader = (
        load_partitioned_data(partition_id, num_partitions, seed=seed, dataset=dataset, batch_size=batch_size)
        if partition_type == "iid"
        else load_partitioned_data_non_iid(partition_id, num_partitions, alpha=alpha_value, seed=seed, dataset=dataset, batch_size=batch_size)
    )

    results_dict = test_fn(model, valloader, device)

    metrics = {
        "accuracy": results_dict["accuracy"],
        "detection_rate": results_dict["detection_rate"],
        "false_positive_rate": results_dict["false_positive_rate"],
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _evaluate_xgboost(msg: Message, context: Context) -> Message:
    X_train, y_train, X_test, y_test = _load_numpy_partition(context)
    params = get_xgb_params(context.run_config)
    global_model_bytes = msg.content["arrays"]["0"].numpy().tobytes()

    results_dict = evaluate_xgb(
        global_model_bytes,
        X_test,
        y_test,
        params,
        X_threshold=X_train,
        y_threshold=y_train,
    )

    metrics = {
        "accuracy": float(results_dict["accuracy"]),
        "detection_rate": float(results_dict["detection_rate"]),
        "false_positive_rate": float(results_dict["false_positive_rate"]),
        "num-examples": len(X_test),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

def _evaluate_random_forest(msg: Message, context: Context) -> Message:
    X_train, y_train, X_test, y_test = _load_numpy_partition(context)
    model_bytes = msg.content["arrays"]["0"].numpy().tobytes()

    results_dict = evaluate_rf(
        model_bytes,
        X_test,
        y_test,
        X_threshold=X_train,
        y_threshold=y_train,
    )

    metrics = {
        "accuracy": float(results_dict["accuracy"]),
        "detection_rate": float(results_dict["detection_rate"]),
        "false_positive_rate": float(results_dict["false_positive_rate"]),
        "num-examples": len(X_test),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)