import sys
import random as py_random
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree._tree import Tree
import numpy as np
import xgboost as xgb
project_root = Path(__file__).resolve().parents[2]
baseline_models_dir = str(project_root / "src" / "baseline" / "models")
if baseline_models_dir not in sys.path:
    sys.path.insert(0, baseline_models_dir)
from dnn import DNN, CNN, evaluate_model_DNN


DATASETS_CONFIG = {
    'DataSense': {
        'input_size': 779,
        'processed_dir': f'{project_root}/datasets/DataSense/processed',
    },
    'Edge-IIoT': {
        'input_size': 40,
        'processed_dir': f'{project_root}/datasets/Edge-IIoT/processed',
    },
}


def set_global_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_partitioned_data(partition_id: int, num_partitions: int, seed: int = 42, dataset: str = 'DataSense', batch_size: int = 64):
    """Load IID-partitioned NIDS data.

    Only the training set is partitioned across clients.  Each client
    holds out 20 % of its partition for local validation.  The global
    test set is never mixed in, preventing data leakage.
    """

    processed_dir = DATASETS_CONFIG[dataset]['processed_dir']
    train_data = np.load(f'{processed_dir}/train_data.npz')

    X_full = train_data['X']
    y_full = train_data['y']

    # Partition training data only
    partition_size = len(X_full) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else len(X_full)

    X_partition = X_full[start_idx:end_idx]
    y_partition = y_full[start_idx:end_idx]

    # If a class has fewer than 2 samples, we cannot stratify on y
    # Fallback to unstratified splitting for this partition to avoid ValueError
    stratify_arg = y_partition if np.all(np.bincount(y_partition, minlength=2) >= 2) else None

    # Local train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_partition, y_partition, test_size=0.2, random_state=seed, stratify=stratify_arg
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)

    return trainloader, valloader

def load_partitioned_data_non_iid(partition_id: int, num_partitions: int, alpha: float = 0.5, seed: int = 42, dataset: str = 'DataSense', batch_size: int = 64):
    """Load non-IID partitioned NIDS data.

    Only the training set is partitioned via Dirichlet allocation.
    The global test set is never mixed in, preventing data leakage.
    A fixed random seed ensures reproducible partitions across clients.
    """

    processed_dir = DATASETS_CONFIG[dataset]['processed_dir']
    train_data = np.load(f'{processed_dir}/train_data.npz')

    X = train_data['X']
    y = train_data['y']
    y_granular = train_data.get('y_granular', y) # Fallback to binary if not available

    # Fixed seed for reproducible partitions across all clients
    rng = np.random.RandomState(seed)

    # Use granular classes for splitting to create true feature skew
    n_classes = len(np.unique(y_granular))
    indices_by_class = {k: np.where(y_granular == k)[0] for k in np.unique(y_granular)}

    min_size = 0
    min_require_size = 32

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_partitions)]

        for k in indices_by_class.keys():
            idx_k = indices_by_class[k].copy()
            rng.shuffle(idx_k)

            proportions = rng.dirichlet(np.repeat(alpha, num_partitions))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch_split = np.split(idx_k, proportions)
            for i in range(num_partitions):
                idx_batch[i].extend(idx_batch_split[i])

        min_size = min(len(d) for d in idx_batch)

    partition_indices = idx_batch[partition_id]
    X_partition = X[partition_indices]
    y_partition = y[partition_indices]

    # If a class has fewer than 2 samples, we cannot stratify on y
    # Fallback to unstratified splitting for this partition to avoid ValueError
    stratify_arg = y_partition if np.all(np.bincount(y_partition, minlength=2) >= 2) else None

    # Local train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_partition, y_partition, test_size=0.2, random_state=seed, stratify=stratify_arg
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)

    return trainloader, valloader


def train(net, trainloader, epochs, lr, device, seed: int = 42):
    """Train the model on the training set."""
    set_global_seed(seed)
    net.to(device)
    
    # Use weighted loss for imbalanced datasets
    weights = torch.tensor([1.0, 1.5], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    net.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    
    avg_trainloss = running_loss / (len(trainloader) * epochs)
    return avg_trainloss

test = evaluate_model_DNN

def load_server_data(dataset: str = 'DataSense'):
    """Load the full NIDS dataset for server-side evaluation."""
    processed_dir = DATASETS_CONFIG[dataset]['processed_dir']
    train_file = f'{processed_dir}/train_data.npz'
    test_file = f'{processed_dir}/test_data.npz'
    
    # Load preprocessed data
    train_data = np.load(train_file)
    test_data = np.load(test_file)
    
    # Extract X and y from the loaded files
    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=64)
    
    return trainloader, testloader


def load_partitioned_data_numpy(partition_id: int, num_partitions: int, seed: int = 42, dataset: str = 'DataSense'):
    """Load IID-partitioned data as numpy arrays (for tree-based models).

    Only the training set is partitioned.  The global test set is not
    mixed in, preventing data leakage.
    """
    processed_dir = DATASETS_CONFIG[dataset]['processed_dir']
    train_data = np.load(f'{processed_dir}/train_data.npz')
    X = train_data['X']
    y = train_data['y']

    partition_size = len(X) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else len(X)

    X_partition = X[start_idx:end_idx]
    y_partition = y[start_idx:end_idx]

    stratify_arg = y_partition if np.all(np.bincount(y_partition, minlength=2) >= 2) else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_partition, y_partition, test_size=0.2, random_state=seed, stratify=stratify_arg
    )
    return X_train, y_train, X_val, y_val


def load_partitioned_data_non_iid_numpy(partition_id: int, num_partitions: int, alpha: float = 0.5, seed: int = 42, dataset: str = 'DataSense'):
    """Load non-IID partitioned data as numpy arrays (for tree-based models).

    Only the training set is partitioned via Dirichlet allocation.
    A fixed random seed ensures reproducible partitions across clients.
    """
    processed_dir = DATASETS_CONFIG[dataset]['processed_dir']
    train_data = np.load(f'{processed_dir}/train_data.npz')
    X = train_data['X']
    y = train_data['y']
    y_granular = train_data.get('y_granular', y) # Fallback to binary if not available

    rng = np.random.RandomState(seed)

    # Use granular classes for splitting
    n_classes = len(np.unique(y_granular))
    indices_by_class = {k: np.where(y_granular == k)[0] for k in np.unique(y_granular)}

    min_size = 0
    min_require_size = 32

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_partitions)]
        for k in indices_by_class.keys():
            idx_k = indices_by_class[k].copy()
            rng.shuffle(idx_k)
            proportions = rng.dirichlet(np.repeat(alpha, num_partitions))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch_split = np.split(idx_k, proportions)
            for i in range(num_partitions):
                idx_batch[i].extend(idx_batch_split[i])
        min_size = min(len(d) for d in idx_batch)

    partition_indices = idx_batch[partition_id]
    X_partition = X[partition_indices]
    y_partition = y[partition_indices]

    stratify_arg = y_partition if np.all(np.bincount(y_partition, minlength=2) >= 2) else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_partition, y_partition, test_size=0.2, random_state=seed, stratify=stratify_arg
    )
    return X_train, y_train, X_val, y_val


def load_server_data_numpy(dataset: str = 'DataSense'):
    """Load full train/test data as numpy arrays for server-side evaluation."""
    processed_dir = DATASETS_CONFIG[dataset]['processed_dir']
    train_data = np.load(f'{processed_dir}/train_data.npz')
    test_data = np.load(f'{processed_dir}/test_data.npz')
    return train_data['X'], train_data['y'], test_data['X'], test_data['y']


def _find_optimal_threshold(y_probs, y_true, target_fpr: float = 0.01):
    """Find threshold maximizing F1 with FPR <= target_fpr.

    This mirrors the logic in nids_baseline/XGBoost.py so that federated
    and baseline evaluations use the same threshold-selection criterion.
    """
    best_threshold = 0.5  # fallback
    best_recall = 0.0
    best_f1 = 0.0
    best_fpr = 1.0
    found_feasible = False

    for threshold in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_probs > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if fpr <= target_fpr:
            if (not found_feasible) or (f1 > best_f1) or (
                np.isclose(f1, best_f1) and recall > best_recall
            ):
                found_feasible = True
                best_f1 = f1
                best_threshold = threshold
                best_recall = recall
                best_fpr = fpr

    if not found_feasible:
        fallback_threshold = 0.5
        fallback_fpr = 1.0
        fallback_f1 = -1.0

        for threshold in np.arange(0.01, 1.00, 0.01):
            y_pred = (y_probs > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if (fpr < fallback_fpr) or (np.isclose(fpr, fallback_fpr) and f1 > fallback_f1):
                fallback_threshold = threshold
                fallback_fpr = fpr
                fallback_f1 = f1

        best_threshold = fallback_threshold
        best_fpr = fallback_fpr
        best_f1 = fallback_f1

    return best_threshold


def get_xgb_params(run_config: dict) -> dict:
    """Build XGBoost param dict from Flower run config, matching baseline hyperparameters."""
    dataset = run_config.get('dataset', 'DataSense')
    seed = int(run_config.get('seed', 42))
    if dataset == "DataSense":
        hyperparameters = {'max_depth': 9, 'learning_rate': 0.20211525651495557, 'n_estimators': 900, 'min_child_weight': 1, 'gamma': 0.0011375210935630209, 'subsample': 0.8669667098532958, 'colsample_bytree': 0.9319591927218307, 'max_delta_step': 8}
    else:
        hyperparameters = {'max_depth': 3, 'learning_rate': 0.017462960628932028, 'n_estimators': 850, 'min_child_weight': 2, 'gamma': 0.6008466563824473, 'subsample': 0.8874331341725578, 'colsample_bytree': 0.7409548092577293, 'max_delta_step': 2}
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'base_score': 0.5,
        'seed': seed,
        'n_jobs': -1,
        **hyperparameters
    }


def train_xgb(X_train, y_train, params: dict, num_local_round: int,
              global_model_bytes: bytes = b"", server_round: int = 1):
    """Train XGBoost locally and return raw model bytes.

    On round 1, trains from scratch. On subsequent rounds, loads the global
    model and performs additional boosting rounds (bagging pattern).
    """
    # Calculate scale_pos_weight from local data
    num_benign = np.sum(y_train == 0)
    num_attack = np.sum(y_train == 1)
    if num_attack > 0:
        params['scale_pos_weight'] = num_benign / num_attack

    # Strip sklearn-only params that the native XGBoost API doesn't recognise
    xgb_params = {k: v for k, v in params.items() if k not in ('n_estimators',)}

    train_dmatrix = xgb.DMatrix(X_train, label=y_train)

    if server_round == 1:
        bst = xgb.train(xgb_params, train_dmatrix, num_boost_round=num_local_round)
    else:
        bst = xgb.Booster(params=xgb_params)
        bst.load_model(bytearray(global_model_bytes))
        # Local boosting: add more trees
        for _ in range(num_local_round):
            bst.update(train_dmatrix, bst.num_boosted_rounds())
        # Bagging: extract only the last N trees for server aggregation
        bst = bst[
            bst.num_boosted_rounds() - num_local_round : bst.num_boosted_rounds()
        ]

    local_model = bst.save_raw("json")
    return local_model


def evaluate_xgb(model_bytes: bytes, X_test, y_test, params: dict, X_threshold=None, y_threshold=None) -> dict:
    """Evaluate an XGBoost model using optimal threshold selection.

    Uses the same threshold-tuning approach as the baseline
    (best F1 with FPR <= 1 %). If X_threshold/y_threshold are provided,
    threshold tuning is done on that split and evaluation is done on X_test/y_test.
    """
    # Strip sklearn-only params that the native XGBoost API doesn't recognise
    xgb_params = {k: v for k, v in params.items() if k not in ('n_estimators',)}
    bst = xgb.Booster(params=xgb_params)
    bst.load_model(bytearray(model_bytes))
    test_dmatrix = xgb.DMatrix(X_test, label=y_test)

    y_probs = bst.predict(test_dmatrix)
    if X_threshold is not None and y_threshold is not None:
        threshold_dmatrix = xgb.DMatrix(X_threshold, label=y_threshold)
        y_threshold_probs = bst.predict(threshold_dmatrix)
        threshold = _find_optimal_threshold(y_threshold_probs, y_threshold, target_fpr=0.01)
    else:
        threshold = _find_optimal_threshold(y_probs, y_test, target_fpr=0.01)

    y_pred = (y_probs > threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'accuracy': accuracy,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'threshold': threshold,
    }

def get_rf_params(run_config: dict) -> dict:
    """Build RF hyperparameter dict from Flower run config, matching baseline hyperparameters."""
    dataset = run_config.get('dataset', 'DataSense')
    seed = int(run_config.get('seed', 42))
    if dataset == "DataSense":
        hyperparameters = {'n_estimators': 400, 'max_depth': 30, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 'sqrt'}
    else:
        hyperparameters = {'n_estimators': 450, 'max_depth': 30, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 'log2'}
    return {
        'random_state': seed,
        'class_weight': 'balanced',
        'n_jobs': -1,
        **hyperparameters
    }


def train_rf(X_train, y_train, params: dict) -> bytes:
    """Train a local Random Forest and return pickled model bytes."""
    rf_params = dict(params)
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    return pickle.dumps(model)


def evaluate_rf(model_bytes: bytes, X_test, y_test, X_threshold=None, y_threshold=None) -> dict:
    """Evaluate a Random Forest model using optimal threshold selection.

    Uses predict_proba + threshold tuning (best F1 with FPR <= 1 %)
    to match the baseline evaluation methodology.
    """
    model = pickle.loads(model_bytes)
    y_probs = model.predict_proba(X_test)[:, 1]
    if X_threshold is not None and y_threshold is not None:
        y_threshold_probs = model.predict_proba(X_threshold)[:, 1]
        threshold = _find_optimal_threshold(y_threshold_probs, y_threshold, target_fpr=0.01)
    else:
        threshold = _find_optimal_threshold(y_probs, y_test, target_fpr=0.01)

    y_pred = (y_probs > threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'accuracy': accuracy,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'threshold': threshold,
    }


def _fix_single_class_tree(tree_estimator):
    """Patch a single-class DecisionTreeClassifier to be 2-class compatible.

    Constructs a new Cython Tree object with max_n_classes=2 and loads the
    modified state into it.  The value array is expanded from
    (n_nodes, 1, 1) to (n_nodes, 1, 2), placing sample counts in the
    column matching the original class label.
    """
    original_class = int(tree_estimator.classes_[0])

    old_tree = tree_estimator.tree_
    state = old_tree.__getstate__().copy()

    old_values = state['values']
    new_values = np.zeros((old_values.shape[0], 1, 2), dtype=old_values.dtype)
    new_values[:, :, original_class] = old_values[:, :, 0]
    state['values'] = new_values

    new_tree = Tree(old_tree.n_features, np.array([2], dtype=np.intp), old_tree.n_outputs)
    new_tree.__setstate__(state)
    tree_estimator.tree_ = new_tree

    tree_estimator.n_classes_ = 2
    tree_estimator.classes_ = np.array([0, 1])
    tree_estimator.n_outputs_ = 1


def combine_rf_models(model_bytes_list: list) -> bytes:
    """Combine multiple RF models by merging all estimators into one forest.

    Handles non-IID partitions where some clients may have seen only one
    class.  Trees trained on a single class have n_classes_=1 and their
    predict_proba returns shape (n, 1), which is incompatible with
    the (n, 2) output expected by the combined forest.  We reconstruct
    such trees to be 2-class compatible via the Cython Tree object's
    __getstate__/__setstate__ mechanism.
    """
    all_estimators = []
    base_model = None
    patched = 0

    for mb in model_bytes_list:
        rf = pickle.loads(mb)
        for tree in rf.estimators_:
            if hasattr(tree, 'n_classes_') and tree.n_classes_ != 2:
                _fix_single_class_tree(tree)
                patched += 1
            all_estimators.append(tree)
        # Pick a model that already knows about both classes as template
        if base_model is None or (hasattr(rf, 'n_classes_') and rf.n_classes_ == 2):
            base_model = rf

    if patched > 0:
        print(f"[RF Combine] Patched {patched} single-class trees for 2-class compatibility")

    # Ensure the base model has 2 classes
    base_model.classes_ = np.array([0, 1])
    base_model.n_classes_ = 2

    # Build combined forest using the base model as a template
    base_model.estimators_ = all_estimators
    base_model.n_estimators = len(all_estimators)
    return pickle.dumps(base_model)