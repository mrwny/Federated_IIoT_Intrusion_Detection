import argparse
import os
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATASETS_CONFIG = {
    'DataSense': {
        'processed_dir': f'{project_root}/datasets/DataSense/processed',
        'input_size': 779,
    },
    'Edge-IIoT': {
        'processed_dir': f'{project_root}/datasets/Edge-IIoT/processed',
        'input_size': 40,
    }
}


def estimate_model_size_bytes(model):
    """Estimate serialized model size in bytes."""
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def format_size(num_bytes):
    """Human-readable size formatter."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes / (1024 ** 2):.2f} MB"

def load_dataset_as_numpy(dataset_name='DataSense', subset=None):
    """Loads dataset directly from .npz to avoid PyTorch imports that cause segfaults on Mac."""
    processed_dir = DATASETS_CONFIG[dataset_name]['processed_dir']
    train_file = f"{processed_dir}/train_data.npz"
    print(f"Loading data directly from {train_file}...")
    
    data = np.load(train_file)
    X_train = data['X']
    y_train = data['y']
    
    if subset is not None:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, 
            train_size=subset, 
            stratify=y_train, 
            random_state=42
        )
        
    return X_train, y_train

def compute_per_fold_metrics(y_true, y_pred, y_proba):
    """
    Compute comprehensive metrics for a single fold.
    y_proba: probability of positive class (shape: (n,))
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics = {
        'precision_attack': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall_attack': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_attack': f1_score(y_true, y_pred, pos_label=1),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'roc_auc': roc_auc_score(y_true, y_proba),
    }
    
    # Compute PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    metrics['pr_auc'] = auc(recall, precision)
    
    return metrics

def objective(trial, model_type, X, y, scale_weight, seed):
    """
    Optuna objective function for evaluating hyperparameter trials.
    Computes comprehensive metrics including attack-class F1, precision, recall,
    specificity, ROC-AUC, and PR-AUC.
    """
    if model_type == 'xgboost':
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': scale_weight,
            'n_jobs': 1,
            'random_state': seed,
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10) 
        }
        model = xgb.XGBClassifier(**params)
        
    elif model_type == 'random_forest':
        params = {
            'class_weight': 'balanced',
            'n_jobs': 1,
            'random_state': seed,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 10, 30, step=10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        }
        model = RandomForestClassifier(**params)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    
    # Compute metrics across all folds
    fold_metrics = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        fold_metrics.append(compute_per_fold_metrics(y_val, y_pred, y_proba))
    
    # Average metrics across folds
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        values = [fm[key] for fm in fold_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)
    
    # Estimate model size
    model_size_bytes = estimate_model_size_bytes(model)
    
    # Store all metrics as trial attributes
    for key, value in avg_metrics.items():
        trial.set_user_attr(key, value)
    trial.set_user_attr('model_size_bytes', model_size_bytes)
    
    # Optimize for attack-class F1 (more important than macro for NIDS)
    return avg_metrics['f1_attack']

def generate_trial_visualizations(study, out_dir, model_type):
    """Generate visualization plots for Optuna trials."""
    # Extract trial data
    trials_data = []
    for trial in study.trials:
        trial_dict = {
            'trial_number': trial.number,
            'f1_macro': trial.user_attrs.get('f1_macro', None),
            'f1_attack': trial.user_attrs.get('f1_attack', None),
            'precision_attack': trial.user_attrs.get('precision_attack', None),
            'recall_attack': trial.user_attrs.get('recall_attack', None),
            'specificity': trial.user_attrs.get('specificity', None),
            'roc_auc': trial.user_attrs.get('roc_auc', None),
            'pr_auc': trial.user_attrs.get('pr_auc', None),
            'model_size_bytes': trial.user_attrs.get('model_size_bytes', None),
        }
        trials_data.append(trial_dict)
    
    df = pd.DataFrame(trials_data)
    
    # Plot 1: F1-Attack over trials
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['trial_number'], df['f1_attack'], marker='o', linestyle='-', alpha=0.7, label='F1-Attack')
    ax.plot(df['trial_number'], df['f1_macro'], marker='s', linestyle='--', alpha=0.7, label='F1-Macro')
    ax.axhline(y=study.best_value, color='r', linestyle='--', linewidth=2, label=f'Best F1-Attack: {study.best_value:.4f}')
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'{model_type.upper()} Optimization Progress', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'trial_f1_progress.png'), dpi=300)
    plt.close()
    
    # Plot 2: Model Size vs F1-Attack (Pareto frontier)
    fig, ax = plt.subplots(figsize=(10, 8))
    sizes_mb = df['model_size_bytes'] / (1024 ** 2)
    scatter = ax.scatter(sizes_mb, df['f1_attack'], c=df['trial_number'], cmap='viridis', s=100, alpha=0.7)
    ax.scatter([study.best_trial.user_attrs['model_size_bytes'] / (1024**2)], 
               [study.best_value], 
               color='red', s=300, marker='*', label='Best Trial', zorder=5)
    ax.set_xlabel('Model Size (MB)', fontsize=12)
    ax.set_ylabel('F1-Attack Score', fontsize=12)
    ax.set_title(f'{model_type.upper()} F1-Attack vs Model Size', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Trial Number')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'f1_vs_model_size.png'), dpi=300)
    plt.close()
    
    # Plot 3: Attack metrics (precision, recall, specificity)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['trial_number'], df['precision_attack'], marker='o', label='Precision (Attack)', alpha=0.8)
    ax.plot(df['trial_number'], df['recall_attack'], marker='s', label='Recall (Attack)', alpha=0.8)
    ax.plot(df['trial_number'], df['specificity'], marker='^', label='Specificity (Benign)', alpha=0.8)
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_type.upper()} Per-Class Metrics Progress', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_metrics.png'), dpi=300)
    plt.close()
    
    print(f"Saved visualizations to {out_dir}")

def run_bayesian_tuning(model_type='xgboost', dataset='DataSense', seed=42, subset=None, n_trials=50):
    print(f"Loading data for {dataset}...")
    X_train, y_train = load_dataset_as_numpy(dataset, subset=subset)
    
    num_benign = np.sum(y_train == 0)
    num_attack = np.sum(y_train == 1)
    scale_weight = num_benign / max(num_attack, 1)
    print(f"Class Balance: Benign={num_benign}, Attack={num_attack} (scale_weight={scale_weight:.2f})")
    
    out_dir = f"results/Baseline/{dataset}/{model_type}/optuna_tuning"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting Optuna Bayesian Optimization for {model_type}...")
    start_time = time.time()
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    
    study.optimize(
        lambda trial: objective(trial, model_type, X_train, y_train, scale_weight, seed), 
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=4
    )
    
    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.2f} seconds.")
    print(f"Best trial F1-Attack score: {study.best_value:.4f}")
    print(f"Best parameters found: {study.best_params}")
    
    # Extract best model metrics
    best_model_size_bytes = study.best_trial.user_attrs.get('model_size_bytes', 'N/A')
    if isinstance(best_model_size_bytes, (int, float)):
        best_model_size_human = format_size(best_model_size_bytes)
        print(f"Best model size: {best_model_size_human} ({best_model_size_bytes} bytes)")
    else:
        best_model_size_human = 'N/A'
    
    best_f1_macro = study.best_trial.user_attrs.get('f1_macro', 'N/A')
    best_precision_attack = study.best_trial.user_attrs.get('precision_attack', 'N/A')
    best_recall_attack = study.best_trial.user_attrs.get('recall_attack', 'N/A')
    best_specificity = study.best_trial.user_attrs.get('specificity', 'N/A')
    best_roc_auc = study.best_trial.user_attrs.get('roc_auc', 'N/A')
    best_pr_auc = study.best_trial.user_attrs.get('pr_auc', 'N/A')
    
    print(f"  F1-Macro: {best_f1_macro}")
    if isinstance(best_precision_attack, (int, float)):
        print(f"  Attack Class - Precision: {best_precision_attack:.10f}, Recall: {best_recall_attack:.10f}")
    if isinstance(best_specificity, (int, float)):
        print(f"  Specificity (Benign): {best_specificity:.10f}")
    if isinstance(best_roc_auc, (int, float)):
        print(f"  ROC-AUC: {best_roc_auc:.10f}, PR-AUC: {best_pr_auc:.10f}")

    # Save results
    best_params_file = os.path.join(out_dir, 'optuna_best_parameters.txt')
    with open(best_params_file, 'w') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Best Parameters: {study.best_params}\n")
        f.write(f"\n=== PRIMARY METRIC (Optimization Objective) ===\n")
        f.write(f"Best CV F1-Attack: {study.best_value:.4f}\n")
        f.write(f"\n=== COMPREHENSIVE METRICS ===\n")
        f.write(f"F1-Macro: {best_f1_macro if isinstance(best_f1_macro, (int, float)) else 'N/A'}\n")
        f.write(f"Precision (Attack): {best_precision_attack if isinstance(best_precision_attack, (int, float)) else 'N/A'}\n")
        f.write(f"Recall (Attack): {best_recall_attack if isinstance(best_recall_attack, (int, float)) else 'N/A'}\n")
        f.write(f"Specificity (Benign): {best_specificity if isinstance(best_specificity, (int, float)) else 'N/A'}\n")
        f.write(f"ROC-AUC: {best_roc_auc if isinstance(best_roc_auc, (int, float)) else 'N/A'}\n")
        f.write(f"PR-AUC: {best_pr_auc if isinstance(best_pr_auc, (int, float)) else 'N/A'}\n")
        f.write(f"\n=== MODEL SIZE ===\n")
        if best_model_size_human != 'N/A':
            f.write(f"Best Model Size: {best_model_size_human} ({best_model_size_bytes} bytes)\n")
        f.write(f"\n=== TIMING ===\n")
        f.write(f"Tuning duration: {elapsed:.2f}s for {n_trials} trials\n")
        
    # Save full trial history
    df_trials = study.trials_dataframe()
    df_trials.to_csv(os.path.join(out_dir, 'optuna_trials_history.csv'), index=False)
    print(f"Saved full Optuna trial history to {out_dir}/optuna_trials_history.csv")
    
    # Generate visualizations
    generate_trial_visualizations(study, out_dir, model_type)
    
    # Save detailed trial data with comprehensive metrics
    detailed_trials = []
    for trial in study.trials:
        trial_info = {
            'trial_number': trial.number,
            'state': trial.state.name,
            'model_size_bytes': trial.user_attrs.get('model_size_bytes', None),
            'model_size_mb': trial.user_attrs.get('model_size_bytes', None) / (1024**2) if trial.user_attrs.get('model_size_bytes') else None,
            'f1_macro': trial.user_attrs.get('f1_macro', None),
            'f1_macro_std': trial.user_attrs.get('f1_macro_std', None),
            'f1_attack': trial.user_attrs.get('f1_attack', None),
            'f1_attack_std': trial.user_attrs.get('f1_attack_std', None),
            'precision_attack': trial.user_attrs.get('precision_attack', None),
            'precision_attack_std': trial.user_attrs.get('precision_attack_std', None),
            'recall_attack': trial.user_attrs.get('recall_attack', None),
            'recall_attack_std': trial.user_attrs.get('recall_attack_std', None),
            'specificity': trial.user_attrs.get('specificity', None),
            'specificity_std': trial.user_attrs.get('specificity_std', None),
            'roc_auc': trial.user_attrs.get('roc_auc', None),
            'roc_auc_std': trial.user_attrs.get('roc_auc_std', None),
            'pr_auc': trial.user_attrs.get('pr_auc', None),
            'pr_auc_std': trial.user_attrs.get('pr_auc_std', None),
        }
        trial_info.update(trial.params)
        detailed_trials.append(trial_info)
    
    df_detailed = pd.DataFrame(detailed_trials)
    df_detailed.to_csv(os.path.join(out_dir, 'trials_with_metrics.csv'), index=False)
    print(f"Saved comprehensive trial metrics to {out_dir}/trials_with_metrics.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bayesian Tuning for NIDS tree models')
    parser.add_argument('--model', type=str, choices=['xgboost', 'random_forest'], required=True)
    parser.add_argument('--dataset', type=str, choices=['DataSense', 'Edge-IIoT'], default='DataSense')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subset', type=float, default=None)
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials to run')
    args = parser.parse_args()
    
    run_bayesian_tuning(model_type=args.model, dataset=args.dataset, seed=args.seed, subset=args.subset, n_trials=args.trials)
