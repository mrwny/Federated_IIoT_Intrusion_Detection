import xgboost as xgb
import pickle
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data_loader import get_feature_names


def _apply_latex_style():
    """Set matplotlib params for LaTeX-matching dissertation figures."""
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def train_model_xgboost(
    trainloader: DataLoader,
    epochs: int = 500,
    learning_rate: float = 0.1,
    current_time: str = None,
    dataset: str = None,
    seed: int = 42
):
    if dataset == "DataSense":
        hyperparameters = {'max_depth': 9, 'learning_rate': 0.20211525651495557, 'n_estimators': 900, 'min_child_weight': 1, 'gamma': 0.0011375210935630209, 'subsample': 0.8669667098532958, 'colsample_bytree': 0.9319591927218307, 'max_delta_step': 8}
    else:
        hyperparameters = {'max_depth': 3, 'learning_rate': 0.017462960628932028, 'n_estimators': 850, 'min_child_weight': 2, 'gamma': 0.6008466563824473, 'subsample': 0.8874331341725578, 'colsample_bytree': 0.7409548092577293, 'max_delta_step': 2}
    X_train = []
    y_train = []
    for inputs, labels in trainloader:
        X_train.append(inputs.numpy())
        y_train.append(labels.numpy())
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    # Hold out a validation split for threshold tuning to avoid test-set leakage
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train
    )

    # Calculate precise class weight
    num_benign = np.sum(y_train_fit == 0)
    num_attack = np.sum(y_train_fit == 1)
    scale_weight = num_benign / num_attack
    print(f"Class Balance: Benign={num_benign}, Attack={num_attack}")
    print(f"Applying scale_pos_weight: {scale_weight:.4f}")

    model = xgb.XGBClassifier(
        **hyperparameters,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=seed
    )

    model.fit(X_train_fit, y_train_fit)

    y_val_probs = model.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold_from_arrays(
        y_probs=y_val_probs,
        y_true=y_val,
        current_time=current_time,
        dataset=dataset,
        model_type='xgboost',
        seed=seed,
        target_fpr=0.01,
        context_label='validation'
    )

    metadata = {
        'model': model,
        'learning_rate': model.get_params().get('learning_rate'),
        'n_estimators': model.get_params().get('n_estimators'),
        'max_depth': model.get_params().get('max_depth'),
        'min_child_weight': model.get_params().get('min_child_weight'),
        'gamma': model.get_params().get('gamma'),
        'subsample': model.get_params().get('subsample'),
        'colsample_bytree': model.get_params().get('colsample_bytree'),
        'max_delta_step': model.get_params().get('max_delta_step'),
        'hyperparameters': hyperparameters,
        'epochs': epochs,
        'optimizer': 'XGBoost',
        'scale_pos_weight': scale_weight,
        'seed': seed,
        'threshold': threshold,
        'threshold_policy': 'maximize_f1_with_fpr_constraint',
        'threshold_tuned_on': 'validation_split',
        'validation_split': 0.2,
        'max_fpr': 0.01
    }
    with open(f'results/Baseline/{dataset}/xgboost/seed_{seed}/{current_time}/nids_model_xgboost.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    return metadata

def evaluate_model_xgboost(
    model: xgb.XGBClassifier,
    trainloader: DataLoader,
    testloader: DataLoader,
    model_metadata: dict = None,
    current_time: str = None,
    dataset: str = None,
    seed: int = 42
) -> None:
    X_test = []
    y_test = []
    for inputs, labels in testloader:
        X_test.append(inputs.numpy())
        y_test.append(labels.numpy())
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    y_probs = model.predict_proba(X_test)[:, 1]

    threshold = 0.5
    if model_metadata and 'threshold' in model_metadata:
        threshold = model_metadata['threshold']
    else:
        print("Warning: No tuned threshold found in metadata. Falling back to 0.5")

    y_pred = (y_probs > threshold).astype(int)

    accuracy = sum(y_pred == y_test) / len(y_test)

    output_file = f'results/Baseline/{dataset}/xgboost/seed_{seed}/{current_time}/nids_evaluation.txt'
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*60}\n")

        # Model metadata
        f.write(f"---------- Model ----------\n")
        f.write(f"Model Type: XGBoost Classifier\n")
        f.write(f"Input Size: {trainloader.dataset.tensors[0].shape[1]}\n")
        if model_metadata:
            f.write(f"Optimizer: {model_metadata.get('optimizer', 'N/A')}\n")
            f.write(f"Learning Rate: {model_metadata.get('learning_rate', 'N/A')}\n")
            f.write(f"Training Epochs: {model_metadata.get('epochs', 'N/A')}\n")
            f.write(f"Decision Threshold: {model_metadata.get('threshold', threshold):.4f}\n")
            f.write(f"Threshold Tuned On: {model_metadata.get('threshold_tuned_on', 'N/A')}\n")
            f.write(f"Threshold Policy: {model_metadata.get('threshold_policy', 'N/A')}\n")
            f.write(f"Max FPR Constraint: {model_metadata.get('max_fpr', 'N/A')}\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n")
        f.write(f"Training Samples: {len(trainloader.dataset)}\n")
        f.write(f"---------------------------\n\n")

        f.write(f"MODEL EVALUATION RESULTS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Overall Accuracy: {accuracy:.10f} ({accuracy*100:.10f}%)\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, 
                                    target_names=['Benign', 'Attack'],
                                    digits=10))
        f.write("\n")
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        f.write("Confusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"              Benign  Attack\n")
        f.write(f"Actual Benign  {cm[0][0]:6d}  {cm[0][1]:6d}\n")
        f.write(f"       Attack  {cm[1][0]:6d}  {cm[1][1]:6d}\n\n")

        tn, fp, fn, tp = cm.ravel()
        
        f.write(f"True Negatives (Benign correctly identified):  {tn}\n")
        f.write(f"False Positives (Benign misclassified):        {fp}\n")
        f.write(f"False Negatives (Attack missed):               {fn}\n")
        f.write(f"True Positives (Attack correctly detected):    {tp}\n\n")
        
        # Detection rates
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        f.write(f"Attack Detection Rate: {detection_rate:.10f} ({detection_rate*100:.10f}%)\n")
        f.write(f"False Positive Rate:   {false_positive_rate:.10f} ({false_positive_rate*100:.10f}%)\n")
        f.write(f"{'='*60}\n\n")
    print(f"Evaluation results written to {output_file}")
    analyze_feature_importance(model, dataset_name=dataset, seed=seed, current_time=current_time)

def find_optimal_threshold_from_arrays(
    y_probs,
    y_true,
    current_time,
    dataset='DataSense',
    model_type='xgboost',
    seed=42,
    target_fpr=0.01,
    context_label='validation'
):
    precisions, recalls, _ = precision_recall_curve(y_true, y_probs)
    
    _apply_latex_style()
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, label=f'{model_type} Model', color='blue', linewidth=2)
    
    # Add "No Skill" baseline (ratio of attacks in dataset)
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill (Random Guess)', color='gray')

    print(
        f"\nThreshold tuning on {context_label} split with constraint: FPR <= {target_fpr*100:.2f}%"
    )
    print(f"{'Threshold':<10} | {'Detection Rate (Recall)':<25} | {'False Pos Rate':<20} | {'F1 Score':<10} | {'Missed Attacks':<15}")
    print("-" * 80)
    
    best_threshold = 0.5
    best_recall = 0.0
    best_precision = 0.0
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
        
        print(f"{threshold:.2f}       | {recall:.4f} ({recall*100:.2f}%)        | {fpr:.4f} ({fpr*100:.2f}%)   | {f1:.4f}       | {fn}")
        
        if fpr <= target_fpr:
            if (not found_feasible) or (f1 > best_f1) or (
                np.isclose(f1, best_f1) and recall > best_recall
            ):
                found_feasible = True
                best_f1 = f1
                best_threshold = threshold
                best_recall = recall
                best_precision = precision
                best_fpr = fpr

    if not found_feasible:
        # Fallback: choose the threshold with lowest FPR, then highest F1
        print("No threshold satisfied FPR constraint; selecting lowest-FPR threshold as fallback.")
        fallback_threshold = 0.5
        fallback_fpr = 1.0
        fallback_f1 = -1.0
        fallback_recall = 0.0
        fallback_precision = 0.0
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
                fallback_recall = recall
                fallback_precision = precision

        best_threshold = fallback_threshold
        best_fpr = fallback_fpr
        best_f1 = fallback_f1
        best_recall = fallback_recall
        best_precision = fallback_precision

    plt.scatter(best_recall, best_precision, marker='o', color='red', s=150, zorder=5, 
                label=(
                    f'Selected Threshold: {best_threshold:.2f}\n'
                    f'(Recall: {best_recall:.2f}, Prec: {best_precision:.2f}, FPR: {best_fpr:.3f})'
                ))
    
    plt.xlabel('Recall (Detection Rate)', fontsize=12)
    plt.ylabel('Precision (Alert Reliability)', fontsize=12)
    plt.title(f'Precision-Recall Curve - {dataset}', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    save_path = f'results/Baseline/{dataset}/{model_type}/seed_{seed}/{current_time}/threshold_analysis.pdf'
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Additional zoomed-in PR view around the selected operating point
    plt.figure(figsize=(10, 7))
    plt.plot(recalls, precisions, label=f'{model_type} Model', color='blue', linewidth=2)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill (Random Guess)', color='gray')
    plt.scatter(best_recall, best_precision, marker='o', color='red', s=150, zorder=5,
                label=(
                    f'Selected Threshold: {best_threshold:.2f}\n'
                    f'(Recall: {best_recall:.2f}, Prec: {best_precision:.2f}, FPR: {best_fpr:.3f})'
                ))

    x_low = max(0.0, best_recall - 0.12)
    x_high = min(1.0, best_recall + 0.08)
    y_low = max(0.0, best_precision - 0.12)
    y_high = min(1.02, best_precision + 0.05)
    if x_high - x_low < 0.12:
        x_low = max(0.0, x_high - 0.12)
    if y_high - y_low < 0.12:
        y_low = max(0.0, y_high - 0.12)

    plt.xlim(x_low, x_high)
    plt.ylim(y_low, y_high)
    plt.xlabel('Recall (Detection Rate)', fontsize=12)
    plt.ylabel('Precision (Alert Reliability)', fontsize=12)
    plt.title(f'Precision-Recall Curve (Zoomed) - {dataset}', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    zoom_save_path = f'results/Baseline/{dataset}/{model_type}/seed_{seed}/{current_time}/threshold_analysis_zoom.pdf'
    plt.savefig(zoom_save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nSaved Precision-Recall curve to {save_path}")
    print(f"Saved zoomed Precision-Recall curve to {zoom_save_path}")
    print("-" * 80)
    print(f"Recommended Threshold under FPR <= {target_fpr*100:.2f}%: {best_threshold:.2f}")
    
    return best_threshold


def find_optimal_threshold(model, testloader, current_time, dataset='DataSense', model_type='xgboost', seed=42, target_fpr=0.01):
    """Backward-compatible wrapper for callers that still pass model + dataloader."""
    X_test = []
    y_test = []
    for inputs, labels in testloader:
        X_test.append(inputs.numpy())
        y_test.append(labels.numpy())
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)

    y_probs = model.predict_proba(X_test)[:, 1]
    return find_optimal_threshold_from_arrays(
        y_probs=y_probs,
        y_true=y_test,
        current_time=current_time,
        dataset=dataset,
        model_type=model_type,
        seed=seed,
        target_fpr=target_fpr,
        context_label='provided_loader'
    )


def analyze_feature_importance(model, dataset_name='DataSense', save_plot=True, model_type='xgboost', seed=42, current_time=None):
    print("Analyzing feature importance...")

    try:
        if dataset_name == 'DataSense':
            csv_path = 'datasets/DataSense/attack_data/'
            files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
            if files:
                csv_path += files[0]
            else:
                raise FileNotFoundError("No CSV files found in the attack_data directory.")
        elif dataset_name == 'Edge-IIoT':
            csv_path = 'datasets/Edge-IIoT/'
            files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
            if files:
                csv_path += files[0]
            else:
                raise FileNotFoundError("No CSV files found in the Edge-IIoT dataset directory.")
        else:
            raise ValueError("Unknown dataset name provided.")
        df_header = pd.read_csv(csv_path, nrows=0)
        feature_names = df_header.columns.tolist()
        if dataset_name == 'DataSense':
            cols = get_feature_names(dataset_name='DataSense')
        elif dataset_name == 'Edge-IIoT':
            cols = get_feature_names(dataset_name='Edge-IIoT')
    except Exception as e:
        print(f"Error loading feature names: {e}")
        return None
    

    importance = model.get_booster().get_score(importance_type='gain')
    
    mapped_importance = {}
    for key, value in importance.items():
        idx = int(key.replace('f', ''))
        if idx < len(cols):
            name = cols[idx]
            # Aggregate hashed buckets back to their conceptual parent feature
            if '_hash_' in name:
                base_name = name.split('_hash_')[0]
                mapped_importance[base_name] = mapped_importance.get(base_name, 0.0) + value
            elif '_class_' in name or (name.split('_')[-1].isdigit() and len(name.split('_')) > 2): 
                # Catch MLB/PCA splits if any exist structurally
                base_name = '_'.join(name.split('_')[:-1])
                mapped_importance[base_name] = mapped_importance.get(base_name, 0.0) + value
            else:
                mapped_importance[name] = mapped_importance.get(name, 0.0) + value
            
    df_imp = pd.DataFrame(list(mapped_importance.items()), columns=['Feature', 'Gain'])
    df_imp = df_imp.sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    print(f"\n{'Rank':<5} | {'Feature Name':<30} | {'Gain Score':<15}")
    print("-" * 55)
    for index, row in df_imp.head(20).iterrows():
        print(f"{index+1:<5} | {row['Feature']:<30} | {row['Gain']:.4f}")
        
    if save_plot:
        _apply_latex_style()
        plt.figure(figsize=(12, 8))
        top_n = df_imp.head(20).sort_values(by='Gain', ascending=True)
        
        plt.barh(top_n['Feature'], top_n['Gain'], color='#1f77b4')
        plt.xlabel('Log Gain (Contribution to Accuracy)')
        plt.title(f'XGBoost Feature Importance Analysis ({dataset_name})')
        plt.xscale('log')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        if current_time is not None:
            output_file = f'results/Baseline/{dataset_name}/{model_type}/seed_{seed}/{current_time}/feature_importance.pdf'
        else:
            output_file = f'results/Baseline/{dataset_name}/{model_type}/feature_importance.pdf'
            
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        print(f"\nSaved feature plot to {output_file}")
        
    return df_imp