from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import get_feature_names
from xgboost_model import find_optimal_threshold_from_arrays, _apply_latex_style

def train_model_RandomForest(
    train_loader: DataLoader,
    n_estimators: int = 150,
    current_time: str = None,
    dataset: str = None,
    seed: int = 42
) -> dict:
    if dataset == "DataSense":
        hyperparameters = {'n_estimators': 400, 'max_depth': 30, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 'sqrt'}
    else:
        hyperparameters = {'n_estimators': 450, 'max_depth': 30, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 'log2'}
    if 'n_estimators' not in hyperparameters:
        hyperparameters['n_estimators'] = n_estimators
    X_train = []
    y_train = []

    for data, labels in train_loader:
        X_train.append(data.numpy())
        y_train.append(labels.numpy())

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        stratify=y_train
    )

    model = RandomForestClassifier(random_state=seed, class_weight='balanced', **hyperparameters)

    model.fit(X_train_fit, y_train_fit)

    y_val_probs = model.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold_from_arrays(
        y_probs=y_val_probs,
        y_true=y_val,
        current_time=current_time,
        dataset=dataset,
        model_type='random_forest',
        seed=seed,
        target_fpr=0.01,
        context_label='validation'
    )

    model_metadata = {
        'optimizer': 'N/A',
        'n_estimators': model.get_params().get('n_estimators'),
        'max_depth': model.get_params().get('max_depth'),
        'min_samples_split': model.get_params().get('min_samples_split'),
        'min_samples_leaf': model.get_params().get('min_samples_leaf'),
        'max_features': model.get_params().get('max_features'),
        'epochs': 'N/A',
        'model': model,
        'scale_pos_weight': 'balanced',
        'seed': seed,
        'threshold': threshold,
        'hyperparameters': hyperparameters,
        'threshold_policy': 'maximize_f1_with_fpr_constraint',
        'threshold_tuned_on': 'validation_split',
        'validation_split': 0.2,
        'max_fpr': 0.01
    }

    with open(f'results/Baseline/{dataset}/random_forest/seed_{seed}/{current_time}/nids_model_random_forest.pkl', 'wb') as f:
        pickle.dump(model_metadata, f)

    return model_metadata

def evaluate_model_RandomForest(
    model: RandomForestClassifier,
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

    output_file = f'results/Baseline/{dataset}/random_forest/seed_{seed}/{current_time}/nids_evaluation.txt'
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*60}\n")

        # Model metadata
        f.write(f"---------- Model ----------\n")
        f.write(f"Model Type: Random Forest Classifier\n")
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
    
    analyze_feature_importance_rf(model, dataset_name=dataset, seed=seed, current_time=current_time)

def analyze_feature_importance_rf(model, dataset_name='DataSense', save_plot=True, model_type='random_forest', seed=42, current_time=None):
    print("Analyzing feature importance for Random Forest...")

    try:
        if dataset_name == 'DataSense':
            cols = get_feature_names(dataset_name='DataSense')
        elif dataset_name == 'Edge-IIoT':
            cols = get_feature_names(dataset_name='Edge-IIoT')
        else:
            raise ValueError("Unknown dataset name provided.")
    except Exception as e:
        print(f"Error loading feature names: {e}")
        return None
    
    importances = model.feature_importances_
    
    mapped_importance = {}
    for idx, value in enumerate(importances):
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
            
    df_imp = pd.DataFrame(list(mapped_importance.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print(f"\n{'Rank':<5} | {'Feature Name':<30} | {'Importance':<15}")
    print("-" * 55)
    for index, row in df_imp.head(20).iterrows():
        print(f"{index+1:<5} | {row['Feature']:<30} | {row['Importance']:.4f}")
        
    if save_plot:
        _apply_latex_style()
        plt.figure(figsize=(12, 8))
        top_n = df_imp.head(20).sort_values(by='Importance', ascending=True)
        
        plt.barh(top_n['Feature'], top_n['Importance'], color='#2ca02c') # Use green for RF
        plt.xlabel('Log Gain (Contribution to Accuracy)')
        plt.title(f'Random Forest Feature Importance Analysis ({dataset_name})')
        plt.xscale('log')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        if current_time is not None:
            output_file = f'results/Baseline/{dataset_name}/{model_type}/seed_{seed}/{current_time}/feature_importance.pdf'
        else:
            output_file = f'results/Baseline/{dataset_name}/{model_type}/feature_importance.pdf'
            
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        print(f"\nSaved feature plot to {output_file}")
        
    return df_imp