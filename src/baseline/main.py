import torch
import numpy as np
import random
import pandas as pd
import pickle
import logging
import os
import argparse
from data_loader import load_data, load_data_edgeIIoT
from models.dnn import DNN, CNN, evaluate_model_DNN, train_model_DNN
from models.xgboost_model import train_model_xgboost, evaluate_model_xgboost
from models.random_forest import evaluate_model_RandomForest, train_model_RandomForest


def set_global_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")

DATASETS_CONFIG = {
        'DataSense': {
            'loader': load_data,
            'input_size': 779,
        },
        'Edge-IIoT': {
            'loader': load_data_edgeIIoT,
            'input_size': 40, 
        }
    }

def main(train: bool = False, 
         dataset: str = 'DataSense', 
         epochs: int =30, 
         learning_rate: float =0.001, 
         batch_size: int =64, 
         model_type: str ='dnn',
         force_reprocess: bool = False,
         use_cpu: bool = False,
         seed: int = 42):

    set_global_seed(seed)

    input_size = DATASETS_CONFIG[dataset]['input_size']
    num_classes = 2  # Binary classification: normal vs attack
    
    # Set device
    if use_cpu:
        device = torch.device('cpu')
        logging.info("Using CPU.")
    else:
        device = torch.device('cpu')  # Default to CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logging.info("Using CUDA.")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logging.info("Using MPS.")
    print(f"Using device: {device}")

    trainloader, testloader = DATASETS_CONFIG[dataset]['loader'](
        csv_path='datasets/' + dataset,
        batch_size=batch_size,
        force_reprocess=force_reprocess
    )

    # Create evaluation directory
    current_time = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    if model_type == 'xgboost' or model_type == 'random_forest':
        model = None
    elif model_type == 'dnn':
        model = DNN(input_size, num_classes)
        model.to(device)
    elif model_type == 'cnn':
        model = CNN(input_size, num_classes)
        model.to(device)
    
    if train:
        eval_dir = f'results/Baseline/{dataset}/{model_type}/seed_{seed}/{current_time}'
        os.makedirs(eval_dir, exist_ok=True)
        if model_type == 'xgboost':
            model_metadata = train_model_xgboost(trainloader, current_time=current_time, dataset=dataset, seed=seed)
        elif model_type == 'random_forest':
            model_metadata = train_model_RandomForest(trainloader, current_time=current_time, dataset=dataset, seed=seed)
        else:
            model_metadata = train_model_DNN(model, trainloader, device, epochs=epochs, learning_rate=learning_rate, current_time=current_time, dataset=dataset, seed=seed)
    else:
        eval_base = f'results/Baseline/{dataset}/{model_type}/seed_{seed}'
        timestamp_dirs = [
            d for d in os.listdir(eval_base) 
            if os.path.isdir(os.path.join(eval_base, d))
        ]
        current_time = sorted(timestamp_dirs)[-1]
        if model_type == 'xgboost':
            with open(f'results/Baseline/{dataset}/{model_type}/seed_{seed}/{current_time}/nids_model_xgboost.pkl', 'rb') as f:
                model_metadata = pickle.load(f)
            model = model_metadata['model']
        elif model_type == 'random_forest':
            with open(f'results/Baseline/{dataset}/{model_type}/seed_{seed}/{current_time}/nids_model_random_forest.pkl', 'rb') as f:
                model_metadata = pickle.load(f)
            model = model_metadata['model']
        else:
            checkpoint = torch.load(f'results/Baseline/{dataset}/{model_type}/seed_{seed}/{current_time}/nids_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            model_metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    
    print("Evaluating model...")
    if model_type == 'xgboost':
        evaluate_model_xgboost(model_metadata['model'], trainloader, testloader, model_metadata=model_metadata, current_time=current_time, dataset=dataset, seed=seed)
    elif model_type == 'random_forest':
        evaluate_model_RandomForest(model_metadata['model'], trainloader, testloader, model_metadata=model_metadata, current_time=current_time, dataset=dataset, seed=seed)
    else:
        evaluate_model_DNN(model, testloader, device, trainloader, model_metadata=model_metadata, current_time=current_time, dataset=dataset, seed=seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate NIDS model')
    
    parser.add_argument('--train', action='store_true',
                        help='Train a new model (default: load existing model)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training and testing (default: 64)')
    parser.add_argument('--dataset', type=str, default='DataSense',
                        choices=['DataSense', 'Edge-IIoT'],
                        help='Dataset to use (default: DataSense)')
    parser.add_argument('--model-type', type=str, default='dnn',
                        choices=['dnn', 'xgboost', 'random_forest', 'cnn'],
                        help='Type of model to use (default: dnn)')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Force reprocessing of the dataset')
    parser.add_argument('--cpu', action='store_true',
                        help='Force training on CPU even if GPU is available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    main(train=args.train, dataset=args.dataset, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size, model_type=args.model_type, force_reprocess=args.force_reprocess, use_cpu=args.cpu, seed=args.seed)