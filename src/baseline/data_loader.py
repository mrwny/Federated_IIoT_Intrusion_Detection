import ast
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import HashingVectorizer
import os

def load_data_edgeIIoT(csv_path: str, batch_size: int, test_split: float = 0.2, force_reprocess: bool = False):
    """Load Edge-IIoT dataset here and return DataLoader objects for training and testing."""
    print("Loading Edge-IIoT data...")

    processed_dir = os.path.join(csv_path, 'processed')
    train_file = os.path.join(processed_dir, 'train_data.npz')
    test_file = os.path.join(processed_dir, 'test_data.npz')

    if not force_reprocess and os.path.exists(train_file) and os.path.exists(test_file):
        print("Loading preprocessed Edge-IIoT data from cache...")
        train_data = np.load(train_file)
        test_data = np.load(test_file)

        X_train = train_data['X']
        y_train = train_data['y']
        X_test = test_data['X']
        y_test = test_data['y']

        print(f"Loaded {len(X_train)} train samples, {len(X_test)} test samples")
    else:
        print("Processing raw Edge-IIoT data...")
        dfs = []
        for file in os.listdir(csv_path):
            if file.endswith('.csv'):
                file_path = os.path.join(csv_path, file)
                # dtype=str prevents the C-engine from crashing during type inference of mixed columns
                dfs.append(pd.read_csv(file_path, dtype=str, low_memory=False))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No CSV files found in Edge-IIoT datasets directory")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Converts hex strings to integers
                try:
                    df[col] = df[col].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else x)
                except (ValueError, TypeError, AttributeError):
                    pass  # Column contains non-hex strings, skip conversion
        
        # Extract granular attack types for federated learning (Feature Skew)
        y_granular_raw = df.get('Attack_type', pd.Series('Normal', index=df.index)).fillna('Normal')
        y_granular = pd.factorize(y_granular_raw)[0].astype(np.int64)

        # Clean and preprocess data
        cols_to_drop = [
            'Attack_label',
            'Attack_type',
            'frame.time',
            'ip.src_host',
            'ip.dst_host',
            'arp.src.proto_ipv4',
            'arp.dst.proto_ipv4',
            'http.file_data',
            'http.request.full_uri',
            'icmp.transmit_timestamp',
            'http.request.uri.query',
            'tcp.options',
            'tcp.payload',
            'tcp.srcport',
            'http.request.method',
            'http.referer',
            'http.request.version',
            'tcp.dstport',
            'udp.port',
            'dns.qry.name.len',
            'mqtt.msg',
            'mqtt.topic',
            'mqtt.protoname'
        ]

        X = df.drop(columns=cols_to_drop)
        y = df['Attack_label']
        
        # Attack_label is already '0' or '1' as strings.
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(np.int64)
            
        print(f"Label distribution - Benign: {(y == 0).sum()}, Attack: {(y == 1).sum()}")
            
        # Force conversion back to numeric right before array casting
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        X = X.fillna(0)
        
        # Save exact feature names before converting to numpy
        final_feature_names = X.columns.tolist()

        X = X.values.astype(np.float32)
        y = y.values.astype(np.int64)

        X_train, X_test, y_train, y_test, y_gran_train, y_gran_test = train_test_split(
            X, y, y_granular, test_size=test_split, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save processed data as .npz for federated loaders
        os.makedirs(processed_dir, exist_ok=True)
        np.savez(train_file, X=X_train, y=y_train, y_granular=y_gran_train)
        np.savez(test_file, X=X_test, y=y_test, y_granular=y_gran_test)
        
        with open(os.path.join(processed_dir, 'feature_names.json'), 'w') as f:
            json.dump(final_feature_names, f)
            
        print(f"Saved processed Edge-IIoT data to {processed_dir}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def load_data(csv_path: str, batch_size: int, test_split: float = 0.2, force_reprocess: bool = False):
    """Load DataSense dataset here and return DataLoader objects for training and testing."""
    print("Loading data...")

    processed_dir = os.path.join(csv_path, 'processed')
    train_file = os.path.join(processed_dir, 'train_data.npz')
    test_file = os.path.join(processed_dir, 'test_data.npz')
    
    if not force_reprocess and os.path.exists(train_file) and os.path.exists(test_file):
        print("Loading preprocessed data from cache...")
        train_data = np.load(train_file)
        test_data = np.load(test_file)
        
        X_train = train_data['X']
        y_train = train_data['y']
        X_test = test_data['X']
        y_test = test_data['y']
        
        print(f"Loaded {len(X_train)} train samples, {len(X_test)} test samples")
    else:
        print("Processing raw data...")
        attack_dfs = []
        for file in os.listdir(csv_path + '/attack_data/'):
            if file.endswith('.csv'):
                file_path = os.path.join(csv_path + '/attack_data/', file)
                df = pd.read_csv(file_path)
                attack_dfs.append(df)
        
        benign_dfs = []
        for file in os.listdir(csv_path + '/benign_data/'):
            if file.endswith('.csv'):
                file_path = os.path.join(csv_path + '/benign_data/', file)
                df = pd.read_csv(file_path)
                benign_dfs.append(df)
        
        # Combine all dataframes
        attack_df = pd.DataFrame()
        benign_df = pd.DataFrame()
        if attack_dfs:
            attack_df = pd.concat(attack_dfs, ignore_index=True)
        if benign_dfs:
            benign_df = pd.concat(benign_dfs, ignore_index=True)
        if attack_dfs and benign_dfs:
            df = pd.concat([attack_df, benign_df], ignore_index=True)
        elif attack_dfs:
            df = attack_df
        elif benign_dfs:
            df = benign_df
        else:
            raise ValueError("No data found in the specified directories.")
        
        print(f"Total samples: {len(df)}")

        if 'timestamp_start' in df.columns and 'timestamp_end' in df.columns:
            try:
                df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], errors='coerce')
                df['timestamp_end'] = pd.to_datetime(df['timestamp_end'], errors='coerce')
                df['duration'] = (df['timestamp_end'] - df['timestamp_start']).dt.total_seconds()
            except Exception as e:
                print(f"Error converting timestamps: {e}")

        # Extract granular labels before separating X and y
        y_granular_raw = df.get('label2', pd.Series('benign', index=df.index)).fillna('benign')
        y_granular = pd.factorize(y_granular_raw)[0].astype(np.int64)

        cols_to_keep = [
            'log_messages_count',
            'log_data-types',
            'network_fragmented-packets',
            'network_ip-flags_max',
            'network_tcp-flags-psh_count',
            'network_ips_all_count',
            'network_ips_dst',
            'network_macs_src',
            'network_packets_all_count',
            'network_ports_all',
            'network_time-delta_avg',
            'network_ttl_avg',
        ]

        X = df[cols_to_keep].copy()
        y = df['label1']

        def parse_list_feature(val):
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    return []
            return val

        mlb_cols = ['log_data-types']
        hashing_cols = ['network_ips_dst', 'network_ports_all', 'network_macs_src']

        mlb = MultiLabelBinarizer()
        for col in mlb_cols:
            X[col] = X[col].apply(parse_list_feature) 
            expanded_cols = mlb.fit_transform(X[col])
            expanded_df = pd.DataFrame(
                expanded_cols, 
                columns=[f"{col}_{cls}" for cls in mlb.classes_],
                index=X.index 
            )
            X = pd.concat([X.drop(columns=[col]), expanded_df], axis=1)
        num_features = 256
        hasher = HashingVectorizer(n_features=num_features, alternate_sign=False) 

        for col in hashing_cols:
            X[col] = X[col].apply(parse_list_feature)
            
            # Convert list to space-separated string for HashingVectorizer
            # e.g. ['192.168.1.1', '10.0.0.5'] -> "192.168.1.1 10.0.0.5"
            col_as_string = X[col].apply(lambda x: ' '.join(str(i) for i in x))
            
            hashed_cols = hasher.fit_transform(col_as_string).toarray()
            
            # FIX: Pass index=X.index here too
            hashed_df = pd.DataFrame(
                hashed_cols,
                columns=[f"{col}_hash_{i}" for i in range(num_features)],
                index=X.index
            )
            X = pd.concat([X.drop(columns=[col]), hashed_df], axis=1)
            

        if y.dtype == 'object': # Convert string labels to binary
            y = y.apply(lambda x: 0 if 'benign' in str(x).lower() else (1 if 'attack' in str(x).lower() else None))
            
            if y.isnull().any():
                print(f"Warning: Unknown label values found: {df['label1'][y.isnull()].unique()}")
                print("Filling unknown labels with 0 (benign)")
                y = y.fillna(0)
            
            print(f"Label distribution - Benign: {(y == 0).sum()}, Attack: {(y == 1).sum()}")
        
        X = X.fillna(0)
        
        # Save exact feature names before converting to numpy
        final_feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.values.astype(np.float32)
        y = y.values.astype(np.int64)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test, y_gran_train, y_gran_test = train_test_split(
            X, y, y_granular, test_size=test_split, random_state=42, stratify=y
        )
        
        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Save processed data as csv
        os.makedirs(processed_dir, exist_ok=True)

        # Save processed data as npz
        np.savez(train_file, X=X_train, y=y_train, y_granular=y_gran_train)
        np.savez(test_file, X=X_test, y=y_test, y_granular=y_gran_test)
        
        with open(os.path.join(processed_dir, 'feature_names.json'), 'w') as f:
            json.dump(final_feature_names, f)
        
        
        print(f"Saved processed data to {processed_dir}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def get_feature_names(dataset_name: str):
    """Retrieve feature names from the processed dataset."""
    if dataset_name not in ['DataSense', 'Edge-IIoT']:
        raise ValueError("Unknown dataset name provided.")
    if dataset_name == 'DataSense':
        feature_file = 'datasets/DataSense/processed/feature_names.json'
    else:
        feature_file = 'datasets/Edge-IIoT/processed/feature_names.json'
    with open(feature_file, 'r') as f:
        feature_names = json.load(f)
    return feature_names