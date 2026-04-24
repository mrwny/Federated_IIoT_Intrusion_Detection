# COMP0029 - Federated Intrusion Detection in IIoT

BSc Computer Science Dissertation - Candidate Number: WBDW1

## Overview

This repository contains the accompanying source code, experiment configurations, and results for the Federated Intrusion Detection in IIoT project. Included are implementations of centralised baseline models, federated learning strategies using Flower and the simulation code providing IID and Non-IID data partitioning and client staggling simulation.

## Repository Structure

```
├── src/                        Source code
│   ├── baseline/               Centralised baseline models
│   │   ├── main.py             Entry point for baseline experiments
│   │   ├── data_loader.py      Dataset loading & preprocessing
│   │   ├── models/             DNN, Random Forest, XGBoost implementations
│   │   └── tuning/             Hyperparameter tuning
│   │
│   ├── federated/                              Flower-based federated learning
│   │   ├── task.py                             Model definitions & data partitioning
│   │   ├── client_app.py                       Flower ClientApp (all strategies)
|   |   ├── AttentionWeightedFedXgbBagging.py   Novel attention-weighted FedXgbBagging strategy 
│   │   ├── server_app.py                       Flower ServerApp (strategy configs)
│   │   ├── pyproject.toml                      Flower project configuration
│   │   ├── run_client_sweep.sh                 Client scalability sweep
│   │   └── run_mu_sweep.sh                     FedProx µ hyperparameter sweep
│   │
│   └── analysis/                           Data distribution analysis
│       ├── visualise_partitions.py   IID vs Non-IID partition simulation
│       └── analyse_fed_results.py
│
├── eval/                       Aggregation & visualisation pipeline
│   ├── aggregate_baseline_results.py
│   ├── aggregate_federated_results.py
│   ├── visualise_baseline.py   Baseline comparison figures
│   └── visualise_results.py    Master figure generation
│
├── results/                    Experiment outputs
│   ├── baseline/               Per-model metrics (DataSense, Edge-IIoT)
│   ├── federated/              Per-strategy metrics + µ sweep
│   └── figures/                PDF figures
│
└── requirements.txt            Python dependencies
```

## Datasets

The dissertation uses two IIoT intrusion detection datasets to compare performance of the centralised and federated learning approaches on different data distributions.

- **[Edge-IIoT](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)**
- **[DataSense](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html)**

> Datasets are not included in this repository due to size. See the respective publications for download instructions.

Once downloaded, you can place the datasets in a root-level directory called `datasets/{dataset_name}`. The scripts will automatically pick up the datasets from this directory.

#### Note on Preprocessing

Before running any federated experiments, you must first preprocess the raw datasets. This is done automatically when running any centralised baseline experiment. For example:

```bash
python src/baseline/main.py --train --model-type dnn --dataset DataSense --seed 42
python src/baseline/main.py --train --model-type dnn --dataset Edge-IIoT --seed 42
```

This will generate the preprocessed `.npz` files in `datasets/{dataset_name}/processed/` that the federated scripts depend on.



## Setup

Before running any experiments, install the required python packages.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Hyperparameter Tuning

To tune the hyperparameters of the tree-based baseline models, you can use the following script:

```bash
bash src/baseline/run_tuning_all.sh
```

The results can be used to change the hyperparameters used in the baseline experiments.



### Running Baseline Experiments

```bash
bash src/baseline/run_baseline_all.sh
```
This script will run all the baseline models on the 3 seeds utilised in the dissertation and save the results in the results/baseline directory.

### Running Federated Experiments

```bash
cd src/federated
flwr run .
```
This will run the Flower simulation based on the parameters specified in the pyproject.toml file.

#### pyproject.toml structure
You may want to change the configuration of the simulation by manually editing the pyproject.toml file. The following is an example of a configuration file which simulates a non-IID DNN FedAvg experiment with 20 clients and 20 communication rounds, including straggler simulation:

```toml
[tool.flwr.app.config]
num-server-rounds = 20
fraction-evaluate = 0.5
fraction-train = 1
min-train-nodes = 2
lr = 0.001
local-epochs = 5
batch-size = 32
partition-type = "non-iid"
model-type = "DNN"
strategy = "FedAvg"
proximal-mu = 1.0
straggler-probability = 0.3
straggler-max-penalty = 0.7
seed = 1702
dataset = "DataSense"
```

Or you can directly use the sweep scripts for scalability and hyperparameter experiments, which runs the exact configurations the experiments were run on:

```bash
bash run_client_sweep.sh    # Client count sweep (2, 4, 8, 16, 20)
bash run_mu_sweep.sh        # FedProx µ sweep (0.01 – 1.0)
```


## Results Processing & Figure Generation
To process the results and generate the figures, you can use the following scripts:

```bash
python eval/aggregate_baseline_results.py
python eval/aggregate_federated_results.py
python eval/visualise_baseline.py --save
python eval/visualise_results.py --save
```



## License

The code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
