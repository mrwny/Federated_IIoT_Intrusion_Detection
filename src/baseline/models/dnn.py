import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch import device
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class DNN(nn.Module): # simple feedforward neural network
    def __init__(self, input_size: int, num_classes: int):
        super(DNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1) # Helps reduce overfitting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        
        x = self.features(x)
        x = self.classifier(x)
        return x

def evaluate_model_DNN(
    net: nn.Module,
    testloader: DataLoader,
    device: device,
    trainloader: DataLoader = None,
    model_metadata: dict = None,
    current_time: str = None,
    dataset: str = None,
    seed: int = 42
) -> dict:
    net.eval()
    all_predictions = []
    all_labels = []

    model_type = 'dnn' if isinstance(net, DNN) else 'cnn'
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = sum([1 for p, l in zip(all_predictions, all_labels) if p == l]) / len(all_labels)
    output_file = f'results/Baseline/{dataset}/{model_type}/seed_{seed}/{current_time}/nids_evaluation.txt'
    
    # Prepare all output content first
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Detection rates
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    if trainloader:
        # Build the complete output string
        output_content = []
        output_content.append(f"\n{'='*60}\n")
        output_content.append(f"---------- Model ----------\n")
        output_content.append(f"Model Architecture: {str(net)}\n")
        output_content.append(f"Input Size: {trainloader.dataset.tensors[0].shape[1]}\n")
        
        if model_metadata:
            output_content.append(f"Loss Function: CrossEntropyLoss\n")
            output_content.append(f"Loss Weights (Benign, Attack): {model_metadata.get('loss_weights', 'N/A')}\n")
            output_content.append(f"Optimizer: {model_metadata.get('optimizer', 'N/A')}\n")
            output_content.append(f"Learning Rate: {model_metadata.get('learning_rate', 'N/A')}\n")
            output_content.append(f"Training Epochs: {model_metadata.get('epochs', 'N/A')}\n")
        
        output_content.append(f"Training Date: {pd.Timestamp.now()}\n")
        output_content.append(f"Training Samples: {len(trainloader.dataset)}\n")
        output_content.append(f"---------------------------\n\n")
        
        output_content.append(f"MODEL EVALUATION RESULTS\n")
        output_content.append(f"{'='*60}\n")
        output_content.append(f"Overall Accuracy: {accuracy:.10f} ({accuracy*100:.10f}%)\n\n")
        
        output_content.append("Classification Report:\n")
        output_content.append(classification_report(all_labels, all_predictions, 
                                    target_names=['Benign', 'Attack'],
                                    digits=10))
        output_content.append("\n")
        
        output_content.append("Confusion Matrix:\n")
        output_content.append(f"                Predicted\n")
        output_content.append(f"              Benign  Attack\n")
        output_content.append(f"Actual Benign  {cm[0][0]:6d}  {cm[0][1]:6d}\n")
        output_content.append(f"       Attack  {cm[1][0]:6d}  {cm[1][1]:6d}\n\n")
        
        output_content.append(f"True Negatives (Benign correctly identified):  {tn}\n")
        output_content.append(f"False Positives (Benign misclassified):        {fp}\n")
        output_content.append(f"False Negatives (Attack missed):               {fn}\n")
        output_content.append(f"True Positives (Attack correctly detected):    {tp}\n\n")
        
        output_content.append(f"Attack Detection Rate: {detection_rate:.10f} ({detection_rate*100:.10f}%)\n")
        output_content.append(f"False Positive Rate:   {false_positive_rate:.10f} ({false_positive_rate*100:.10f}%)\n")
        output_content.append(f"{'='*60}\n\n")
        
        # Write everything at once
        with open(output_file, 'a') as f:
            f.write(''.join(output_content))

    results = {
        "accuracy": accuracy,
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

    return results

def train_model_DNN(
    net: nn.Module,
    trainloader: DataLoader,
    device: device,
    epochs: int = 30,
    learning_rate: float = 0.001,
    current_time: str = None,
    dataset: str = None,
    seed: int = 42
):
    weights = torch.tensor([1, 1.5], dtype=torch.float32).to(device)
    """Train the DNN/CNN model."""

    model_type = 'dnn' if isinstance(net, DNN) else 'cnn'
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    total_batches = len(trainloader)

    net.to(device)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        for i, batch in enumerate(trainloader):
            # Extract inputs and labels from the batch
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_batches}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    metadata = {
        'model_state_dict': net.state_dict(),
        'loss_weights': weights.cpu().numpy().tolist(),
        'learning_rate': learning_rate,
        'epochs': epochs,
        'optimizer': 'Adam',
        'seed': seed
    }
    torch.save(metadata, f'results/Baseline/{dataset}/{model_type}/seed_{seed}/{current_time}/nids_model.pth')
    return metadata