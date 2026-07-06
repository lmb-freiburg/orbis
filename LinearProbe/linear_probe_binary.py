import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path)
        self.features = data['features']
        self.labels = data['labels'].long() # Ensure labels are integers for CrossEntropy
        print ('Dataset shape - ', self.features.shape , self.labels.shape)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        # A single linear layer mapping STDiT hidden dimension to class logits
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

def train_linear_probe(hidden_dim=1024, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Cached Data from Block 20
    train_dataset = CachedFeatureDataset("./cached_features/train_block20.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block20.pt")
    # 1. Load Cached Data from Block 10
    train_dataset = CachedFeatureDataset("./cached_features/train_block10.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block10.pt")

    print ('-------',len(train_dataset), len(val_dataset), '---------')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 2. Initialize Probe
    model = LinearProbe(input_dim=hidden_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # 4. Validation Loop
        model.eval()
        
        # Lists to store batch outputs for sklearn metrics
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                
                # Get predicted classes
                _, predicted = outputs.max(1)
                
                # Get probabilities for the positive class (class 1) for AUC
                probs = F.softmax(outputs, dim=1)[:, 1]
                
                # Store them on CPU as numpy arrays
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                
        # 5. Calculate Metrics via Scikit-Learn
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
        val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0) * 100
        val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0) * 100
        
        # AUC requires both classes to be present in the validation set, catching potential errors
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError:
            val_auc = float('nan')
            
        cm = confusion_matrix(all_val_labels, all_val_preds)
        
        # 6. Print the formatted output
        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"--> Val Acc: {val_acc:.2f}% | Precision: {val_precision:.2f}% | Recall: {val_recall:.2f}% | AUC: {val_auc:.4f}")
        print(f"--> Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    train_linear_probe(hidden_dim=768, epochs=100, lr=1e-3)