import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path)
        self.features = data['features']
        self.labels = data['labels'].long() # Ensure labels are integers for CrossEntropy

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
    
    # 1. Load Cached Data
    train_dataset = CachedFeatureDataset("./cached_features/train_block20.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block20.pt")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 2. Initialize Probe
    # Make sure 'hidden_dim' matches your STDiT configuration (e.g., DiT-L = 1024, DiT-XL = 1152)
    model = LinearProbe(input_dim=hidden_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
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
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

if __name__ == "__main__":
    # Update hidden_dim based on your STDiT dimensions (e.g., 768, 1024, 1152)
    train_linear_probe(hidden_dim=1024, epochs=30, lr=1e-3)