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
        # If you saved IDs, you can also load self.ids = data['ids'] here
        print ('Dataset shape - ', self.features.shape , self.labels.shape)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Add self.ids[idx] here if you added IDs to your caching script!
        return self.features[idx], self.labels[idx]

class AttentionProbe(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_heads=8):
        super().__init__()
        # 1. Learnable query token (Think of this as the "Detective")
        # Shape: [1, 1, 768]
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # 2. Multi-head Attention Layer
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        # 3. Final linear classifier
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x is your unpooled cached features. Shape: [Batch, 576, 768]
        B = x.size(0)
        
        # Expand our single query token to match the batch size
        # Shape becomes: [Batch, 1, 768]
        q = self.query.expand(B, -1, -1)
        
        # Cross-Attention: 
        # Query = q
        # Key/Value = x (The 576 spatial tokens from ORBIS)
        # attn_out is the pooled feature vector: [Batch, 1, 768]
        # attn_weights is the heatmap matrix: [Batch, 1, 576]
        attn_out, attn_weights = self.attn(query=q, key=x, value=x)
        
        # Squeeze out the sequence dimension: [Batch, 1, 768] -> [Batch, 768]
        pooled_features = attn_out.squeeze(1)
        
        # Pass the attended features into the final classifier
        logits = self.classifier(pooled_features)
        
        # (Optional) You can return attn_weights here during evaluation to generate heatmaps
        return logits

def train_linear_probe(hidden_dim=768, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Cached Data from Block 18
    train_dataset = CachedFeatureDataset("./cached_features/train_block18.pt")
    val_dataset = CachedFeatureDataset("./cached_features/val_block18.pt")

    print ('-------',len(train_dataset), len(val_dataset), '---------')
    
    # Drop last to avoid batch size mismatch issues in edge cases
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 2. Initialize Attention Probe instead of Linear Probe
    model = AttentionProbe(input_dim=hidden_dim, num_heads=8).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Use unpack operator (*_) in case your dataset returns IDs as a 3rd element
        for idx, (features, labels, *_) in enumerate(train_loader):
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
        
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for features, labels, *_ in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                
                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)[:, 1]
                
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                
        # 5. Calculate Metrics
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
        val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0) * 100
        val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0) * 100
        
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except ValueError:
            val_auc = float('nan')
            
        cm = confusion_matrix(all_val_labels, all_val_preds)
        
        # 6. Print Output
        print(f"\nEpoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"--> Val Acc: {val_acc:.2f}% | Precision: {val_precision:.2f}% | Recall: {val_recall:.2f}% | AUC: {val_auc:.4f}")
        print(f"--> Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    train_linear_probe(hidden_dim=768, epochs=50, lr=1e-5)