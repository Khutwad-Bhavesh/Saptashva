import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
import numpy as np

# Adjust imports from project structure
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.goes_ingestion import parse_goes_timeseries
from src.features.engineering import extract_features, prepare_lstm_sequences
from src.features.labeling import apply_physics_labels
from src.stage0.lstm_model import Stage0LSTM

class FocalLoss(nn.Module):
    """
    Focal Loss designed for highly imbalanced datasets.
    It down-weights easy examples and forces the model to focus on hard, rare examples (Eruptive flares).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) # Prevents nans when probability 0
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def main():
    print("=== SAPTASHVA Phase 7: The 95% Accuracy Push (Focal Optimization) ===")
    
    # 1. Fetch full 11-year Solar Cycle 24 GOES data (2010-2021)
    data_dir = "datasets/goes_11_year"
    df = parse_goes_timeseries(data_dir=data_dir)
    
    # Align column names with engineering.py expectations
    df = df.rename(columns={'soft_flux': 'soft_xray_flux', 'hard_flux': 'hard_xray_flux'})
    
    # 2. Extract standard model features
    df = extract_features(df)
    
    # 3. Apply advanced mathematical peak detection labels
    df = apply_physics_labels(df, flux_col='soft_xray_flux')
    
    print("Class distribution after physics labeling:")
    print(df['state'].value_counts())
    
    # Define features
    feature_cols = ['soft_xray_flux', 'hard_xray_flux', 'flux_ratio', 'soft_flux_deriv', 'soft_flux_roll_std']
    
    # 4. Scale features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save scaler for future inference
    os.makedirs('models', exist_ok=True)
    dump(scaler, 'models/goes_scaler.joblib')
    
    # 5. Prepare sequences for LSTM (V1 champion lookback=60)
    X, y = prepare_lstm_sequences(df, feature_cols, target_col='state', lookback=60)
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
    
    # 6. CAPACITY BUMP: Upgrade to (128, 64) architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage0LSTM(input_size=len(feature_cols), hidden_size1=128, hidden_size2=64, num_classes=4).to(device)
    
    # Compute Class Weights for Focal Loss Alpha
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    full_weights = np.ones(4)
    for c, w in zip(classes, weights):
        full_weights[c] = w
    class_weights = torch.tensor(full_weights, dtype=torch.float).to(device)
    print(f"Computed Focal Alpha Weights: {class_weights.cpu().numpy()}")
    
    # FOCAL LOSS REVERT: Cross Entropy proved more stable for this specific time-series distribution
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # STABLE OPTIMIZATION REVERT
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # Lower, smoother learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print("\nStarting Phase 7 Extended Capacity training with Early Stopping...")
    epochs = 50 # Let it train longer
    
    # Early Stopping tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 10 # More patience to allow plateau reductions to work
    epochs_no_improve = 0
    
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total if total > 0 else 0
            
            # Step the scheduler
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.6f} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early Stopping Logic (tracking val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                
                # MID-TRAINING CHECKPOINT SAVE
                checkpoint_path = 'models/stage0_goes_checkpoint.pth'
                torch.save(best_model_weights, checkpoint_path)
                
                print(f"   => New best model! (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}) -> Saved to disk!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"   => Early stopping triggered after {epoch+1} epochs.")
                    model.load_state_dict(best_model_weights)
                    break
    except KeyboardInterrupt:
        print("\n[!] Training manually interrupted by user. Salvaging best checkpoint...")
                
    if epochs_no_improve < patience:
        print("   => Training finished normally.")
        
    print("\nLoading best weights from disk checkpoint...")
    try:
        model.load_state_dict(torch.load('models/stage0_goes_checkpoint.pth'))
    except FileNotFoundError:
        model.load_state_dict(best_model_weights)
        
    # 7. Save the expanded foundation model
    torch.save(model.state_dict(), 'models/stage0_goes.pth')
    print(f"\n=== FINAL RESULT: Best Val Acc: {best_val_acc:.4f} | Best Val Loss: {best_val_loss:.4f} ===")
    
    if best_val_acc > 0.95:
        print("\n🏆 THE 95% BARRIER HAS BEEN OFFICIALLY BROKEN! 🏆")

if __name__ == "__main__":
    main()
