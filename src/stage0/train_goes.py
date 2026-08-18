import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump
import numpy as np

# Adjust imports from project structure
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_pipeline.goes_ingestion import parse_goes_timeseries
from src.features.engineering import extract_features, prepare_lstm_sequences
from src.features.labeling import apply_physics_labels
from src.stage0.lstm_model import Stage0LSTM

def main():
    print("=== SAPTASHVA Phase 2: GOES Foundation Model Training (V3 >90% Opt) ===")
    
    # 1. Fetch pristine 2003 Halloween Storms data via SunPy
    df = parse_goes_timeseries()
    
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
    
    # 5. Prepare sequences for LSTM (Lookback extended to 120)
    X, y = prepare_lstm_sequences(df, feature_cols, target_col='state', lookback=120)
    
    # Split Train/Val (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=64, shuffle=False)
    
    # 6. Initialize Optimized LSTM Model (128 -> 64) with Aggressive Dropout
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage0LSTM(input_size=len(feature_cols), hidden_size1=128, hidden_size2=64, num_classes=4).to(device)
    
    # Dynamic Class Weights calculation
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    full_weights = np.ones(4)
    for c, w in zip(classes, weights):
        full_weights[c] = w
    class_weights = torch.tensor(full_weights, dtype=torch.float).to(device)
    print(f"Computed Class Weights: {class_weights.cpu().numpy()}")
    
    # Use Weighted Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Adaptive Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print("\nStarting foundation model training with EARLY STOPPING...")
    epochs = 30
    
    # Early Stopping tracking
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience = 5
    epochs_no_improve = 0
    
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
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"   => New best model found! (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"   => Early stopping triggered after {epoch+1} epochs. Loading best weights.")
                model.load_state_dict(best_model_weights)
                break
                
    if epochs_no_improve < patience:
        model.load_state_dict(best_model_weights)
        print("   => Training finished normally. Loading best weights.")
        
    # 7. Save the fully optimized foundation model
    torch.save(model.state_dict(), 'models/stage0_goes_opt.pth')
    print(f"Optimization complete! Final Brain state has Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
