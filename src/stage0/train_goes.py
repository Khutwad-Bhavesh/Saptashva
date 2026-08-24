import os
import copy
import torch
import torch.nn as nn
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

def main():
    print("=== SAPTASHVA Phase 4: Expanded Dataset Foundation Model ===")
    
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
    
    # CRITICAL FIX: Shuffle data before splitting to avoid temporal bias
    # (Temporal split was poisoning val set with unseen flare patterns)
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
    
    # 6. V1 Champion Architecture: (64, 32) with standard dropout
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Stage0LSTM(input_size=len(feature_cols), hidden_size1=64, hidden_size2=32, num_classes=4).to(device)
    
    # CRITICAL FIX: Class weights to handle 83% Active dominance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    full_weights = np.ones(4)
    for c, w in zip(classes, weights):
        full_weights[c] = w
    class_weights = torch.tensor(full_weights, dtype=torch.float).to(device)
    print(f"Computed Class Weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Adaptive Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print("\nStarting expanded dataset training with Early Stopping...")
    epochs = 30
    
    # Early Stopping tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
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
        
        # Early Stopping Logic (tracking val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"   => New best model! (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"   => Early stopping triggered after {epoch+1} epochs.")
                model.load_state_dict(best_model_weights)
                break
                
    if epochs_no_improve < patience:
        model.load_state_dict(best_model_weights)
        print("   => Training finished. Loading best weights.")
        
    # 7. Save the expanded foundation model
    torch.save(model.state_dict(), 'models/stage0_goes.pth')
    print(f"\n=== FINAL RESULT: Best Val Acc: {best_val_acc:.4f} | Best Val Loss: {best_val_loss:.4f} ===")

if __name__ == "__main__":
    main()
