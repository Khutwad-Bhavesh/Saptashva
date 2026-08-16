import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Ensure SAPTASHVA src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_generator import generate_synthetic_data
from src.features.engineering import extract_features, prepare_lstm_sequences
from src.stage0.lstm_model import Stage0LSTM
from src.data_pipeline.pradan_ingestion import process_pradan_directory, add_pseudo_labels

def train_stage0(epochs=10, batch_size=64, lr=0.001, data_dir=None):
    if data_dir and os.path.exists(data_dir):
        print(f"Loading real PRADAN data from {data_dir}...")
        df = process_pradan_directory(data_dir)
        df = add_pseudo_labels(df)
    else:
        print("Generating synthetic data...")
        df = generate_synthetic_data(10000)
    
    print("Extracting features...")
    df_features = extract_features(df)
    
    # We want 5 features
    feature_cols = ['soft_xray_flux', 'hard_xray_flux', 'flux_ratio', 'soft_flux_deriv', 'soft_flux_roll_std']
    
    print("Preparing sequences (lookback=60)...")
    X, y = prepare_lstm_sequences(df_features, feature_cols=feature_cols, target_col='state', lookback=60)
    
    # Scale features
    # Shape of X is (samples, lookback, features). We need to reshape to 2D for scaler, then back.
    samples, lookback, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(samples, lookback, num_features)
    
    # Train/Val split (80/20)
    split_idx = int(0.8 * samples)
    X_train, y_train = X_scaled[:split_idx], y[:split_idx]
    X_val, y_val = X_scaled[split_idx:], y[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, optimizer
    model = Stage0LSTM(input_size=5, hidden_size1=64, hidden_size2=32, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            _, predicted = torch.max(val_outputs.data, 1)
            accuracy = (predicted == y_val_tensor).sum().item() / len(y_val_tensor)
            
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {accuracy:.4f}")
        
    # Save model and scaler
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, 'stage0_lstm.pth'))
    joblib.dump(scaler, os.path.join(model_dir, 'stage0_scaler.pkl'))
    print("Model and scaler saved to models/")

if __name__ == '__main__':
    train_stage0(epochs=5)
