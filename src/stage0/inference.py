import torch
import torch.nn.functional as F
import joblib
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.stage0.lstm_model import Stage0LSTM

class Stage0Predictor:
    def __init__(self, model_path=None, scaler_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'stage0_lstm.pth')
        if scaler_path is None:
            scaler_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'stage0_scaler.pkl')
            
        self.scaler = joblib.load(scaler_path)
        
        self.model = Stage0LSTM(input_size=5, hidden_size1=64, hidden_size2=32, num_classes=4)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def predict(self, sequence):
        """
        Predicts state probabilities for a given sequence.
        sequence shape: (seq_len, num_features) usually (60, 5)
        Returns: array of 4 probabilities [Quiet, Active, Eruptive, Recovery]
        """
        # Scale the sequence
        scaled_seq = self.scaler.transform(sequence)
        
        # Convert to tensor and add batch dimension
        seq_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(seq_tensor)
            probs = F.softmax(logits, dim=1)
            
        return probs.numpy()[0]
