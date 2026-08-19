import torch
import torch.nn as nn

class Stage0LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size1=128, hidden_size2=64, num_classes=4):
        """
        Vanilla stacked LSTM for Solar Activity State Estimation.
        4-state output: Quiet, Active, Eruptive, Recovery.
        Architecture: LSTM(128) -> Dropout(0.3) -> LSTM(64) -> Dropout(0.2) -> Linear(4)
        """
        super(Stage0LSTM, self).__init__()
        
        # We use batch_first=True so input shape is (batch, seq_len, features)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # We only care about the output at the last time step for state estimation
        # out[:, -1, :] gets the hidden state of the last time step for all items in batch
        last_timestep_out = out[:, -1, :]
        
        logits = self.fc(last_timestep_out)
        return logits
