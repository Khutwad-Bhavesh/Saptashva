import xgboost as xgb
import numpy as np

class EscalationEngine:
    def __init__(self):
        """
        Wrapper for the XGBoost model that implements the 3-tier one-way escalating alerts.
        Tiers:
        0 - None
        1 - Watch
        2 - Warning
        3 - Alert
        """
        self.model = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            objective='multi:softprob',
            num_class=4
        )
        self.current_state = 0
        self.is_trained = False
        
    def train(self, X, y):
        """
        Train the XGBoost model.
        X: Tabular features (e.g. LSTM probabilities + raw X-ray stats)
        y: True escalation tier
        """
        self.model.fit(X, y)
        self.is_trained = True
        
    def reset(self):
        """
        Reset the one-way state machine to None (0).
        Used when an event has fully passed or manually reset.
        """
        self.current_state = 0
        
    def predict_step(self, features):
        """
        Predict the next escalation state for a single timestep and apply one-way logic.
        features: 2D array-like of shape (1, num_features)
        Returns: (applied_state, predicted_state)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
            
        predicted_state = self.model.predict(features)[0]
        
        # One-way escalation logic
        if predicted_state > self.current_state:
            self.current_state = predicted_state
            
        return self.current_state, predicted_state
    
    def save_model(self, path):
        self.model.save_model(path)
        
    def load_model(self, path):
        self.model.load_model(path)
        self.is_trained = True
