import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.escalation.xgboost_model import EscalationEngine

class EscalationPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'escalation_xgb.json')
            
        self.engine = EscalationEngine()
        if os.path.exists(model_path):
            self.engine.load_model(model_path)
            
    def predict(self, features):
        """
        features: 2D array of shape (1, num_features)
        Returns: (applied_state, predicted_state)
        """
        return self.engine.predict_step(features)
        
    def reset(self):
        self.engine.reset()
