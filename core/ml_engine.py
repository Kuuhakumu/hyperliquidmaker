# ml_engine.py - machine learning predictions for trading
import logging
import os
import time
import numpy as np
import joblib
import pandas as pd
from typing import Dict, Optional, Tuple

import config

logger = logging.getLogger("MLEngine")

class MLEngine:
    def __init__(self):
        self.model = None
        self.enabled = config.USE_ML_MODEL
        self.confidence_threshold = config.ML_CONFIDENCE_THRESHOLD
        
        if self.enabled:
            self.load_model()
            
    def load_model(self):
        """Load the trained model from disk."""
        path = config.ML_MODEL_PATH
        if os.path.exists(path):
            try:
                self.model = joblib.load(path)
                logger.info(f" ML Model loaded from {path}")
            except Exception as e:
                logger.error(f" Failed to load ML model: {e}")
                self.enabled = False
        else:
            logger.warning(f" ML Model not found at {path}. ML disabled.")
            self.model = None
            
    def predict(self, market_data, analysis) -> Tuple[float, float]:
        """
        Predict price direction.
        Returns (signal, confidence)
        signal: 1.0 (Buy), -1.0 (Sell), 0.0 (Neutral)
        confidence: 0.0 to 1.0
        """
        if not self.model or not self.enabled:
            return 0.0, 0.0
            
        try:
            # Extract features (Must match training features!)
            # Features: [imbalance_1, imbalance_5, spread, volatility, pressure]
            
            # Top 1 Imbalance
            bid_vol = market_data.bids[0][1] if market_data.bids else 0
            ask_vol = market_data.asks[0][1] if market_data.asks else 0
            top_imbalance = bid_vol / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.5
            
            # Depth Imbalance (Top 10)
            bid_depth = sum(b[1] for b in market_data.bids[:10])
            ask_depth = sum(a[1] for a in market_data.asks[:10])
            depth_imbalance = bid_depth / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0.5
            
            # Spread
            spread = (market_data.best_ask - market_data.best_bid) / market_data.mid_price
            
            # Construct feature vector
            # Ensure this matches training script exactly
            features = pd.DataFrame([{
                'top_imbalance': top_imbalance,
                'depth_imbalance': depth_imbalance,
                'spread': spread,
                'volatility': market_data.volatility,
                'pressure': analysis.vwap_pressure,
                'micro_diff': (analysis.micro_price - market_data.mid_price) / market_data.mid_price
            }])
            
            # Predict Probabilities
            # Classes: [0: Down, 1: Up] (Usually)
            probs = self.model.predict_proba(features)[0]
            
            # Assuming Class 1 is UP
            prob_up = probs[1]
            prob_down = probs[0]
            
            if prob_up > self.confidence_threshold:
                return 1.0, prob_up
            elif prob_down > self.confidence_threshold:
                return -1.0, prob_down
            else:
                return 0.0, max(prob_up, prob_down)
                
        except Exception as e:
            # logger.error(f"Prediction error: {e}") # Reduce noise
            return 0.0, 0.0
