"""
Train ML Model - Learns from captured market data.
Predicts next 10-tick price direction based on orderbook state.
"""

import json
import argparse
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def load_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    return data

def extract_features(data):
    """
    Extract features and targets.
    Target: 1 if price Up > threshold in N ticks, 0 if Down < -threshold, 2 else.
    But for simple binary, let's just do Up vs Not Up? Or Multi-class.
    Let's do: 1 (Up), 0 (Down/Flat). Or -1, 0, 1.
    """
    df = []
    
    # Needs to match MLEngine as closely as possible.
    # Recorder stores bids/asks; we can compute micro-price and a VWAP-like pressure here.
    for i, tick in enumerate(data):
        # Features
        bid_vol = tick['bids'][0][1] if tick['bids'] else 0
        ask_vol = tick['asks'][0][1] if tick['asks'] else 0
        top_imbalance = bid_vol / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0.5
        
        bid_depth = sum(b[1] for b in tick['bids'][:10])
        ask_depth = sum(a[1] for a in tick['asks'][:10])
        depth_imbalance = bid_depth / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0.5
        
        mid_price = tick['mp']
        spread = (tick['ba'] - tick['bb']) / mid_price
        
        # Prefer recorder-provided fields (new captures), fall back to deriving them (old captures).
        micro_diff = tick.get('micro_diff', None)
        pressure = tick.get('pressure', None)

        if micro_diff is None or pressure is None:
            # --- Micro-price (top-of-book) ---
            best_bid = tick['bb']
            best_ask = tick['ba']
            bid1_sz = tick['bids'][0][1] if tick['bids'] else 0.0
            ask1_sz = tick['asks'][0][1] if tick['asks'] else 0.0
            denom = (bid1_sz + ask1_sz)
            if denom > 0:
                micro_price = (best_bid * ask1_sz + best_ask * bid1_sz) / denom
            else:
                micro_price = mid_price

            if micro_diff is None:
                micro_diff = (micro_price - mid_price) / mid_price if mid_price else 0.0

            # --- VWAP-like pressure (-1..+1) ---
            # Similar spirit to OrderbookAnalyzer._calc_vwap_pressure.
            bids = tick.get('bids') or []
            asks = tick.get('asks') or []
            depth = min(10, len(bids), len(asks))
            bid_pressure = 0.0
            ask_pressure = 0.0
            pressure_dist = 50.0  # must stay stable across training runs

            for px, sz in bids[:depth]:
                dist = abs(px - mid_price) / mid_price if mid_price else 1.0
                w = 1.0 / (1.0 + dist * pressure_dist)
                bid_pressure += float(sz) * w
            for px, sz in asks[:depth]:
                dist = abs(px - mid_price) / mid_price if mid_price else 1.0
                w = 1.0 / (1.0 + dist * pressure_dist)
                ask_pressure += float(sz) * w

            total_pressure = bid_pressure + ask_pressure
            if pressure is None:
                if total_pressure > 0:
                    imbalance_pressure = bid_pressure / total_pressure
                    pressure = (imbalance_pressure - 0.5) * 2.0
                else:
                    pressure = 0.0
        
        row = {
            'top_imbalance': top_imbalance,
            'depth_imbalance': depth_imbalance,
            'spread': spread,
            'volatility': tick['vol'],
            'pressure': pressure,
            'micro_diff': micro_diff,
        }
        
        # TARGET GENERATION (Look ahead 10 ticks ~ 1 second)
        LOOK_AHEAD = 10
        if i < len(data) - LOOK_AHEAD:
            future_mid = data[i + LOOK_AHEAD]['mp']
            change = (future_mid - mid_price) / mid_price
            
            # Threshold: 2 bps (net of fees approx)
            if change > 0.0002:
                target = 1 # Up
            elif change < -0.0002:
                target = 0 # Down
            else:
                target = -1 # Flat (Ignore for binary? or Class 2)
            
            row['target'] = target
            df.append(row)
            
    return pd.DataFrame(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Path to capture file")
    args = parser.parse_args()
    
    print(f"Loading {args.file}...")
    raw_data = load_data(args.file)
    print(f"Loaded {len(raw_data)} ticks.")
    
    print("Extracting features...")
    df = extract_features(raw_data)
    
    # Filter out 'Flat' if desired, or keep as class
    # Let's filter out Flat to train a specific Signal Generator (only trade when sure)
    df_trade = df[df['target'] != -1]
    
    if len(df_trade) < 100:
        print("Not enough volatility to train (mostly flat). Try a longer capture.")
        return

    X = df_trade.drop('target', axis=1)
    y = df_trade['target']
    
    print(f"Training on {len(X)} samples...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Validate
    preds = clf.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, preds))
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.2%}")
    
    # Save
    if not os.path.exists("models"):
        os.makedirs("models")
        
    joblib.dump(clf, "models/direction_classifier.joblib")
    print("\n Model saved to models/direction_classifier.joblib")
    print("   Set USE_ML_MODEL=True in .env to enable.")

if __name__ == "__main__":
    main()
