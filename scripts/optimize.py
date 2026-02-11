import json
import argparse
import random
import logging
import sys
import os
import itertools
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace
from typing import List, Dict, Tuple

# Mock minimal config for import
import config

# Import core engines
from core.strategy_engine import StrategyEngine
from core.regime_engine import RegimeEngine, TradingRegime

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Optimizer")

class SimulationEnvironment:
    """
    Simulates market environment from recorded data.
    """
    def __init__(self, data_file: str, split_ratio: float = 0.7):
        self.data_file = data_file
        self.split_ratio = split_ratio
        self.train_ticks = []
        self.test_ticks = []
        self._load_and_split_data()

    def _load_and_split_data(self):
        """Load JSONL data and split into train/test sets."""
        logger.info(f"Loading data from {self.data_file}...")
        all_ticks = []
        try:
            with open(self.data_file, 'r') as f:
                for line in f:
                    try:
                        all_ticks.append(json.loads(line))
                    except:
                        pass
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_file}")
            return

        # Sort by timestamp just in case
        all_ticks.sort(key=lambda x: x['ts'])
        
        # Split
        split_idx = int(len(all_ticks) * self.split_ratio)
        self.train_ticks = all_ticks[:split_idx]
        self.test_ticks = all_ticks[split_idx:]
        
        logger.info(f"Data Loaded: {len(all_ticks)} total ticks")
        logger.info(f"   Train Set: {len(self.train_ticks)} ticks ({self.split_ratio:.0%})")
        logger.info(f"   Test Set:  {len(self.test_ticks)} ticks")

    def run_backtest(self, params: Dict, use_test_set: bool = False) -> Dict:
        """
        Run backtest with specific parameters.
        Returns: Dict with PnL, Trade Count, Trades/Min
        """
        ticks = self.test_ticks if use_test_set else self.train_ticks
        
        if not ticks:
            return {'pnl': 0.0, 'trades': 0, 'tpm': 0.0}

        # Setup simulation duration
        start_time = ticks[0]['ts']
        end_time = ticks[-1]['ts']
        duration_min = (end_time - start_time) / 60.0
        
        # 1. Setup simulated bot state
        regime_engine = RegimeEngine()
        strategy_engine = StrategyEngine()
        
        # Override config params (Save old values)
        old_vol_thresh = config.VOLATILITY_THRESHOLD
        old_skew = config.SKEW_FACTOR
        old_bull = config.IMBALANCE_BULL_THRESHOLD
        old_bear = config.IMBALANCE_BEAR_THRESHOLD
        old_spread = config.BASE_SPREAD
        old_ob_decay = getattr(config, 'OB_OIMB_DECAY', 1.5)
        old_press_scale = getattr(config, 'VWAP_PRESSURE_SCALER', 0.0003)
        old_skew_ent = getattr(config, 'SKEW_MODE_ENTRY_THRESHOLD', 0.5)
        old_vol_exit = getattr(config, 'VOLATILITY_EXIT_MULTIPLIER', 0.7)
        old_hunter_agg = getattr(config, 'HUNTER_MAX_AGGRESSION_BPS', 5.0)
        
        # Apply Overrides
        config.VOLATILITY_THRESHOLD = params.get('vol_thresh', old_vol_thresh)
        config.BASE_SPREAD = params.get('spread', old_spread)
        
        if 'ob_decay' in params: config.OB_OIMB_DECAY = params['ob_decay']
        if 'press_scale' in params: config.VWAP_PRESSURE_SCALER = params['press_scale']
        if 'skew_ent' in params: 
            config.SKEW_MODE_ENTRY_THRESHOLD = params['skew_ent']
            config.SKEW_MODE_EXIT_THRESHOLD = params['skew_ent'] / 2
        if 'vol_exit_mult' in params: config.VOLATILITY_EXIT_MULTIPLIER = params['vol_exit_mult']
        if 'hunter_agg' in params: config.HUNTER_MAX_AGGRESSION_BPS = params['hunter_agg']

        # Simulation State
        position_size = 0.0
        entry_price = 0.0
        realized_pnl = 0.0
        trade_count = 0
        
        for tick in ticks:
            # Reconstruct market inputs
            mid_price = tick['mp']
            volatility = tick['vol']
            imbalance = tick['imb']
            
            # 1. Regime Eval
            regime, metadata = regime_engine.evaluate(
                volatility=volatility,
                imbalance=imbalance,
                current_position_usd=position_size * mid_price,
                max_position_usd=1000
            )
            
            # 2. Strategy Eval
            bids = tick.get('bids', [])
            asks = tick.get('asks', [])
            
            if not bids or not asks:
                continue
                
            decision = strategy_engine.generate_orders(
                regime=regime,
                metadata=metadata,
                mid_price=mid_price,
                bids=bids,
                asks=asks,
                current_position_usd=position_size * mid_price,
                volatility=volatility
            )
            
            # 3. Fill Simulation
            best_bid = tick['bb']
            best_ask = tick['ba']
            
            filled = False
            
            # Process Buys
            for order in decision.buy_orders:
                # Taker fill (crossed spread)
                if order.price >= best_ask:
                    fill_price = best_ask
                    fee = fill_price * order.size * config.TAKER_FEE
                    
                    if position_size >= 0:
                         entry_price = ((position_size * entry_price) + (order.size * fill_price)) / (position_size + order.size)
                    else:
                        closing_size = min(abs(position_size), order.size)
                        realized_pnl += (entry_price - fill_price) * closing_size
                        if order.size > abs(position_size):
                             entry_price = fill_price 
                    
                    position_size += order.size
                    realized_pnl -= fee
                    trade_count += 1
                    filled = True
                
                # Maker fill (price matched best bid)
                elif order.price <= best_bid:
                     if order.price == best_bid and random.random() < 0.2:
                        fill_price = order.price
                        fee = fill_price * order.size * config.MAKER_FEE 
                        
                        if position_size >= 0:
                             entry_price = ((position_size * entry_price) + (order.size * fill_price)) / (position_size + order.size)
                        else:
                            closing_size = min(abs(position_size), order.size)
                            realized_pnl += (entry_price - fill_price) * closing_size
                            if order.size > abs(position_size):
                                 entry_price = fill_price
                        
                        position_size += order.size
                        realized_pnl -= fee
                        trade_count += 1
                        filled = True

            if filled: 
                continue # Don't fill buy and sell in same tick for sim simplicity

            # Process Sells
            for order in decision.sell_orders:
                # Taker fill
                if order.price <= best_bid:
                    fill_price = best_bid
                    fee = fill_price * order.size * config.TAKER_FEE
                    
                    if position_size <= 0:
                        entry_price = ((abs(position_size) * entry_price) + (order.size * fill_price)) / (abs(position_size) + order.size)
                    else:
                        closing_size = min(position_size, order.size)
                        realized_pnl += (fill_price - entry_price) * closing_size
                        if order.size > position_size:
                            entry_price = fill_price
                            
                    position_size -= order.size
                    realized_pnl -= fee
                    trade_count += 1
                    
                # Maker fill
                elif order.price >= best_ask:
                    if order.price == best_ask and random.random() < 0.2:
                        fill_price = order.price
                        fee = fill_price * order.size * config.MAKER_FEE
                        
                        if position_size <= 0:
                            entry_price = ((abs(position_size) * entry_price) + (order.size * fill_price)) / (abs(position_size) + order.size)
                        else:
                            closing_size = min(position_size, order.size)
                            realized_pnl += (fill_price - entry_price) * closing_size
                            if order.size > position_size:
                                entry_price = fill_price
                        
                        position_size -= order.size
                        realized_pnl -= fee
                        trade_count += 1
        
        # Restore config
        config.VOLATILITY_THRESHOLD = old_vol_thresh
        config.BASE_SPREAD = old_spread
        config.SKEW_FACTOR = old_skew
        config.IMBALANCE_BULL_THRESHOLD = old_bull
        config.IMBALANCE_BEAR_THRESHOLD = old_bear
        config.OB_OIMB_DECAY = old_ob_decay
        config.VWAP_PRESSURE_SCALER = old_press_scale
        config.SKEW_MODE_ENTRY_THRESHOLD = old_skew_ent
        config.SKEW_MODE_EXIT_THRESHOLD = old_skew_ent / 2
        config.VOLATILITY_EXIT_MULTIPLIER = old_vol_exit
        config.HUNTER_MAX_AGGRESSION_BPS = old_hunter_agg
        
        tpm = trade_count / duration_min if duration_min > 0 else 0
        return {
            'pnl': realized_pnl,
            'trades': trade_count,
            'tpm': tpm
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="Capture file to replay")
    args = parser.parse_args()
    
    # Init sim with 70/30 Test/Train split
    sim = SimulationEnvironment(args.file, split_ratio=0.70)
    
    # Grid Search Parameters
    # We now have MANY parameters. Using a focused grid to prevent explosion.
    vol_thresholds = [0.003, 0.005]           # Higher vol needed for HFT safety
    spreads = [0.0003, 0.0005, 0.0010]        # 3, 5, 10 bps
    # skews = [0.001, 0.002]                  # Removing skew factor to prioritize meaningful params
    
    # NEW params
    ob_decays = [0.5, 1.0, 1.5]               # Depth perception (low vs high decay)
    press_scalers = [0.0001, 0.0003]          # Impact of orderbook pressure
    skew_entry = [0.3, 0.5]                   # Position skew entry (30% vs 50%)
    
    # Additional Params
    hunter_aggs = [3.0, 5.0]                  # 3bps vs 5bps aggression
    vol_exits = [0.7, 0.9]                    # Hysteresis (0.7 = lazy exit, 0.9 = quick exit)

    logger.info(" Starting Forward-Walk Optimization...")
    logger.info("   Phase 1: Finding best params on TRAIN set (In-Sample)")
    
    best_train_pnl = -float('inf')
    best_params = {}
    
    # Grid Search using itertools
    param_grid = list(itertools.product(
        vol_thresholds, spreads, ob_decays, press_scalers, skew_entry, hunter_aggs, vol_exits
    ))
    
    total_combinations = len(param_grid)
    logger.info(f"   Testing {total_combinations} combinations...")
    
    counter = 0
    
    for p in param_grid:
        counter += 1
        if counter % 10 == 0:
            print(f"   Progress: {counter}/{total_combinations}...", end='\r')
            
        # Unpack parameters
        vol, spread, ob_decay, press_scale, skew_ent, hunter_agg, vol_exit = p
        
        params = {
            'vol_thresh': vol, 
            'spread': spread,
            'ob_decay': ob_decay,
            'press_scale': press_scale,
            'skew_ent': skew_ent,
            'hunter_agg': hunter_agg,
            'vol_exit_mult': vol_exit
        }
        try:
            res = sim.run_backtest(params)
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            continue
            
        # Evaluation Metric: PnL > 0 AND Trades > 0 (relaxed for bootstrap)
        if res['pnl'] > best_train_pnl and res['trades'] > 0:
            best_train_pnl = res['pnl']
            best_params = params.copy()
            # print(f"   New Best: ${res['pnl']:.2f} ({res['tpm']:.1f} tpm) | {params}")

    if not best_params:
        logger.error(" No parameters met the minimum trade frequency criteria!")
        return

    logger.info("")
    logger.info(f" BEST TRAIN PARAMETERS")
    logger.info(f"   {best_params}")
    logger.info(f"   In-Sample PnL: ${best_train_pnl:.2f}")
    
    # Phase 2: Validate on TEST set
    logger.info("\n   Phase 2: Validating on TEST set (Out-of-Sample)")
    test_res = sim.run_backtest(best_params, use_test_set=True)
    
    logger.info(f"   Out-of-Sample PnL: ${test_res['pnl']:.2f}")
    logger.info(f"   Trade Freq: {test_res['tpm']:.1f} trades/min")
    
    if test_res['pnl'] > 0:
        logger.info("    VALIDATION SUCCESS: Strategy is profitable on unseen data.")
    else:
        logger.info("    VALIDATION WARNING: Strategy lost money on unseen data (Likely Curve Fits).")
        logger.info("    SAVING ANYWAY for bootstrap purposes.")

    # Save best parameters for the bot to pick up
    save_path = "data/best_params.json"
    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"    Saved optimized params to {save_path}")
    logger.info("")

if __name__ == "__main__":
    main()
