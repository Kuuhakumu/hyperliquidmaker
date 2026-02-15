# recorder.py - records market data for backtesting
import json
import logging
import os
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any
import config

logger = logging.getLogger("Recorder")

class DataRecorder:
    """
    asynchronous market data recorder.
    Buffers snapshots and writes them to disk in a background thread.
    """

    def __init__(self, coin: str):
        self.coin = coin
        self.running = False
        self.queue = queue.Queue()
        self.thread = None
        self.file_handle = None
        self.filename = ""
        
        # Ensure data directory exists
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def start(self):
        """Start the recording thread."""
        if self.running:
            return

        # Cleanup old files
        self._cleanup_old_files()

        # Create new daily file
        date_str = datetime.now().strftime("%Y-%m-%d")
        ts_str = int(time.time())
        self.filename = os.path.join(self.data_dir, f"capture_{self.coin}_{date_str}_{ts_str}.jsonl")
        
        try:
            self.file_handle = open(self.filename, 'a', encoding='utf-8')
            logger.info(f" Recording market data to {self.filename}")
        except Exception as e:
            logger.error(f"Failed to open recording file: {e}")
            return

        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True, name="DataRecorder")
        self.thread.start()

    def _cleanup_old_files(self):
        """Delete old capture files to save space."""
        try:
            files = [f for f in os.listdir(self.data_dir) if f.startswith("capture_") and f.endswith(".jsonl")]
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)))
            
            # Default to 10 files if config not found
            max_files = getattr(config, 'MAX_DATA_RETENTION_FILES', 10)
            
            while len(files) >= max_files:
                oldest = files.pop(0)
                path = os.path.join(self.data_dir, oldest)
                try:
                    os.remove(path)
                    logger.info(f" Deleted old capture file: {oldest}")
                except Exception as e:
                    logger.warning(f"Failed to delete {oldest}: {e}")
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")

    def stop(self):
        """Stop the recorder."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            logger.info(" Recording stopped.")

    def snapshot(self, market_data, regime: str, strategy_decision=None):
        """
        Take a snapshot of the current state.
        This must be fast/non-blocking.
        """
        if not self.running:
            return

        # Extract only essential data to minimize overhead
        # Deep copy is expensive, so we just extract primitives
        try:
            # --- ML parity features (computed from the same recorded bids/asks) ---
            # Keep this simple and stable so training/inference stay aligned.
            bids10 = [x[:2] for x in market_data.bids[:10]]
            asks10 = [x[:2] for x in market_data.asks[:10]]

            mid = float(market_data.mid_price or 0.0)
            best_bid = float(market_data.best_bid or 0.0)
            best_ask = float(market_data.best_ask or 0.0)

            bid1_sz = float(bids10[0][1]) if bids10 else 0.0
            ask1_sz = float(asks10[0][1]) if asks10 else 0.0
            denom = bid1_sz + ask1_sz
            micro_price = ((best_bid * ask1_sz) + (best_ask * bid1_sz)) / denom if denom > 0 else mid
            micro_diff = (micro_price - mid) / mid if mid else 0.0

            # VWAP-like pressure (-1..+1)
            pressure_dist = 50.0
            bid_pressure = 0.0
            ask_pressure = 0.0
            if mid and bids10 and asks10:
                for px, sz in bids10:
                    dist = abs(float(px) - mid) / mid
                    w = 1.0 / (1.0 + dist * pressure_dist)
                    bid_pressure += float(sz) * w
                for px, sz in asks10:
                    dist = abs(float(px) - mid) / mid
                    w = 1.0 / (1.0 + dist * pressure_dist)
                    ask_pressure += float(sz) * w

            total_pressure = bid_pressure + ask_pressure
            if total_pressure > 0:
                imbalance_pressure = bid_pressure / total_pressure
                pressure = (imbalance_pressure - 0.5) * 2.0
            else:
                pressure = 0.0

            snapshot = {
                "ts": time.time(),  # timestamp
                "mp": market_data.mid_price,  # mid price
                "bb": market_data.best_bid,   # best bid
                "ba": market_data.best_ask,   # best ask
                "imb": market_data.imbalance, # imbalance
                "vol": market_data.volatility,# volatility
                "r": regime,                  # current regime

                # ML parity fields
                "micro_price": micro_price,
                "micro_diff": micro_diff,
                "pressure": pressure,
                
                # Orderbook (Top 10 to match OrderbookAnalyzer depth)
                # list of [price, size]
                "bids": bids10,
                "asks": asks10,
            }
            
            # Optional: Record strategy decision if provided
            if strategy_decision:
                # Just record the prices we wanted to quote
                # format: [buy_p1, buy_p2], [sell_p1, sell_p2]
                snapshot["strat"] = {
                    "b": [o.price for o in strategy_decision.buy_orders],
                    "s": [o.price for o in strategy_decision.sell_orders] 
                }

            self.queue.put(snapshot)
            
        except Exception as e:
            # Don't crash the bot if recording fails
            if getattr(self, '_log_error_count', 0) < 5:
                logger.error(f" Recorder Snapshot Error: {e}")
                self._log_error_count = getattr(self, '_log_error_count', 0) + 1
            pass

    def _writer_loop(self):
        """Background loop to write data from queue to disk."""
        while self.running or not self.queue.empty():
            try:
                # Get batch of items if possible
                items = []
                try:
                    # Blocking get with timeout (wait for at least one item)
                    item = self.queue.get(timeout=1.0)
                    items.append(item)
                    
                    # Grab any others immediately available (up to 100)
                    # We catch Empty locally to stop the batching, but proceed to write
                    try:
                        for _ in range(100):
                            items.append(self.queue.get_nowait())
                    except queue.Empty:
                        pass # Batch complete
                        
                except queue.Empty:
                    # Outer empty means no data for 1.0s, just loop
                    continue

                if self.file_handle and items:
                    lines = [json.dumps(x) for x in items]
                    self.file_handle.write("\n".join(lines) + "\n")
                    self.file_handle.flush()
                    
            except Exception as e:
                logger.error(f"Recorder write error: {e}")
                time.sleep(1)
