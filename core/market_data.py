# market_data.py - connects to hyperliquid and gets price data
# uses websocket if available, otherwise polls REST api

import asyncio
import json
import logging
import time
import numpy as np
import threading
from collections import deque
from typing import Dict, Optional, Callable
from hyperliquid.info import Info
from hyperliquid.utils import constants

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    
    WEBSOCKETS_AVAILABLE = False
    
import config
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("MarketData")


class MarketDataManager:
    """gets market data from hyperliquid, keeps track of prices and orderbook"""

    def __init__(self, coin: str, use_testnet: bool = True):
        self.coin = coin
        self.use_testnet = use_testnet
        
        # API URL selection
        self.api_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
        
        # Websocket URL (Hyperliquid uses separate WS endpoint)
        if use_testnet:
            self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws"
        else:
            self.ws_url = "wss://api.hyperliquid.xyz/ws"
        
        # Mode selection: prefer websocket for lower latency
        self.use_websocket = getattr(config, 'USE_WEBSOCKET', True) and WEBSOCKETS_AVAILABLE
        self._ws_connected = False
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_lock = threading.Lock()

        # WS sequence tracking to detect drift (prev_n/n gaps)
        self._ws_last_n: Optional[int] = None
        
        # Initialize REST connection (fallback)
        self.session = self._create_optimized_session()
        self.info: Optional[Info] = None
        self._connected = False
        
        # Local state
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.mid_price = 0.0
        self.spread = 0.0
        self.spread_pct = 0.0
        
        # Orderbook depth (top 5 levels)
        self.bids: list = []  # [[price, size], ...]
        self.asks: list = []
        
        # Volume metrics
        self.bid_volume = 0.0
        self.ask_volume = 0.0
        self.imbalance = 0.5  # 0.0 = all sellers, 1.0 = all buyers

        # Depth-imbalance metric (decayed weighting), aligned with OrderbookAnalyzer.
        self.depth_imbalance = 0.5
        
        # Price history for volatility
        self.price_history = deque(maxlen=200)
        self.volatility = 0.0
        self.volatility_short = 0.0  # Short-term (10 ticks)
        
        # Callbacks for price updates
        self._callbacks: list[Callable] = []
        
        # Last update timestamp
        self.last_update = 0.0

        # Feed quality flags
        # True when we had to synthesize a missing side (one-sided book).
        self.is_synthetic_book = False

        # Rate-limit noisy warnings (e.g., one-sided books on testnet)
        self._last_rest_warning_time = 0.0
        
        # Funding rate tracking
        self.funding_rate = 0.0  # Current funding rate (positive = longs pay shorts)
        self.predicted_funding = 0.0  # Predicted next funding
        self._funding_last_fetch = 0.0  # Timestamp of last fetch
        self._funding_cache_seconds = getattr(config, 'FUNDING_CACHE_SECONDS', 60)

        self._funding_last_fetch = 0.0  # Timestamp of last fetch
        self._funding_cache_seconds = getattr(config, 'FUNDING_CACHE_SECONDS', 60)

    def _create_optimized_session(self) -> requests.Session:
        """make http session with retries"""
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10, 
            pool_maxsize=10,
            max_retries=Retry(
                total=3, 
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _warn_rate_limited(self, message: str, min_interval_sec: float = 5.0):
        now = time.time()
        if now - self._last_rest_warning_time >= min_interval_sec:
            self._last_rest_warning_time = now
            logger.warning(message)

    async def connect(self):
        """connect to hyperliquid api"""
        try:
            logger.info(f" Connecting to Hyperliquid {'Testnet' if self.use_testnet else 'Mainnet'}...")
            
            # Always init REST as fallback
            self.info = Info(self.api_url, skip_ws=True)
            if hasattr(self.info, 'session'):
                self.info.session = self.session
            self._connected = True

            # Seed initial state from REST so we have a usable book immediately
            # (even if WS is connected but hasn't delivered any updates yet).
            try:
                book = self.info.l2_snapshot(self.coin)
                levels = book.get('levels', []) if isinstance(book, dict) else []
                if len(levels) >= 2:
                    with self._ws_lock:
                        self._update_from_levels(levels[0], levels[1])
            except Exception as seed_err:
                logger.debug(f"REST seed snapshot failed: {seed_err}")
            
            # Try websocket for low-latency updates
            if self.use_websocket:
                try:
                    await self._connect_websocket()
                    logger.info(f" Connected via WEBSOCKET to {self.coin} L2 Book (low-latency mode)")
                    # Prevent an immediate 'stale' warning on the first loop iteration
                    # if the first WS message hasn't arrived yet.
                    if not self.last_update:
                        self.last_update = time.time()
                except Exception as ws_err:
                    logger.warning(f" Websocket failed: {ws_err} - falling back to REST polling")
                    self._ws_connected = False
            else:
                logger.info(f" Connected via REST polling to {self.coin} (add 'websockets' package for lower latency)")
            
        except Exception as e:
            logger.error(f" Connection failed: {e}")
            self._connected = False
            raise
    
    async def _connect_websocket(self):
        """start websocket connection"""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package not installed")
        
        # Start websocket listener task
        self._ws_task = asyncio.create_task(self._websocket_listener())
        
        # Wait briefly for initial connection
        await asyncio.sleep(0.5)
        
        if not self._ws_connected:
            raise ConnectionError("Websocket failed to connect")
    
    async def _websocket_listener(self):
        """listen for websocket messages in background"""
        import websockets
        
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    # Subscribe to L2 book
                    subscribe_msg = {
                        "method": "subscribe",
                        "subscription": {
                            "type": "l2Book",
                            "coin": self.coin
                        }
                    }
                    await ws.send(json.dumps(subscribe_msg))

                    # Reset sequence tracking on each (re)connect
                    self._ws_last_n = None
                    
                    self._ws_connected = True
                    retry_count = 0  # Reset on successful connect
                    logger.debug(f"WS subscribed to {self.coin} l2Book")
                    
                    # Listen for messages
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._process_ws_message(data)
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON from websocket")
                            
            except asyncio.CancelledError:
                logger.info("Websocket listener cancelled")
                break
            except Exception as e:
                self._ws_connected = False
                retry_count += 1
                logger.warning(f"Websocket error (retry {retry_count}/{max_retries}): {e}")
                await asyncio.sleep(min(2 ** retry_count, 30))  # Exponential backoff
        
        self._ws_connected = False
        logger.warning("Websocket listener stopped - falling back to REST")
    
    async def _process_ws_message(self, data: dict):
        """handle incoming ws message"""
        try:
            # Hyperliquid L2 book update format
            if data.get('channel') == 'l2Book':
                book_data = data.get('data', {})

                # --- Sequence drift protection ---
                # Hyperliquid provides sequence numbers to detect missed messages.
                # If we miss an update, our book is wrong; force reconnect.
                n = book_data.get('n', None)
                prev_n = book_data.get('prev_n', None)
                if prev_n is None:
                    prev_n = book_data.get('prevN', None)
                if prev_n is None:
                    prev_n = book_data.get('prevN', None)
                if prev_n is None:
                    prev_n = book_data.get('prev', None)
                if prev_n is None:
                    prev_n = book_data.get('prevN', None)

                if isinstance(n, (int, float)):
                    n = int(n)
                if isinstance(prev_n, (int, float)):
                    prev_n = int(prev_n)

                if self._ws_last_n is not None and prev_n is not None:
                    if prev_n != self._ws_last_n:
                        raise ConnectionError(
                            f"WS l2Book sequence gap: prev_n={prev_n} last_n={self._ws_last_n}"
                        )

                if n is not None:
                    self._ws_last_n = int(n)

                levels = book_data.get('levels', [])
                
                if len(levels) >= 2:
                    bids_raw = levels[0]
                    asks_raw = levels[1]
                    
                    with self._ws_lock:
                        self._update_from_levels(bids_raw, asks_raw)
                        
        except Exception as e:
            # Bubble up drift/connection errors so the listener can reconnect.
            if isinstance(e, ConnectionError):
                logger.warning(str(e))
                raise
            logger.debug(f"WS message processing error: {e}")
    
    def _update_from_levels(self, bids_raw: list, asks_raw: list):
        """update our local orderbook from the data we got"""
        bids_raw = bids_raw or []
        asks_raw = asks_raw or []

        self.is_synthetic_book = False

        # Some feeds (especially testnet) can briefly be one-sided.
        # Do NOT overwrite a last-good book with empty levels; keep the last book and
        # mark the feed as fresh so we don't permanently fall back to REST.
        if not bids_raw or not asks_raw:
            if self.bids and self.asks and self.last_update and (time.time() - self.last_update) < 15.0:
                self.last_update = time.time()
                return

            # No recent good book: synthesize a minimal missing side so the rest of the
            # pipeline can proceed (strategy guards should avoid trading into this).
            try:
                if not bids_raw and asks_raw:
                    best_ask = float(asks_raw[0].get('px', 0) or 0)
                    if best_ask > 0:
                        bids_raw = [{
                            'px': str(best_ask * (1 - 0.0005)),
                            'sz': str(asks_raw[0].get('sz', '1')),
                            'n': 1,
                        }]
                        self.is_synthetic_book = True
                elif not asks_raw and bids_raw:
                    best_bid = float(bids_raw[0].get('px', 0) or 0)
                    if best_bid > 0:
                        asks_raw = [{
                            'px': str(best_bid * (1 + 0.0005)),
                            'sz': str(bids_raw[0].get('sz', '1')),
                            'n': 1,
                        }]
                        self.is_synthetic_book = True
            except Exception:
                return

        bids = [[float(b['px']), float(b['sz'])] for b in bids_raw[:10]]
        asks = [[float(a['px']), float(a['sz'])] for a in asks_raw[:10]]

        if not bids or not asks:
            return

        # Update orderbook
        self.bids = bids
        self.asks = asks
        
        # Update top of book
        self.best_bid = self.bids[0][0]
        self.best_ask = self.asks[0][0]
        self.mid_price = (self.best_bid + self.best_ask) / 2
        
        # Spread calculation
        self.spread = self.best_ask - self.best_bid
        self.spread_pct = self.spread / self.mid_price if self.mid_price > 0 else 0
        
        # Volume imbalance (top 5 levels)
        self.bid_volume = sum(b[1] for b in self.bids[:5])
        self.ask_volume = sum(a[1] for a in self.asks[:5])
        total_vol = self.bid_volume + self.ask_volume
        self.imbalance = self.bid_volume / total_vol if total_vol > 0 else 0.5

        # Depth imbalance (decayed weighting across top levels)
        try:
            levels = 5
            oimb_decay = float(getattr(config, 'OB_OIMB_DECAY', 2.0) or 2.0)
            bid_weighted = 0.0
            ask_weighted = 0.0
            for i, (bid, ask) in enumerate(zip(self.bids[:levels], self.asks[:levels])):
                w = 1.0 / pow(i + 1, oimb_decay)
                bid_weighted += float(bid[1]) * w
                ask_weighted += float(ask[1]) * w
            total_w = bid_weighted + ask_weighted
            self.depth_imbalance = (bid_weighted / total_w) if total_w > 0 else 0.5
        except Exception:
            self.depth_imbalance = self.imbalance
        
        # Update price history
        self.price_history.append(self.mid_price)
        
        # Calculate volatility
        self._calculate_volatility()
        
        # Update timestamp
        self.last_update = time.time()
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def is_connected(self) -> bool:
        """check if connected"""
        return (self._ws_connected or self._connected) and self.info is not None
    
    def is_websocket_mode(self) -> bool:
        """check if using websocket"""
        return self._ws_connected

    async def update(self) -> bool:
        """
        Fetch latest orderbook data and update local state.
        Returns True if data was updated successfully.
        
        If websocket is connected, this just checks freshness.
        If using REST, this fetches new data.
        """
        if not self.is_connected():
            return False
        
        # If websocket is updating us, just verify freshness
        if self._ws_connected:
            # Check if we've received recent data (within 5 seconds)
            if time.time() - self.last_update < 5.0:
                return True
            else:
                self._warn_rate_limited("Websocket data stale, falling back to REST")
                # Fall through to REST update
        
        # REST fallback - Throttled to prevent 429 spam
        now = time.time()
        # If WS is down, only poll every 500ms (Hyperliquid snapshot limit is tight)
        rest_poll_interval = 0.5
        if now - getattr(self, '_last_rest_update', 0) < rest_poll_interval:
            return True # Use cached book from last successfully processed update/tick
            
        try:
            # Check if we're technically "rate limited" (if we had a helper for this)
            # For now, just use the frequency guard above.
            
            # Update last try time to prevent immediate retry
            self._last_rest_update = now
            
            # Get L2 book snapshot
            book = self.info.l2_snapshot(self.coin)
            
            if not book or 'levels' not in book:
                return False
                
            bids = book['levels'][0]
            asks = book['levels'][1]
            
            if not bids or not asks:
                # Some markets (especially testnet) can briefly show a one-sided book.
                # If we recently had a valid book, keep using it rather than failing every tick.
                self._warn_rate_limited(
                    f"REST l2_snapshot returned one-sided/empty book for {self.coin} (bids={len(bids)}, asks={len(asks)})"
                )
                if self.last_update and (time.time() - self.last_update) < 15.0:
                    return True

                # No recent good book: synthesize the missing side so the loop can continue.
                # This is safer than trading blindly: the strategy's stability checks should
                # generally prevent quoting into this condition.
                try:
                    if not bids and asks:
                        best_ask = float(asks[0].get('px', 0) or 0)
                        if best_ask > 0:
                            bids = [{
                                'px': str(best_ask * (1 - 0.0005)),
                                'sz': str(asks[0].get('sz', '1')),
                                'n': 1,
                            }]
                    elif not asks and bids:
                        best_bid = float(bids[0].get('px', 0) or 0)
                        if best_bid > 0:
                            asks = [{
                                'px': str(best_bid * (1 + 0.0005)),
                                'sz': str(bids[0].get('sz', '1')),
                                'n': 1,
                            }]
                except Exception:
                    return False

                if not bids or not asks:
                    return False
            
            # Use shared update logic (lock to avoid races with WS updates)
            with self._ws_lock:
                self._update_from_levels(bids, asks)
            
            return True
            
        except Exception as e:
            logger.error(f"Update error: {e}")
            return False

    def _calculate_volatility(self):
        """calculate how volatile the price is"""
        if len(self.price_history) < 10:
            self.volatility = 0.0
            self.volatility_short = 0.0
            return
            
        prices = np.array(list(self.price_history))
        
        # Calculate returns (percentage changes) - P2 fix
        returns = np.diff(prices) / prices[:-1]
        
        # Long-term volatility (last 50 returns)
        if len(returns) >= 50:
            recent_returns = returns[-50:]
            self.volatility = np.std(recent_returns)
        else:
            self.volatility = np.std(returns) if len(returns) > 0 else 0
            
        # Short-term volatility (last 10 returns) - for crash detection
        if len(returns) >= 10:
            short_returns = returns[-10:]
            self.volatility_short = np.std(short_returns)
        else:
            self.volatility_short = np.std(returns) if len(returns) > 0 else 0

    def get_micro_price(self) -> float:
        """get the micro price (weighted by volume on each side)"""
        if self.bid_volume + self.ask_volume == 0:
            return self.mid_price
            
        numerator = (self.best_bid * self.ask_volume) + (self.best_ask * self.bid_volume)
        denominator = self.bid_volume + self.ask_volume
        
        return numerator / denominator

    def get_price_change(self, lookback: int = 10) -> float:
        """
        Get percentage price change over last N ticks.
        Positive = price went up, Negative = price went down.
        """
        if len(self.price_history) < lookback + 1:
            return 0.0
            
        old_price = self.price_history[-lookback - 1]
        current = self.price_history[-1]
        
        if old_price == 0:
            return 0.0
            
        return (current - old_price) / old_price

    def get_trend_direction(self) -> str:
        """
        Simple trend detection based on recent price movement.
        Returns: "UP", "DOWN", or "NEUTRAL"
        """
        change = self.get_price_change(20)
        
        if change > 0.001:  # > 0.1%
            return "UP"
        elif change < -0.001:
            return "DOWN"
        return "NEUTRAL"

    def on_update(self, callback: Callable):
        """Register a callback to be called on each price update."""
        self._callbacks.append(callback)

    def get_snapshot(self) -> Dict:
        """
        Get a snapshot of current market state for logging/analysis.
        """
        return {
            "coin": self.coin,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread_pct": self.spread_pct,
            "imbalance": self.imbalance,
            "depth_imbalance": self.depth_imbalance,
            "volatility": self.volatility,
            "volatility_short": self.volatility_short,
            "micro_price": self.get_micro_price(),
            "trend": self.get_trend_direction(),
            "bid_volume": self.bid_volume,
            "ask_volume": self.ask_volume,
            "funding_rate": self.funding_rate,
            "predicted_funding": self.predicted_funding,
        }

    async def fetch_funding_rate(self) -> float:
        """
        Fetch current funding rate from Hyperliquid API.
        Caches result to avoid excessive API calls.
        Returns funding rate as decimal (e.g., 0.0001 = 0.01%)
        """
        current_time = time.time()
        
        # Use cached value if fresh
        if current_time - self._funding_last_fetch < self._funding_cache_seconds:
            return self.funding_rate
        
        try:
            if not self.info:
                return self.funding_rate
            
            # Fetch meta info which contains funding rates
            meta = self.info.meta()
            
            if meta and 'universe' in meta:
                for asset in meta['universe']:
                    if asset.get('name') == self.coin:
                        # Current funding rate
                        self.funding_rate = float(asset.get('funding', 0))
                        break
            
            # Also try to get predicted funding from asset contexts
            try:
                contexts = self.info.meta_and_asset_ctxs()
                if contexts and len(contexts) > 1:
                    asset_ctxs = contexts[1] if isinstance(contexts, list) else contexts.get('assetCtxs', [])
                    for i, ctx in enumerate(asset_ctxs):
                        # Match by index with universe
                        if meta and 'universe' in meta and i < len(meta['universe']):
                            if meta['universe'][i].get('name') == self.coin:
                                self.predicted_funding = float(ctx.get('funding', 0))
                                break
            except Exception:
                pass  # Predicted funding is optional
            
            self._funding_last_fetch = current_time
            logger.debug(f" Funding rate for {self.coin}: {self.funding_rate*100:.4f}% (predicted: {self.predicted_funding*100:.4f}%)")
            
        except Exception as e:
            logger.debug(f"Failed to fetch funding rate: {e}")
        
        return self.funding_rate

    async def run_update_loop(self, interval: float = 0.1):
        """
        Continuously update market data.
        Run this in a background task.
        """
        logger.info(f" Starting market data loop for {self.coin} (interval: {interval}s)")
        
        while True:
            try:
                await self.update()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logger.info("Market data loop cancelled")
                break
            except Exception as e:
                logger.error(f"Market data loop error: {e}")
                await asyncio.sleep(1)


class MultiCoinDataManager:
    """handles data for multiple coins at once"""

    def __init__(self, coins: list[str], use_testnet: bool = True):
        self.coins = coins
        self.managers: Dict[str, MarketDataManager] = {}
        
        for coin in coins:
            self.managers[coin] = MarketDataManager(coin, use_testnet)

    async def connect_all(self):
        """Connect to all coin feeds."""
        for coin, manager in self.managers.items():
            await manager.connect()

    def get(self, coin: str) -> Optional[MarketDataManager]:
        """Get the data manager for a specific coin."""
        return self.managers.get(coin)

    async def update_all(self):
        """Update all coin data."""
        tasks = [m.update() for m in self.managers.values()]
        await asyncio.gather(*tasks)
