# order_manager.py - places and cancels orders on hyperliquid
# tries to be smart about not sending unnecessary api calls

import time
import random
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import eth_account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

import config
import config
from core.strategy_engine import OrderLevel
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("OrderManager")


@dataclass
class ActiveOrder:
    """tracks one active order"""
    oid: str
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    timestamp: float
    reduce_only: bool = False


class OrderManager:
    """handles placing and cancelling orders, only sends changes to save api calls"""

    def __init__(self, private_key: str, coin: str, use_testnet: bool = True):
        self.coin = coin
        self.use_testnet = use_testnet
        
        # API setup
        api_url = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
        self.account = eth_account.Account.from_key(private_key)
        self.address = self.account.address
        
        self.info = Info(api_url, skip_ws=True)
        self.exchange = Exchange(self.account, api_url)

        # HFT OPTIMIZATION: Use persistent session with connection pooling
        self.session = self._create_optimized_session()
        
        # Inject optimized session into SDK objects
        # This replaces the default fresh-connection-per-call session
        if hasattr(self.info, 'session'):
            self.info.session = self.session
        
        if hasattr(self.exchange, 'session'):
            self.exchange.session = self.session
            logger.info(" Injected optimized persistent session into Exchange SDK")
        
        # Local order tracking
        self.active_orders: Dict[str, ActiveOrder] = {}
        
        # Rate limiting with 429 backoff
        self.weight_used = 0
        self.weight_reset_time = time.time()
        self.max_weight = config.MAX_WEIGHT_PER_MINUTE
        self.warning_threshold = config.WEIGHT_WARNING_THRESHOLD
        
        # 429 rate limit backoff state
        self._rate_limit_backoff = 0.0  # Current backoff delay in seconds
        self._rate_limit_until = 0.0    # Time when backoff expires
        self._consecutive_429s = 0       # Count of consecutive 429 errors
        
        # Price/size precision - fetch from API for accuracy
        self.size_decimals = config.get_size_precision(coin)  # Fallback
        self.price_decimals = config.get_price_precision(coin)  # Fallback
        self.tick_size = None  # Will be set by _fetch_asset_metadata
        
        # Fetch actual precision from HyperLiquid API
        self._fetch_asset_metadata()
        
        # Statistics
        self.orders_placed = 0
        self.orders_cancelled = 0
        self.orders_filled = 0
        
        # Maker/taker fill tracking
        self.maker_fills = 0
        self.taker_fills = 0
        
        # Sniper protection (jitter) settings
        self.jitter_enabled = getattr(config, 'ORDER_JITTER_ENABLED', True)
        self.jitter_min_ms = getattr(config, 'ORDER_JITTER_MIN_MS', 5.0)
        self.jitter_max_ms = getattr(config, 'ORDER_JITTER_MAX_MS', 50.0)

        # One-time warnings
        self._warned_maker_only_ioc = False
        
        # Adverse price move cancellation 
        self.adverse_cancel_enabled = getattr(config, 'ADVERSE_CANCEL_ENABLED', True)
        self.adverse_cancel_threshold_bps = getattr(config, 'ADVERSE_CANCEL_BPS', 2.0)  # Cancel if price moved 2bps against
        
        # Spread capture tracking
        self.spread_captures: list = []  # (entry_price, exit_price, side, timestamp)
        self.total_spread_captured_bps = 0.0
        self.avg_spread_captured_bps = 0.0
        
        # Queue position tracking
        self.queue_positions: Dict[str, float] = {}  # oid -> estimated queue depth
        
        # Dynamic maker ratio targeting
        self.target_maker_ratio = getattr(config, 'TARGET_MAKER_RATIO', 0.90)
        self.maker_ratio_window: list = []  # Rolling window of (is_maker, timestamp)
        
        # Partial fill handling 
        self.partial_fill_enabled = getattr(config, 'PARTIAL_FILL_REPLENISH', True)
        self.partial_fill_threshold = getattr(config, 'PARTIAL_FILL_THRESHOLD', 0.5)
        self.original_order_sizes: Dict[str, float] = {}  # oid -> original size

        # Cached state for 429 backoff (prevents false $0 readings)
        self._cached_account_value: float = 0.0
        self._cached_position: dict = {"size": 0, "entry_price": 0, "unrealized_pnl": 0, "side": "FLAT"}

    def _fetch_asset_metadata(self):
        """
        Fetch asset metadata (szDecimals, tick size) from HyperLiquid API.
        
        HyperLiquid price precision rules:
        - Prices can have up to 5 significant figures
        - Max decimal places = MAX_DECIMALS - szDecimals (MAX_DECIMALS=6 for perps)
        - Prices must be divisible by the tick size
        
        For ETH with szDecimals=3 and price ~3000:
        - Tick size = 0.10 (prices like 3130.80, 3130.90, etc.)
        """
        try:
            meta = self.info.meta()
            if meta and 'universe' in meta:
                for i, asset in enumerate(meta['universe']):
                    if asset.get('name') == self.coin:
                        # Get szDecimals from the API
                        sz_decimals = int(asset.get('szDecimals', 0))
                        self.size_decimals = sz_decimals
                        
                        # Calculate price decimals: MAX_DECIMALS (6 for perps) - szDecimals
                        # But also limited by 5 significant figures rule
                        max_price_decimals = 6 - sz_decimals
                        self.price_decimals = max(0, max_price_decimals)
                        
                        # Calculate tick size based on current price level
                        # For prices with 4-digit integer parts (like 3000.xx), tick size is 0.10
                        # We'll dynamically calculate tick size when rounding
                        # For now, set a default tick size that respects decimals
                        self.tick_size = 10 ** (-self.price_decimals)
                        
                        logger.info(f" Asset metadata fetched: {self.coin} szDecimals={sz_decimals}, "
                                    f"priceDecimals={self.price_decimals}, tickSize={self.tick_size}")
                        return
                        
            # If not found, use config defaults
            logger.warning(f" Could not fetch metadata for {self.coin}, using config defaults")
            self.tick_size = 10 ** (-self.price_decimals)
            
        except Exception as e:
            logger.warning(f" Error fetching asset metadata: {e}. Using config defaults.")
            self.tick_size = 10 ** (-self.price_decimals)

    def _create_optimized_session(self) -> requests.Session:
        """make http session with connection pooling"""
        session = requests.Session()
        
        # Connection Pooling: Keep 10 connections open, max 10
        # This prevents the "Connection Pool Full" warning and blocking
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

    def _is_rate_limited(self) -> bool:
        """Check if we're currently in a rate-limit backoff period."""
        if time.time() < self._rate_limit_until:
            return True
        return False

    def _handle_rate_limit(self):
        """Handle 429 rate limit by applying exponential backoff."""
        now = time.time()
        # If we already handled this in the last 500ms, don't increase backoff again
        if now < getattr(self, '_last_429_time', 0) + 0.5:
            return

        self._consecutive_429s += 1
        self._last_429_time = now
        
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
        self._rate_limit_backoff = min(30.0, (2 ** (self._consecutive_429s - 1)))
        self._rate_limit_until = now + self._rate_limit_backoff
        logger.warning(f" Rate limited (429). Backing off for {self._rate_limit_backoff:.1f}s (attempt #{self._consecutive_429s})")

    def _reset_rate_limit_backoff(self):
        """Reset backoff after a successful API call."""
        if self._consecutive_429s > 0:
            self._consecutive_429s = 0
            self._rate_limit_backoff = 0.0
            self._rate_limit_until = 0.0

    def _is_429_error(self, error: Exception) -> bool:
        """Check if an exception is a 429 rate limit error."""
        error_str = str(error)
        # Check for 429 in error tuple or message
        return '429' in error_str or 'rate limit' in error_str.lower()

    def _apply_jitter(self):
        """Apply random delay to prevent front-running/sniping."""
        if self.jitter_enabled:
            jitter_ms = random.uniform(self.jitter_min_ms, self.jitter_max_ms)
            time.sleep(jitter_ms / 1000.0)

    def get_fill_ratio(self) -> float:
        """Get maker fill ratio (higher is better for fees)."""
        total = self.maker_fills + self.taker_fills
        return self.maker_fills / total if total > 0 else 1.0

    def sync_orders(self) -> Dict[str, ActiveOrder]:
        """
        Sync local order cache with exchange state.
        Call this periodically to ensure we're in sync.
        """
        # Check if we're in rate-limit backoff
        if self._is_rate_limited():
            return self.active_orders
        
        try:
            open_orders = self.info.open_orders(self.address)
            self._reset_rate_limit_backoff()  # Successful call
            
            current_oids = set()
            new_orders = {}
            
            for order in open_orders:
                if order['coin'] != self.coin:
                    continue
                    
                oid = order['oid']
                current_oids.add(oid)
                
                # Check if this is a new order or existing
                if oid in self.active_orders:
                    # Update existing
                    new_orders[oid] = self.active_orders[oid]
                else:
                    # New order (maybe placed externally)
                    ro = False
                    try:
                        ro = bool(order.get('reduceOnly', order.get('reduce_only', False)))
                    except Exception:
                        ro = False
                    new_orders[oid] = ActiveOrder(
                        oid=oid,
                        price=float(order['limitPx']),
                        size=float(order['sz']),
                        side="BUY" if order['side'] == 'B' else "SELL",
                        timestamp=time.time(),
                        reduce_only=ro
                    )
            
            # Check for filled/cancelled orders
            filled_oids = set(self.active_orders.keys()) - current_oids
            if filled_oids:
                self.orders_filled += len(filled_oids)
                logger.info(f" {len(filled_oids)} order(s) filled or cancelled externally")
            
            self.active_orders = new_orders
            return self.active_orders
            
        except Exception as e:
            if self._is_429_error(e):
                self._handle_rate_limit()
            else:
                logger.error(f"Error syncing orders: {e}")
            return self.active_orders

    def update_queue_positions(self, bids: List[List[float]], asks: List[List[float]]):
        """
        Update estimated queue position for all active orders.
        
        Args:
            bids: Current bid orderbook
            asks: Current ask orderbook
        """
        for oid, order in self.active_orders.items():
            book = bids if order.side == "BUY" else asks
            self.queue_positions[oid] = self.estimate_queue_position(order.price, order.side, book)

    def estimate_queue_position(self, price: float, side: str, book: List[List[float]]) -> float:
        """
        Estimate how much volume is ahead of us in the queue.
        
        Args:
            price: Our order price
            side: "BUY" or "SELL"
            book: Orderbook side [[price, size], ...]
            
        Returns:
            Volume ahead of us
        """
        queue_depth = 0.0
        
        for level_price, level_size in book:
            # For BUY: Higher prices are ahead
            # For SELL: Lower prices are ahead
            if side == "BUY":
                if level_price > price + 0.00001: # Strictly greater
                    queue_depth += level_size
                elif abs(level_price - price) < 0.0001:  # Our level
                    # Assume we are roughly in the middle of our level
                    queue_depth += level_size * 0.5
                    break
                else:
                    # Price is lower than ours (behind us)
                    break
            else:  # SELL
                if level_price < price - 0.00001: # Strictly lower
                    queue_depth += level_size
                elif abs(level_price - price) < 0.0001:
                    queue_depth += level_size * 0.5
                    break
                else:
                    # Price is higher than ours (behind us)
                    break
            
        return queue_depth

    def execute_strategy(
        self,
        buy_orders: List[OrderLevel],
        sell_orders: List[OrderLevel],
        cancel_all: bool = False
    ) -> Tuple[int, int]:
        """
        Execute a strategy decision using order diffing.
        
        Args:
            buy_orders: Desired buy order levels
            sell_orders: Desired sell order levels
            cancel_all: If True, cancel all orders first
            
        Returns:
            Tuple of (orders_placed, orders_cancelled)
        """
        # Check if we're in rate-limit backoff - skip this cycle
        if self._is_rate_limited():
            remaining = self._rate_limit_until - time.time()
            if remaining > 0:
                logger.debug(f"Rate limited, skipping strategy execution ({remaining:.1f}s remaining)")
            return (0, 0)
        
        # Check rate limits
        self._check_rate_limit()
        
        placed = 0
        cancelled = 0
        
        # 1. Sync current state (Throttled for HFT)
        # We only sync from the exchange once every 10 ticks to avoid 
        # blocking the hot path with a redundant REST call.
        if getattr(self, '_tick_count', 0) % 10 == 0:
            self.sync_orders()
        self._tick_count = getattr(self, '_tick_count', 0) + 1
        
        # Cancel stale orders first (P0 Fix - rotten fish rule)
        stale_cancelled = self.cancel_stale_orders()
        cancelled += stale_cancelled
        
        if cancel_all:
            # Nuclear option - cancel everything
            cancelled = self._cancel_all_orders()
            self.active_orders = {}
            
        # Build target order sets (price, size, is_aggressive, reduce_only)
        target_buys = [(
            self._round_price(o.price),
            self._round_size(o.size),
            getattr(o, 'is_aggressive', False),
            bool(getattr(o, 'reduce_only', False)),
        ) for o in buy_orders]
        target_sells = [(
            self._round_price(o.price),
            self._round_size(o.size),
            getattr(o, 'is_aggressive', False),
            bool(getattr(o, 'reduce_only', False)),
        ) for o in sell_orders]
        
        # Separate current orders by side
        current_buys = {oid: o for oid, o in self.active_orders.items() if o.side == "BUY"}
        current_sells = {oid: o for oid, o in self.active_orders.items() if o.side == "SELL"}
        
        # DIFF: Find orders to cancel (not in target)
        orders_to_cancel = []
        
        for oid, order in current_buys.items():
            if not self._order_matches_target(order, [(p, s, ro) for p, s, _, ro in target_buys]):
                orders_to_cancel.append(oid)
                
        for oid, order in current_sells.items():
            if not self._order_matches_target(order, [(p, s, ro) for p, s, _, ro in target_sells]):
                orders_to_cancel.append(oid)
        
        # DIFF: Find orders to place (not already active with same price AND size)
        orders_to_place = []
        
        for price, size, is_agg, reduce_only in target_buys:
            if not self._target_exists(price, size, "BUY", reduce_only=reduce_only):
                orders_to_place.append({"side": "BUY", "price": price, "size": size,
                                        "is_aggressive": is_agg, "reduce_only": reduce_only})
                
        for price, size, is_agg, reduce_only in target_sells:
            if not self._target_exists(price, size, "SELL", reduce_only=reduce_only):
                orders_to_place.append({"side": "SELL", "price": price, "size": size,
                                        "is_aggressive": is_agg, "reduce_only": reduce_only})
        
        # Execute cancellations (Using BULK for speed)
        if orders_to_cancel:
            self._cancel_bulk(orders_to_cancel)
            cancelled = len(orders_to_cancel)
            self.orders_cancelled += cancelled
        
        # Execute placements
        if orders_to_place:
            self._place_orders(orders_to_place)
            placed = len(orders_to_place)
            self.orders_placed += placed
        
        if placed > 0 or cancelled > 0:
            logger.info(f" Diff result: +{placed} orders, -{cancelled} cancels")
        
        return placed, cancelled

    def _order_matches_target(
        self,
        order: ActiveOrder,
        targets: List[Tuple[float, float, bool]]
    ) -> bool:
        """Check if an existing order matches any target (price, size, reduce_only)."""
        for price, size, reduce_only in targets:
            # Price must be very close
            price_matches = abs(order.price - price) < 10 ** -(self.price_decimals + 1)
            # Size must also match (within 1% tolerance for rounding)
            size_matches = abs(order.size - size) / max(size, 0.0001) < 0.01
            reduce_matches = bool(getattr(order, 'reduce_only', False)) == bool(reduce_only)
            
            if price_matches and size_matches and reduce_matches:
                return True
        return False

    def _target_exists(self, price: float, size: float, side: str, reduce_only: bool = False) -> bool:
        """Check if a target already has an active order with matching price, size, and reduce_only."""
        for order in self.active_orders.values():
            if order.side == side:
                price_matches = abs(order.price - price) < 10 ** -(self.price_decimals + 1)
                size_matches = abs(order.size - size) / max(size, 0.0001) < 0.01
                reduce_matches = bool(getattr(order, 'reduce_only', False)) == bool(reduce_only)
                
                if price_matches and size_matches and reduce_matches:
                    return True
        return False

    def _place_orders(self, orders: List[dict]):
        """Place multiple orders in a batch with retry logic and sniper protection."""
        if not orders:
            return

        # Check rate-limit backoff
        if self._is_rate_limited():
            return

        # Apply jitter for sniper protection (anti-front-running)
        self._apply_jitter()

        # Determine TIF per order
        maker_only = getattr(config, 'MAKER_ONLY', False)
        maker_tif = getattr(config, 'MAKER_TIF', 'Alo')
        
        # Track which orders are taker vs maker for stats
        taker_count = 0
        maker_count = 0
            
        payloads = []
        for order in orders:
            is_aggressive = order.get("is_aggressive", False)
            reduce_only = order.get("reduce_only", False)

            # If MAKER_ONLY, force Post-Only (EXCEPT for reduce_only orders)
            # We allow exits (reduce_only) to be takers for safety during shocks.
            if maker_only and not reduce_only:
                tif = maker_tif
            else:
                tif = "Ioc" if is_aggressive else maker_tif
            
            # Track expected fill type
            if is_aggressive or tif == "Ioc":
                taker_count += 1
            else:
                maker_count += 1

            if (
                getattr(config, 'MAKER_ONLY', False)
                and is_aggressive
                and not reduce_only
                and not self._warned_maker_only_ioc
            ):
                self._warned_maker_only_ioc = True
                logger.warning(
                    "MAKER_ONLY=True but placing IOC (taker) order(s). "
                    "This is expected in HYBRID Hunter mode; set MAKER_ONLY=False if you want taker behavior explicitly."
                )

            payload = {
                "coin": self.coin,
                "is_buy": order["side"] == "BUY",
                "sz": order["size"],
                "limit_px": order["price"],
                "order_type": {"limit": {"tif": tif}},
                "reduce_only": reduce_only
            }
            payloads.append(payload)
        
        # Retry with exponential backoff (P1 fix)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.exchange.bulk_orders(payloads)
                self.weight_used += len(payloads)
                
                # Log results and track fill types
                actually_placed = 0
                if result and 'response' in result:
                    statuses = result['response'].get('data', {}).get('statuses', [])
                    for i, status in enumerate(statuses):
                        if 'resting' in status:
                            # Order is resting (maker)
                            oid = status['resting']['oid']
                            self.active_orders[oid] = ActiveOrder(
                                oid=oid,
                                price=orders[i]["price"],
                                size=orders[i]["size"],
                                side=orders[i]["side"],
                                timestamp=time.time(),
                                reduce_only=bool(orders[i].get("reduce_only", False))
                            )
                            actually_placed += 1
                        elif 'filled' in status:
                            # Order was immediately filled (taker)
                            self.taker_fills += 1
                            actually_placed += 1
                            logger.debug(f" Taker fill: {orders[i]['side']} {orders[i]['size']} @ {orders[i]['price']}")
                        elif 'error' in status:
                            # Order was rejected - log it!
                            err_msg = status.get('error', 'Unknown error')
                            logger.warning(f" Order rejected: {orders[i]['side']} {orders[i]['size']}@{orders[i]['price']} - {err_msg}")
                        else:
                            # Some other status (e.g., 'success' without resting means crossed & rejected for Alo)
                            logger.debug(f" Order status unknown: {status}")
                
                # Update maker fills count (orders that are now resting)
                self.maker_fills += actually_placed if actually_placed else maker_count
                self._reset_rate_limit_backoff()  # Successful call
                
                return  # Success, exit retry loop
                            
            except Exception as e:
                if self._is_429_error(e):
                    self._handle_rate_limit()
                    return  # Exit and let next cycle retry
                elif attempt < max_retries - 1:
                    backoff = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    logger.warning(f"Order placement failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.error(f"Order placement failed after {max_retries} attempts: {e}")
                    # Force sync to recover state
                    self.sync_orders()

    def _cancel_bulk(self, oids: List[str]):
        """Cancel multiple orders in a single bulk request (HFT best practice)."""
        if not oids:
            return

        # Check rate-limit backoff
        if self._is_rate_limited():
            return
            
        cancel_requests = [{"coin": self.coin, "oid": int(oid)} for oid in oids]
        
        try:
            result = self.exchange.bulk_cancel(cancel_requests)
            self.weight_used += len(oids)
            
            # Update local cache
            for oid in oids:
                if oid in self.active_orders:
                    del self.active_orders[oid]
            
            self._reset_rate_limit_backoff()
            logger.debug(f" Bulk cancelled {len(oids)} orders")
        except Exception as e:
            if self._is_429_error(e):
                self._handle_rate_limit()
            else:
                logger.error(f"Bulk cancel failed: {e}")
                # Fallback to individual for robustness? No, just sync next tick.
                self.sync_orders()

    def _cancel_orders(self, oids: List[str]):
        """Cancel specific orders by OID with individual retries (Deprecated for HFT)."""
        self._cancel_bulk(oids)

    def _cancel_all_orders(self) -> int:
        """Cancel all orders for this coin using bulk_cancel."""
        try:
            # First sync to get current orders
            self.sync_orders()
            
            if not self.active_orders:
                return 0
            
            count = len(self.active_orders)
            oids = list(self.active_orders.keys())
            
            # Use bulk_cancel with list of (coin, oid) tuples
            cancel_requests = [{"coin": self.coin, "oid": int(oid)} for oid in oids]
            
            if cancel_requests:
                result = self.exchange.bulk_cancel(cancel_requests)
                self.weight_used += len(cancel_requests)
                
                # Log any errors from the result
                if result and 'response' in result:
                    statuses = result['response'].get('data', {}).get('statuses', [])
                    errors = [s for s in statuses if 'error' in s]
                    if errors:
                        logger.warning(f"Some cancel requests failed: {errors}")
            
            self.active_orders = {}
            logger.info(f" Cancelled all orders ({count})")
            return count
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            # Fallback: try to cancel individually
            try:
                cancelled = 0
                for oid in list(self.active_orders.keys()):
                    try:
                        self.exchange.cancel(self.coin, int(oid))
                        cancelled += 1
                    except Exception:
                        pass
                self.active_orders = {}
                return cancelled
            except Exception:
                return 0

    def cancel_all(self):
        """Public method to cancel all orders."""
        return self._cancel_all_orders()
    
    def get_stale_orders(self) -> List[str]:
        """
        Identify orders that have been sitting too long (stale/rotten fish).
        Stale orders are "dumb money" that HFTs will pick off.
        
        Returns:
            List of order IDs that are stale and should be cancelled
        """
        stale_oids = []
        current_time = time.time()
        stale_threshold = config.ORDER_STALE_SECONDS
        
        # Don't check if stale detection is disabled (threshold = 0)
        if stale_threshold <= 0:
            return stale_oids
        
        for oid, order in self.active_orders.items():
            age = current_time - order.timestamp
            if age > stale_threshold:
                logger.debug(f" Stale order {oid}: {order.side} @ ${order.price:.4f} (age: {age:.1f}s)")
                stale_oids.append(oid)
        
        return stale_oids
    
    def get_adverse_orders(self, current_mid: float) -> List[str]:
        """
        Identify orders at risk due to adverse price movement .
        
        Critical for HFT: Cancel orders BEFORE they get picked off by faster traders.
        If mid price moved against our order, it's likely to be adversely selected.
        
        Args:
            current_mid: Current mid price
            
        Returns:
            List of order IDs that should be cancelled due to adverse price move
        """
        if not self.adverse_cancel_enabled:
            return []
        
        adverse_oids = []
        threshold = self.adverse_cancel_threshold_bps / 10000  # Convert bps to decimal
        
        for oid, order in self.active_orders.items():
            # Skip reduce-only orders (exits should stay)
            if order.reduce_only:
                continue
                
            if order.side == "BUY":
                # Our bid is at risk if price dropped below it
                # Price dropped = mid is now below our bid = we'll get filled on the way down
                price_move = (order.price - current_mid) / current_mid
                if price_move > threshold:  # Our bid is above mid by more than threshold
                    logger.debug(f" Adverse BUY {oid}: bid ${order.price:.4f} vs mid ${current_mid:.4f} ({price_move*10000:.1f}bps)")
                    adverse_oids.append(oid)
            else:  # SELL
                # Our ask is at risk if price rose above it
                price_move = (current_mid - order.price) / current_mid
                if price_move > threshold:  # Mid is above our ask by more than threshold
                    logger.debug(f" Adverse SELL {oid}: ask ${order.price:.4f} vs mid ${current_mid:.4f} ({price_move*10000:.1f}bps)")
                    adverse_oids.append(oid)
        
        return adverse_oids
    
    def cancel_adverse_orders(self, current_mid: float) -> int:
        """
        Cancel orders at risk of adverse selection.
        
        Returns:
            Number of orders cancelled
        """
        adverse_oids = self.get_adverse_orders(current_mid)
        if adverse_oids:
            logger.info(f" Cancelling {len(adverse_oids)} order(s) due to adverse price move")
            self._cancel_orders(adverse_oids)
            self.orders_cancelled += len(adverse_oids)
        return len(adverse_oids)
    
    def estimate_queue_position_for_order(self, order: ActiveOrder, orderbook_side: List[List[float]]) -> float:
        """
        Estimate how far back we are in the queue at our price level.
        
        Lower queue position = higher fill probability.
        This helps decide whether to reprice vs wait.
        
        Args:
            order: The order to check
            orderbook_side: Bid or ask side of orderbook [[price, size], ...]
            
        Returns:
            Estimated volume ahead of us in queue (0 = front of queue)
        """
        if not orderbook_side:
            return 0.0
        
        for price, size in orderbook_side:
            if abs(price - order.price) < 10 ** -(self.price_decimals + 1):
                # Found our price level - assume we're at the back
                self.queue_positions[order.oid] = size
                return size
        
        # Our order is not at any visible level (behind all visible depth)
        # Return sum of all visible depth as upper bound
        total_depth = sum(level[1] for level in orderbook_side[:5])
        self.queue_positions[order.oid] = total_depth
        return total_depth
    
    def record_spread_capture(self, entry_side: str, entry_price: float, exit_price: float, size: float):
        """
        Track actual spread captured per round-trip trade.
        
        This is the TRUE measure of market-making profitability.
        
        Args:
            entry_side: "BUY" or "SELL" for the entry leg
            entry_price: Price we entered at
            exit_price: Price we exited at
            size: Position size
        """
        if entry_side == "BUY":
            # We bought at entry, sold at exit
            spread_captured = exit_price - entry_price
        else:
            # We sold at entry, bought at exit
            spread_captured = entry_price - exit_price
        
        spread_captured_bps = (spread_captured / entry_price) * 10000
        
        self.spread_captures.append({
            'entry_side': entry_side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'spread_bps': spread_captured_bps,
            'spread_usd': spread_captured * size,
            'timestamp': time.time()
        })
        
        # Keep last 500 captures
        if len(self.spread_captures) > 500:
            self.spread_captures = self.spread_captures[-500:]
        
        # Update running totals
        self.total_spread_captured_bps += spread_captured_bps
        self.avg_spread_captured_bps = self.total_spread_captured_bps / len(self.spread_captures)
        
        logger.debug(f" Spread captured: {spread_captured_bps:.2f}bps (${spread_captured * size:.4f})")
    
    def get_current_maker_ratio(self) -> float:
        """
        Get maker fill ratio from recent fills.
        
        Returns:
            Ratio of maker fills (0.0 to 1.0)
        """
        # Clean old entries (keep last 5 minutes)
        current_time = time.time()
        self.maker_ratio_window = [
            entry for entry in self.maker_ratio_window
            if current_time - entry[1] < 300
        ]
        
        if not self.maker_ratio_window:
            return 1.0  # Assume 100% maker if no data
        
        maker_count = sum(1 for entry in self.maker_ratio_window if entry[0])
        return maker_count / len(self.maker_ratio_window)
    
    def record_fill_type(self, is_maker: bool):
        """
        Record whether a fill was maker or taker for ratio tracking.
        """
        self.maker_ratio_window.append((is_maker, time.time()))
        
        # Keep window reasonable size
        if len(self.maker_ratio_window) > 1000:
            self.maker_ratio_window = self.maker_ratio_window[-500:]
    
    def should_widen_spread(self) -> Tuple[bool, float]:
        """
        Check if spreads should be widened based on maker ratio.
        
        Returns:
            Tuple of (should_widen, suggested_multiplier)
        """
        current_ratio = self.get_current_maker_ratio()
        
        if current_ratio < self.target_maker_ratio - 0.05:
            # We're taking too much - widen spreads
            deficit = self.target_maker_ratio - current_ratio
            multiplier = 1.0 + (deficit * 2)  # Up to 1.2x wider
            return True, min(multiplier, 1.3)
        
        return False, 1.0
    
    def get_spread_capture_stats(self) -> dict:
        """
        Get spread capture statistics.
        """
        if not self.spread_captures:
            return {
                'count': 0,
                'avg_spread_bps': 0.0,
                'total_spread_usd': 0.0,
                'positive_pct': 0.0
            }
        
        total_usd = sum(c['spread_usd'] for c in self.spread_captures)
        positive_count = sum(1 for c in self.spread_captures if c['spread_bps'] > 0)
        
        return {
            'count': len(self.spread_captures),
            'avg_spread_bps': self.avg_spread_captured_bps,
            'total_spread_usd': total_usd,
            'positive_pct': positive_count / len(self.spread_captures) * 100
        }
    
    def get_partial_fills(self) -> List[Tuple[str, ActiveOrder, float]]:
        """
        Identify orders that have been partially filled.
        
        Returns:
            List of tuples: (oid, order, remaining_ratio)
            remaining_ratio is current_size / original_size
        """
        if not self.partial_fill_enabled:
            return []
        
        partial_fills = []
        
        try:
            # Sync with exchange to get current sizes
            open_orders = self.info.open_orders(self.address)
            
            for order_info in open_orders:
                if order_info['coin'] != self.coin:
                    continue
                
                oid = order_info['oid']
                current_size = float(order_info['sz'])
                
                # Check if we have original size tracked
                if oid in self.original_order_sizes:
                    original_size = self.original_order_sizes[oid]
                    if current_size < original_size:
                        remaining_ratio = current_size / original_size
                        if remaining_ratio <= self.partial_fill_threshold:
                            # More than threshold filled
                            if oid in self.active_orders:
                                partial_fills.append((oid, self.active_orders[oid], remaining_ratio))
                                
        except Exception as e:
            logger.debug(f"Error checking partial fills: {e}")
        
        return partial_fills
    
    def handle_partial_fills(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> int:
        """
        Handle partially filled orders by replenishing at new optimal levels.
        
        Args:
            bids: Current orderbook bids
            asks: Current orderbook asks
            mid_price: Current mid price
            
        Returns:
            Number of orders replenished
        """
        if not self.partial_fill_enabled:
            return 0
        
        partial_fills = self.get_partial_fills()
        replenished = 0
        
        for oid, order, remaining_ratio in partial_fills:
            # Cancel the remaining partial order
            try:
                self.exchange.cancel(self.coin, oid)
                if oid in self.active_orders:
                    del self.active_orders[oid]
                if oid in self.original_order_sizes:
                    del self.original_order_sizes[oid]
                    
                # Record the partial fill for maker ratio tracking
                filled_portion = 1.0 - remaining_ratio
                self.record_fill_type(is_maker=True)  # Partial fills are maker
                
                logger.info(f" Partial fill detected: {order.side} @ ${order.price:.4f} ({filled_portion:.0%} filled)")
                replenished += 1
                
            except Exception as e:
                logger.debug(f"Error handling partial fill {oid}: {e}")
        
        return replenished
    
    def track_original_size(self, oid: str, size: float):
        """Track original order size for partial fill detection."""
        self.original_order_sizes[oid] = size
    
    def cancel_stale_orders(self) -> int:
        """
        Cancel all stale orders.
        
        Returns:
            Number of stale orders cancelled
        """
        stale_oids = self.get_stale_orders()
        if stale_oids:
            logger.info(f" Cancelling {len(stale_oids)} stale order(s)")
            self._cancel_orders(stale_oids)
            self.orders_cancelled += len(stale_oids)
        return len(stale_oids)

    def _round_price(self, price: float) -> float:
        """
        Round price to valid precision for HyperLiquid.
        
        HyperLiquid price rules:
        1. Prices can have at most 5 significant figures
        2. Prices must be divisible by the tick size
        3. Tick size is derived from szDecimals: MAX_DECIMALS (6) - szDecimals
        
        For ETH (szDecimals=3) at price ~3000:
        - 4-digit integer part means we need to limit decimals
        - Tick size = 0.10 (round to nearest 0.10)
        
        Examples:
        - 3130.81 -> 3130.80 (round down to nearest 0.10)
        - 3130.85 -> 3130.90 (round up to nearest 0.10)
        - 3130.94 -> 3130.90 (round down to nearest 0.10)
        """
        import math
        
        if price <= 0:
            return 0.0
        
        # Calculate significant figures in the price
        # For HyperLiquid: max 5 significant figures
        MAX_SIG_FIGS = 5
        
        # Calculate the order of magnitude of the price
        order = math.floor(math.log10(price))
        
        # Calculate the tick size based on significant figures
        # The tick size is 10^(order - MAX_SIG_FIGS + 1)
        # For price = 3130, order = 3, tick = 10^(3-5+1) = 10^(-1) = 0.1
        sig_fig_tick = 10 ** (order - MAX_SIG_FIGS + 1)
        
        # Also respect the szDecimals-derived tick size
        # Use the larger of the two (more restrictive)
        if self.tick_size is not None:
            tick = max(sig_fig_tick, self.tick_size)
        else:
            tick = sig_fig_tick
        
        # Round to the nearest tick
        rounded_price = round(price / tick) * tick
        
        # Clean up floating point artifacts
        # Determine how many decimals to keep based on tick size
        if tick >= 1.0:
            return round(rounded_price)
        else:
            decimals = max(0, -int(math.floor(math.log10(tick))))
            return round(rounded_price, decimals)

    def _round_size(self, size: float) -> float:
        """Round size to valid precision."""
        return round(size, self.size_decimals)

    def analyze_fill_quality(self) -> dict:
        """Analyze fill quality for optimization."""
        if not self.spread_captures:
            return {}
        
        captures = self.spread_captures[-100:]  # Last 100
        positive = [c for c in captures if c['spread_bps'] > 0]
        negative = [c for c in captures if c['spread_bps'] <= 0]
        
        import numpy as np
        return {
            'win_rate': len(positive) / len(captures) if captures else 0,
            'avg_win_bps': np.mean([c['spread_bps'] for c in positive]) if positive else 0,
            'avg_loss_bps': np.mean([c['spread_bps'] for c in negative]) if negative else 0,
            'maker_ratio': self.maker_fills / (self.maker_fills + self.taker_fills) if (self.maker_fills + self.taker_fills) > 0 else 0
        }

    def _check_rate_limit(self):
        """Check and reset rate limit counter. Blocks if limit would be exceeded."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.weight_reset_time >= 60:
            self.weight_used = 0
            self.weight_reset_time = current_time
        
        # HARD STOP if at limit (P1 fix - actually prevent exceeding)
        if self.weight_used >= self.max_weight:
            sleep_time = 60 - (current_time - self.weight_reset_time)
            if sleep_time > 0:
                logger.warning(f" Rate limit reached ({self.weight_used}/{self.max_weight}). Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                self.weight_used = 0
                self.weight_reset_time = time.time()
        # Warn if approaching limit
        elif self.weight_used >= self.warning_threshold:
            logger.warning(f" Rate limit warning: {self.weight_used}/{self.max_weight}")
            time.sleep(0.5)  # Gentle slowdown

    def _update_user_state(self):
        """Internal helper to fetch user state (shared by position/account_value)."""
        now = time.time()
        # 1-second cache for HFT speed
        if now - getattr(self, '_last_user_state_time', 0) < 1.0:
            return
            
        # Update last try time immediately to prevent tick-by-tick retry on failure
        self._last_user_state_time = now

        # Check rate-limit backoff
        if self._is_rate_limited():
            return
            
        try:
            user_state = self.info.user_state(self.address)
            self._reset_rate_limit_backoff()
            
            # Cache account value
            val = float(user_state.get('marginSummary', {}).get('accountValue', 0))
            if val > 0:
                self._cached_account_value = val
                
            # Cache position
            found = False
            for pos in user_state.get('assetPositions', []):
                if pos['position']['coin'] == self.coin:
                    size = float(pos['position']['szi'])
                    entry_px = float(pos['position'].get('entryPx', 0))
                    unrealized_pnl = float(pos['position'].get('unrealizedPnl', 0))
                    
                    self._cached_position = {
                        "size": size,
                        "entry_price": entry_px,
                        "unrealized_pnl": unrealized_pnl,
                        "side": "LONG" if size > 0 else "SHORT" if size < 0 else "FLAT"
                    }
                    found = True
                    break
            
            if not found:
                self._cached_position = {"size": 0, "entry_price": 0, "unrealized_pnl": 0, "side": "FLAT"}
                
        except Exception as e:
            if self._is_429_error(e):
                self._handle_rate_limit()
            else:
                logger.debug(f"User state update failed (using cache): {e}")

    def get_position(self) -> dict:
        """Get current position for this coin (cached)."""
        self._update_user_state()
        return self._cached_position

    def get_account_value(self) -> float:
        """Get total account value in USD (cached)."""
        self._update_user_state()
        return self._cached_account_value

    def market_close_position(self):
        """
        Close the current position at market price.
        Used for emergency exits.
        """
        position = self.get_position()
        
        if abs(position["size"]) < 0.0001:
            return
            
        try:
            # Cancel all first
            self._cancel_all_orders()
            
            # Market close using IOC order far from current price
            is_buy = position["size"] < 0  # If short, buy to close
            
            # Use a very aggressive price to ensure fill
            self.exchange.market_close(self.coin)
            
            logger.warning(f" Market closed position: {position['size']}")
            
        except Exception as e:
            logger.error(f"Error market closing: {e}")

    def get_stats(self) -> dict:
        """Get order manager statistics."""
        return {
            "active_orders": len(self.active_orders),
            "orders_placed": self.orders_placed,
            "orders_cancelled": self.orders_cancelled,
            "orders_filled": self.orders_filled,
            "weight_used": self.weight_used,
            "rate_limit_pct": min((self.weight_used / self.max_weight) * 100, 100.0)  # P1 fix: cap at 100%
        }
    
    def get_recent_fills(self, limit: int = 100) -> List[dict]:
        """
        Fetch recent fills from the exchange (Throttled for HFT).
        """
        now = time.time()
        # Journaling only needs updates every 10 seconds, not every 500ms
        if now - getattr(self, '_last_fills_time', 0) < 10.0:
            return []
            
        self._last_fills_time = now
        try:
            # Check if we're in rate-limit backoff
            if self._is_rate_limited():
                return []
            
            # Hyperliquid API: user_fills returns recent fills
            fills = self.info.user_fills(self.address)
            self.weight_used += 1
            self._reset_rate_limit_backoff()  # Successful call
            
            # Filter to this coin and limit
            coin_fills = [f for f in fills if f.get('coin') == self.coin]
            return coin_fills[:limit]
            
        except Exception as e:
            if self._is_429_error(e):
                self._handle_rate_limit()
            else:
                logger.error(f"Error fetching fills: {e}")
            return []
