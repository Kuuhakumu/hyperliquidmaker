# orderbook_analyzer.py - looks at the orderbook to find good prices
# detects walls, calculates imbalance, figures out where to place orders

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque
import time

import config

logger = logging.getLogger("OrderbookAnalyzer")


@dataclass
class OrderbookLevel:
    """Single orderbook level with analysis data."""
    price: float
    size: float
    cumulative_size: float  # Total size up to this level
    distance_from_mid: float  # % distance from mid price
    is_wall: bool = False  # True if this is a significant cluster


@dataclass 
class OrderbookAnalysis:
    """Full orderbook analysis result."""
    # Spread analysis
    current_spread_bps: float  # Spread in basis points (1bp = 0.01%)
    adaptive_spread_bps: float  # Recommended spread based on conditions
    
    # Imbalance (0 = all sellers, 1 = all buyers)
    top_imbalance: float  # Top 1 level
    depth_imbalance: float  # Weighted across 10 levels
    cumulative_imbalance: float  # Total volume ratio
    
    # Walls/Clusters
    bid_wall_price: Optional[float]
    bid_wall_size: float
    ask_wall_price: Optional[float]  
    ask_wall_size: float
    
    # Strategic levels
    optimal_bid_price: float  # Where to place bid
    optimal_ask_price: float  # Where to place ask
    
    # Micro-structure
    micro_price: float  # Volume-weighted fair value
    vwap_pressure: float  # -1 to 1, buy vs sell pressure
    
    # Trade flow (if trades tracked)
    recent_buy_volume: float
    recent_sell_volume: float
    trade_flow_bias: float  # -1 to 1
    
    # Confidence
    liquidity_score: float  # 0-1, how liquid is this market
    is_stable: bool  # True if orderbook is stable enough to trade


class OrderbookAnalyzer:
    """analyzes the orderbook to find walls, calculate spread, etc"""

    def __init__(self):
        # Config for wall detection
        self.wall_threshold = getattr(config, 'WALL_THRESHOLD', 3.0)
        self.depth_levels = getattr(config, 'ORDERBOOK_DEPTH', 10)
        
        # Trade flow tracking
        self.trade_history: deque = deque(maxlen=200)
        self.last_mid_price = 0.0
        
        
        # Spread calculation params - from config for HFT
        self.min_spread_bps = getattr(config, 'MIN_SPREAD_BPS', 1.5)  # Very tight for HFT
        self.max_spread_bps = getattr(config, 'MAX_SPREAD_BPS', 20.0)
        self.base_spread_bps = getattr(config, 'BASE_SPREAD_BPS', 3.0)  # Tight default
        
        # Physics Params
        self.oimb_decay = config.OB_OIMB_DECAY
        self.pressure_dist_factor = config.OB_PRESSURE_DIST_FACTOR
        self.vol_adjust_scaler = config.OB_VOL_ADJUST_SCALER
        self.liq_depth_divisor = config.OB_LIQ_DEPTH_DIVISOR
        
        # For detecting changes
        self.last_bid_wall = None
        self.last_ask_wall = None
        
        # === FLICKER PROTECTION (Anti-Spoofing) ===
        # Track wall persistence to filter out spoof orders
        self.wall_flicker_enabled = getattr(config, 'WALL_FLICKER_ENABLED', True)
        self.wall_min_persistence_sec = getattr(config, 'WALL_MIN_PERSISTENCE_SEC', 2.0)
        self.wall_min_updates = getattr(config, 'WALL_MIN_UPDATES', 3)
        
        # Wall tracking state: {price: {'first_seen': time, 'updates': count, 'last_size': size}}
        self.bid_wall_tracker: Dict[float, dict] = {}
        self.ask_wall_tracker: Dict[float, dict] = {}
        self.wall_tracker_max_age = 30.0  # Forget walls older than 30s

    def analyze(
        self,
        bids: List[List[float]],  # [[price, size], ...]
        asks: List[List[float]],
        mid_price: float,
        volatility: float = 0.0
    ) -> OrderbookAnalysis:
        """
        Perform full orderbook analysis.
        
        Args:
            bids: Bid levels [[price, size], ...]
            asks: Ask levels [[price, size], ...]
            mid_price: Current mid price
            volatility: Recent volatility for spread adjustment
            
        Returns:
            OrderbookAnalysis with all metrics
        """
        if not bids or not asks:
            return self._empty_analysis(mid_price)
        
        # Process levels
        bid_levels = self._process_levels(bids, mid_price, is_bid=True)
        ask_levels = self._process_levels(asks, mid_price, is_bid=False)
        
        # Current spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        current_spread = best_ask - best_bid
        current_spread_bps = (current_spread / mid_price) * 10000
        
        # Imbalance calculations
        top_imbalance = self._calc_top_imbalance(bids, asks)
        depth_imbalance = self._calc_depth_imbalance(bids, asks)
        cumulative_imbalance = self._calc_cumulative_imbalance(bid_levels, ask_levels)
        
        # Find walls (with flicker protection if enabled)
        # Use validated walls that have persisted, falling back to raw detection
        bid_wall = self.find_validated_wall(bid_levels, is_bid=True)
        if bid_wall is None:
            bid_wall = self._find_wall(bid_levels)  # Fallback for first detection
        ask_wall = self.find_validated_wall(ask_levels, is_bid=False)
        if ask_wall is None:
            ask_wall = self._find_wall(ask_levels)  # Fallback for first detection
        
        # Calculate micro-price (volume-weighted fair value)
        micro_price = self._calc_micro_price(bids, asks)
        
        # VWAP pressure across depth
        vwap_pressure = self._calc_vwap_pressure(bids, asks, mid_price)
        
        # Calculate adaptive spread
        adaptive_spread = self._calc_adaptive_spread(
            current_spread_bps, 
            volatility,
            depth_imbalance,
            bid_levels,
            ask_levels
        )
        
        # Find optimal order placement levels
        optimal_bid, optimal_ask = self._find_optimal_levels(
            bids, asks, mid_price, micro_price, 
            bid_wall, ask_wall, adaptive_spread
        )
        
        # Trade flow analysis
        trade_flow = self._analyze_trade_flow()
        
        # Liquidity score
        liquidity = self._calc_liquidity_score(bid_levels, ask_levels, current_spread_bps)
        
        # Stability check
        is_stable = self._check_stability(
            current_spread_bps, volatility, depth_imbalance, liquidity
        )
        
        return OrderbookAnalysis(
            current_spread_bps=current_spread_bps,
            adaptive_spread_bps=adaptive_spread,
            top_imbalance=top_imbalance,
            depth_imbalance=depth_imbalance,
            cumulative_imbalance=cumulative_imbalance,
            bid_wall_price=bid_wall[0] if bid_wall else None,
            bid_wall_size=bid_wall[1] if bid_wall else 0,
            ask_wall_price=ask_wall[0] if ask_wall else None,
            ask_wall_size=ask_wall[1] if ask_wall else 0,
            optimal_bid_price=optimal_bid,
            optimal_ask_price=optimal_ask,
            micro_price=micro_price,
            vwap_pressure=vwap_pressure,
            recent_buy_volume=trade_flow['buy_vol'],
            recent_sell_volume=trade_flow['sell_vol'],
            trade_flow_bias=trade_flow['bias'],
            liquidity_score=liquidity,
            is_stable=is_stable
        )

    def _process_levels(
        self, 
        levels: List[List[float]], 
        mid_price: float,
        is_bid: bool
    ) -> List[OrderbookLevel]:
        """Process raw levels into analyzed OrderbookLevel objects."""
        if not levels:
            return []
            
        processed = []
        cumulative = 0.0
        sizes = [l[1] for l in levels[:self.depth_levels]]
        avg_size = np.mean(sizes) if sizes else 0
        
        for price, size in levels[:self.depth_levels]:
            cumulative += size
            distance = abs(price - mid_price) / mid_price
            is_wall = size > (avg_size * self.wall_threshold) if avg_size > 0 else False
            
            processed.append(OrderbookLevel(
                price=price,
                size=size,
                cumulative_size=cumulative,
                distance_from_mid=distance,
                is_wall=is_wall
            ))
            
        return processed

    def _calc_top_imbalance(
        self, 
        bids: List[List[float]], 
        asks: List[List[float]]
    ) -> float:
        """Calculate imbalance at top of book only."""
        if not bids or not asks:
            return 0.5
        bid_vol = bids[0][1]
        ask_vol = asks[0][1]
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5

    def _calc_depth_imbalance(
        self, 
        bids: List[List[float]], 
        asks: List[List[float]],
        levels: int = 5
    ) -> float:
        """
        Calculate weighted imbalance across depth.
        Closer levels weighted more heavily.
        """
        bid_weighted = 0.0
        ask_weighted = 0.0
        
        
        for i, (bid, ask) in enumerate(zip(bids[:levels], asks[:levels])):
            weight = 1.0 / pow(i + 1, self.oimb_decay)  # Geometric decay
            bid_weighted += bid[1] * weight
            ask_weighted += ask[1] * weight
            
        total = bid_weighted + ask_weighted
        return bid_weighted / total if total > 0 else 0.5

    def _calc_cumulative_imbalance(
        self,
        bid_levels: List[OrderbookLevel],
        ask_levels: List[OrderbookLevel]
    ) -> float:
        """Calculate imbalance using cumulative volume."""
        if not bid_levels or not ask_levels:
            return 0.5
        bid_total = bid_levels[-1].cumulative_size if bid_levels else 0
        ask_total = ask_levels[-1].cumulative_size if ask_levels else 0
        total = bid_total + ask_total
        return bid_total / total if total > 0 else 0.5

    def _find_wall(
        self, 
        levels: List[OrderbookLevel]
    ) -> Optional[Tuple[float, float]]:
        """Find the first significant wall in the levels."""
        for level in levels:
            if level.is_wall:
                return (level.price, level.size)
        return None

    def _is_wall_persistent(
        self,
        price: float,
        size: float,
        is_bid: bool
    ) -> bool:
        """
        Check if a wall has persisted long enough to be considered valid.
        This protects against spoofing - orders placed briefly to manipulate.
        
        A wall is valid if:
        1. It has been seen for at least wall_min_persistence_sec seconds
        2. It has been present in at least wall_min_updates orderbook snapshots
        3. Its size hasn't decreased dramatically (potential partial cancellation)
        """
        if not self.wall_flicker_enabled:
            return True  # Flicker protection disabled
            
        tracker = self.bid_wall_tracker if is_bid else self.ask_wall_tracker
        now = time.time()
        
        # Clean up old entries
        stale_prices = [p for p, data in tracker.items() 
                       if now - data['first_seen'] > self.wall_tracker_max_age]
        for p in stale_prices:
            del tracker[p]
        
        # Price tolerance for matching (0.01% to account for price granularity)
        price_tolerance = price * 0.0001
        matched_price = None
        for tracked_price in tracker:
            if abs(tracked_price - price) < price_tolerance:
                matched_price = tracked_price
                break
        
        if matched_price is None:
            # First time seeing this wall
            tracker[price] = {
                'first_seen': now,
                'updates': 1,
                'last_size': size,
                'max_size': size
            }
            return False  # Not persistent yet
        
        # Update existing wall tracking
        data = tracker[matched_price]
        data['updates'] += 1
        
        # Check for size reduction (potential partial cancel / spoof)
        if size < data['max_size'] * 0.5:
            # Wall reduced by more than 50% - might be spoofing
            logger.debug(f"Wall at {price:.4f} shrunk {data['max_size']:.2f} -> {size:.2f}, suspicious")
            # Don't reset completely, but reduce confidence
            data['updates'] = max(1, data['updates'] - 1)
        
        data['last_size'] = size
        data['max_size'] = max(data['max_size'], size)
        
        # Check persistence criteria
        age = now - data['first_seen']
        is_persistent = (
            age >= self.wall_min_persistence_sec and
            data['updates'] >= self.wall_min_updates
        )
        
        return is_persistent

    def find_validated_wall(
        self,
        levels: List[OrderbookLevel],
        is_bid: bool
    ) -> Optional[Tuple[float, float]]:
        """
        Find a wall that has passed flicker protection.
        Only returns walls that have persisted long enough.
        """
        for level in levels:
            if level.is_wall:
                if self._is_wall_persistent(level.price, level.size, is_bid):
                    return (level.price, level.size)
        return None

    def _calc_micro_price(
        self, 
        bids: List[List[float]], 
        asks: List[List[float]]
    ) -> float:
        """
        Calculate volume-weighted micro-price.
        This is the best estimate of fair value based on orderbook.
        """
        if not bids or not asks:
            return 0.0
            
        best_bid, bid_vol = bids[0]
        best_ask, ask_vol = asks[0]
        
        # Classic micro-price formula
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return (best_bid + best_ask) / 2
            
        return (best_bid * ask_vol + best_ask * bid_vol) / total_vol

    def _calc_vwap_pressure(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        mid_price: float
    ) -> float:
        """
        Calculate volume-weighted pressure direction.
        Returns -1 (strong sell) to +1 (strong buy).
        """
        if not bids or not asks:
            return 0.0
            
        # Weight volume by distance from mid (closer = more pressure)
        bid_pressure = 0.0
        ask_pressure = 0.0
        
        for price, size in bids[:self.depth_levels]:
            distance = abs(price - mid_price) / mid_price
            weight = 1.0 / (1.0 + distance * self.pressure_dist_factor)  # Closer = higher weight
            bid_pressure += size * weight
            
        for price, size in asks[:self.depth_levels]:
            distance = abs(price - mid_price) / mid_price
            weight = 1.0 / (1.0 + distance * self.pressure_dist_factor)
            ask_pressure += size * weight
            
        total = bid_pressure + ask_pressure
        if total == 0:
            return 0.0
            
        # Normalize to -1 to 1
        imbalance = bid_pressure / total  # 0 to 1
        return (imbalance - 0.5) * 2  # -1 to 1

    def _calc_adaptive_spread(
        self,
        current_spread_bps: float,
        volatility: float,
        depth_imbalance: float,
        bid_levels: List[OrderbookLevel],
        ask_levels: List[OrderbookLevel]
    ) -> float:
        """
        Calculate adaptive spread based on market conditions.
        
        Optimized for HFT (100+ trades/hour):
        - Start with tight base spread
        - Only widen significantly on danger signals
        - Aim to be competitive with top of book
        """
        # Start with tight base spread for HFT
        spread = self.base_spread_bps
        
        # 1. VOLATILITY ADJUSTMENT (Smart Scaling)
        # Low Vol (<0.001): Minimal adjustment to maximize fills
        # High Vol (>0.005): Massive adjustment to protect capital (Ret/DD optimization)
        # Use simple scaler now that we have config control
        vol_adjustment = volatility * self.vol_adjust_scaler
        
        if len(ask_levels) < 3: # Panic override if book empty
             vol_adjustment *= 2
            
        spread += vol_adjustment
        
        # 2. IMBALANCE ADJUSTMENT (only on extreme imbalance)
        imbalance_skew = abs(depth_imbalance - 0.5)
        if imbalance_skew > 0.4:  # Only react to very strong imbalance
            spread += imbalance_skew * 5  # Reduced from 20 to avoid over-widening
        
        # 3. LIQUIDITY ADJUSTMENT (minimal for HFT)
        # We assume trading liquid coins
        bid_liq = sum(l.size for l in bid_levels[:3]) if bid_levels else 0
        ask_liq = sum(l.size for l in ask_levels[:3]) if ask_levels else 0
        avg_liq = (bid_liq + ask_liq) / 2
        
        # Only widen on very thin books
        if avg_liq < 50:
            spread += (50 - avg_liq) / 10  # Up to +5 bps
        
        # 4. COMPETITIVE SPREAD - key for HFT
        # We want to be AT or NEAR the current spread to get filled
        # If market spread is tighter than our target, match it
        if current_spread_bps < spread and current_spread_bps > self.min_spread_bps:
            spread = current_spread_bps * 1.1  # Just slightly wider than market
        
        # 5. CLAMP TO BOUNDS
        spread = max(self.min_spread_bps, min(self.max_spread_bps, spread))
        
        return spread

    def _find_optimal_levels(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        mid_price: float,
        micro_price: float,
        bid_wall: Optional[Tuple[float, float]],
        ask_wall: Optional[Tuple[float, float]],
        spread_bps: float
    ) -> Tuple[float, float]:
        """
        Find optimal price levels for order placement.
        
        Instead of fixed grid, we place orders strategically:
        - In front of walls (if they exist)
        - At micro-price adjusted levels
        - Respecting our adaptive spread
        """
        half_spread = (spread_bps / 10000) / 2  # Convert bps to decimal, then half
        
        # Start with micro-price as fair value (it has predictive power)
        fair_value = micro_price
        
        # Calculate initial levels
        optimal_bid = fair_value * (1 - half_spread)
        optimal_ask = fair_value * (1 + half_spread)
        
        # Adjust for walls
        if bid_wall:
            wall_price = bid_wall[0]
            # If there's a strong bid wall, place our bid just in front of it
            if wall_price < optimal_bid and wall_price > mid_price * 0.99:
                # Place just above the wall to get priority
                optimal_bid = wall_price + (mid_price * 0.0001)
                
        if ask_wall:
            wall_price = ask_wall[0]
            # If there's a strong ask wall, place our ask just in front of it
            if wall_price > optimal_ask and wall_price < mid_price * 1.01:
                optimal_ask = wall_price - (mid_price * 0.0001)
        
        # Ensure we don't cross the book
        best_bid = bids[0][0] if bids else mid_price * 0.999
        best_ask = asks[0][0] if asks else mid_price * 1.001
        
        optimal_bid = min(optimal_bid, best_ask - (mid_price * 0.0001))
        optimal_ask = max(optimal_ask, best_bid + (mid_price * 0.0001))
        
        return optimal_bid, optimal_ask

    def record_trade(
        self, 
        side: str, 
        price: float, 
        size: float,
        timestamp: float = None
    ):
        """Record a trade for flow analysis."""
        if timestamp is None:
            timestamp = time.time()
        self.trade_history.append({
            'side': side,
            'price': price,
            'size': size,
            'time': timestamp
        })

    def _analyze_trade_flow(self) -> Dict:
        """Analyze recent trade flow for directional bias."""
        now = time.time()
        cutoff = now - 60  # Last 60 seconds
        
        recent = [t for t in self.trade_history if t['time'] > cutoff]
        
        buy_vol = sum(t['size'] for t in recent if t['side'] == 'BUY')
        sell_vol = sum(t['size'] for t in recent if t['side'] == 'SELL')
        
        total = buy_vol + sell_vol
        if total == 0:
            bias = 0.0
        else:
            bias = (buy_vol - sell_vol) / total  # -1 to 1
            
        return {
            'buy_vol': buy_vol,
            'sell_vol': sell_vol,
            'bias': bias
        }

    def _calc_liquidity_score(
        self,
        bid_levels: List[OrderbookLevel],
        ask_levels: List[OrderbookLevel],
        spread_bps: float
    ) -> float:
        """
        Calculate a liquidity score from 0 to 1.
        Higher = more liquid, easier to trade.
        """
        if not bid_levels or not ask_levels:
            return 0.0
            
        # Total depth
        bid_depth = bid_levels[-1].cumulative_size if bid_levels else 0
        ask_depth = ask_levels[-1].cumulative_size if ask_levels else 0
        total_depth = bid_depth + ask_depth
        
        # Normalize depth score (adjust based on your market)
        depth_score = min(1.0, total_depth / self.liq_depth_divisor)  # Reduced for HFT on smaller coins
        
        # Spread score (tighter = better) - for HFT, we're more tolerant
        spread_score = max(0, 1 - (spread_bps / 30))  # 30 bps = 0 score (was 50)
        
        # Combine
        return (depth_score * 0.6) + (spread_score * 0.4)

    def _check_stability(
        self,
        spread_bps: float,
        volatility: float,
        imbalance: float,
        liquidity: float
    ) -> bool:
        """
        Check if market is stable enough for market making.
        
        Keep this conservative: only block on clear data/market quality failures.
        ToxicFlowGuard and RiskManager handle most risk, but if the book is clearly
        unhealthy (spread blowout / near-zero liquidity), quoting is usually worse
        than doing nothing.
        """

        # 1) Spread blowout: if spread is far beyond our configured max, the book is unstable.
        # Use a loose multiplier so we don't over-block.
        max_bps = float(getattr(self, 'max_spread_bps', 20.0) or 20.0)
        if spread_bps >= max(50.0, max_bps * 2.5):
            return False

        # 2) Liquidity floor: if our computed liquidity score is near-zero, avoid quoting.
        # (This is already normalized 0..1.)
        if liquidity <= 0.05:
            return False

        # 3) Extreme short-term volatility: if std of returns is very high, the book is unstable.
        # Use ToxicFlowGuard MAX_VOL_SHORT as a sanity anchor when available.
        max_vol_short = float(getattr(config, 'MAX_SHORT_TERM_VOLATILITY', 0.0) or 0.0)
        if max_vol_short > 0 and volatility > (max_vol_short * 2.0):
            return False

        return True

    def _empty_analysis(self, mid_price: float) -> OrderbookAnalysis:
        """Return empty analysis when no data."""
        return OrderbookAnalysis(
            current_spread_bps=0,
            adaptive_spread_bps=self.base_spread_bps,
            top_imbalance=0.5,
            depth_imbalance=0.5,
            cumulative_imbalance=0.5,
            bid_wall_price=None,
            bid_wall_size=0,
            ask_wall_price=None,
            ask_wall_size=0,
            optimal_bid_price=mid_price * 0.999,
            optimal_ask_price=mid_price * 1.001,
            micro_price=mid_price,
            vwap_pressure=0,
            recent_buy_volume=0,
            recent_sell_volume=0,
            trade_flow_bias=0,
            liquidity_score=0,
            is_stable=False
        )

    def get_summary(self, analysis: OrderbookAnalysis) -> str:
        """Get human-readable summary of analysis."""
        direction = "NEUTRAL"
        if analysis.vwap_pressure > 0.3:
            direction = "BULLISH"
        elif analysis.vwap_pressure < -0.3:
            direction = "BEARISH"
            
        return (
            f" Spread: {analysis.current_spread_bps:.1f}bps (adapt: {analysis.adaptive_spread_bps:.1f}bps) | "
            f"Imbalance: {analysis.depth_imbalance:.1%} | "
            f"Pressure: {direction} ({analysis.vwap_pressure:+.2f}) | "
            f"Liquidity: {analysis.liquidity_score:.1%}"
        )
