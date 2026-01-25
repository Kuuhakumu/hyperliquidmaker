# strategy_engine.py - figures out where to place orders based on the orderbook
# not grid trading, we look at the actual book to decide prices

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import time
import numpy as np

import config
from core.regime_engine import TradingRegime
from core.orderbook_analyzer import OrderbookAnalyzer, OrderbookAnalysis

logger = logging.getLogger("StrategyEngine")


@dataclass
class LatencyMetrics:
    """
    Track latency at each stage of the trading pipeline .
    
    CRITICAL for HFT: The strategy relies on sniping best levels and reacting
    to micro-structure changes. Python introduces latency that can lead to
    adverse selection (getting picked off).
    
    Recommendations:
    - Server should be geographically close to exchange (AWS Tokyo for most crypto)
    - Monitor P99 latencies - spikes indicate risk of adverse selection
    - tick_to_trade_ms > 50ms is dangerous in volatile markets
    """
    # Rolling window of latency measurements
    ws_to_process_ms: List[float] = field(default_factory=list)  # WS message to processed
    strategy_calc_ms: List[float] = field(default_factory=list)  # Time to calculate orders
    order_submit_ms: List[float] = field(default_factory=list)   # Time to submit to exchange
    tick_to_trade_ms: List[float] = field(default_factory=list)  # Full tick latency
    
    # Additional HFT-critical metrics
    orderbook_age_ms: List[float] = field(default_factory=list)  # Staleness of orderbook data
    cancel_latency_ms: List[float] = field(default_factory=list)  # Time to cancel orders
    fill_notification_ms: List[float] = field(default_factory=list)  # Time to receive fill
    
    # Latency warning thresholds (from config)
    warn_tick_to_trade_ms: float = field(default_factory=lambda: getattr(config, 'LATENCY_WARN_TICK_TO_TRADE_MS', 50.0))
    warn_order_submit_ms: float = field(default_factory=lambda: getattr(config, 'LATENCY_WARN_ORDER_SUBMIT_MS', 30.0))
    
    # Adverse selection tracking
    adverse_selection_count: int = 0  # Orders filled at worse price due to latency
    last_adverse_time: float = 0.0
    
    def record(self, stage: str, latency_ms: float):
        """Record a latency measurement."""
        target = getattr(self, stage, None)
        if target is not None:
            target.append(latency_ms)
            # Keep last 100 measurements
            if len(target) > 100:
                target.pop(0)
            
            # Check for latency warnings
            if stage == 'tick_to_trade_ms' and latency_ms > self.warn_tick_to_trade_ms:
                logger.warning(f" High tick-to-trade latency: {latency_ms:.1f}ms (threshold: {self.warn_tick_to_trade_ms}ms)")
            elif stage == 'order_submit_ms' and latency_ms > self.warn_order_submit_ms:
                logger.warning(f" High order submit latency: {latency_ms:.1f}ms")
    
    def record_adverse_selection(self, expected_price: float, actual_price: float, side: str):
        """Track when we get adversely selected (filled at worse price)."""
        self.adverse_selection_count += 1
        self.last_adverse_time = time.time()
        
        slippage_bps = abs(actual_price - expected_price) / expected_price * 10000
        logger.warning(
            f" ADVERSE SELECTION #{self.adverse_selection_count}: "
            f"{side} expected {expected_price:.4f}, got {actual_price:.4f} "
            f"({slippage_bps:.1f} bps slippage)"
        )
    
    def get_avg(self, stage: str) -> float:
        """Get average latency for a stage."""
        target = getattr(self, stage, [])
        return sum(target) / len(target) if target else 0.0
    
    def get_p99(self, stage: str) -> float:
        """Get 99th percentile latency for a stage."""
        target = getattr(self, stage, [])
        if not target:
            return 0.0
        sorted_vals = sorted(target)
        idx = int(len(sorted_vals) * 0.99)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    def get_p50(self, stage: str) -> float:
        """Get median (50th percentile) latency for a stage."""
        target = getattr(self, stage, [])
        if not target:
            return 0.0
        sorted_vals = sorted(target)
        idx = len(sorted_vals) // 2
        return sorted_vals[idx]
    
    def get_summary(self) -> dict:
        """Get full latency summary."""
        return {
            'ws_to_process_avg_ms': self.get_avg('ws_to_process_ms'),
            'ws_to_process_p99_ms': self.get_p99('ws_to_process_ms'),
            'strategy_calc_avg_ms': self.get_avg('strategy_calc_ms'),
            'strategy_calc_p99_ms': self.get_p99('strategy_calc_ms'),
            'order_submit_avg_ms': self.get_avg('order_submit_ms'),
            'order_submit_p99_ms': self.get_p99('order_submit_ms'),
            'tick_to_trade_avg_ms': self.get_avg('tick_to_trade_ms'),
            'tick_to_trade_p99_ms': self.get_p99('tick_to_trade_ms'),
            'cancel_latency_avg_ms': self.get_avg('cancel_latency_ms'),
            'orderbook_age_avg_ms': self.get_avg('orderbook_age_ms'),
            'adverse_selection_count': self.adverse_selection_count,
        }
    
    def get_health_status(self) -> Tuple[str, str]:
        """
        Get overall latency health status.
        
        Returns:
            Tuple of (status_emoji, description)
        """
        p99_tick = self.get_p99('tick_to_trade_ms')
        avg_tick = self.get_avg('tick_to_trade_ms')
        
        if p99_tick == 0:
            return "", "No data yet"
        elif p99_tick < 30:
            return "", f"Excellent ({avg_tick:.0f}ms avg, {p99_tick:.0f}ms p99)"
        elif p99_tick < 50:
            return "", f"Good ({avg_tick:.0f}ms avg, {p99_tick:.0f}ms p99)"
        elif p99_tick < 100:
            return "", f"Moderate ({avg_tick:.0f}ms avg, {p99_tick:.0f}ms p99) - consider geo-proximity"
        else:
            return "", f"High latency ({avg_tick:.0f}ms avg, {p99_tick:.0f}ms p99) - adverse selection risk!"


@dataclass
class OrderLevel:
    """Represents a single order level."""
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    is_aggressive: bool = False
    reduce_only: bool = False  # True for flush/TP orders to prevent position flip
    reason: str = ""  # Why this level was chosen


@dataclass
class StrategyDecision:
    """The output of the strategy engine."""
    buy_orders: List[OrderLevel]
    sell_orders: List[OrderLevel]
    should_cancel_all: bool = False
    should_flush_inventory: bool = False
    reason: str = ""
    analysis_summary: str = ""  # Orderbook analysis info


class StrategyEngine:
    """
    Orderbook-driven strategy engine.
    
    FARMER Mode (Market Making):
    - NOT a grid. We place 1-2 orders per side at OPTIMAL levels.
    - Levels determined by: micro-price, walls, adaptive spread
    - Quotes skewed based on inventory and orderbook pressure
    
    HUNTER Mode (Trend Following):
    - Aggressive orders in trend direction
    - Uses orderbook imbalance + trade flow for conviction
    """

    def __init__(self, ml_engine=None):
        # Initialize orderbook analyzer
        self.ob_analyzer = OrderbookAnalyzer()
        self.ml_engine = ml_engine
        
        # Config
        self.order_size_usd = config.ORDER_SIZE_USD
        
        # Skew parameters
        self.inventory_skew_factor = config.SKEW_FACTOR
        self.max_skew_bps = 20.0  # Maximum skew in basis points
        
        # Convex skew parameters
        self.use_convex_skew = getattr(config, 'USE_CONVEX_SKEW', True)
        self.convex_max_skew_bps = getattr(config, 'CONVEX_MAX_SKEW_BPS', 25.0)
        
        # Funding rate parameters
        self.funding_bias_enabled = getattr(config, 'FUNDING_BIAS_ENABLED', True)
        self.funding_threshold = getattr(config, 'FUNDING_THRESHOLD', 0.0001)
        self.funding_skew_mult = getattr(config, 'FUNDING_SKEW_MULTIPLIER', 10.0)
        
        # Time-of-day parameters
        self.time_adjust_enabled = getattr(config, 'TIME_ADJUST_ENABLED', True)
        self.low_volume_hours = getattr(config, 'LOW_VOLUME_HOURS', [2, 3, 4, 5, 6, 7])
        self.high_volume_hours = getattr(config, 'HIGH_VOLUME_HOURS', [13, 14, 15, 16])
        self.low_vol_spread_mult = getattr(config, 'LOW_VOLUME_SPREAD_MULT', 1.3)
        self.high_vol_spread_mult = getattr(config, 'HIGH_VOLUME_SPREAD_MULT', 0.9)
        
        # Kelly sizing parameters
        self.use_kelly_sizing = getattr(config, 'USE_KELLY_SIZING', False)
        self.kelly_fraction = getattr(config, 'KELLY_FRACTION', 0.5)
        self.kelly_min_trades = getattr(config, 'KELLY_MIN_TRADES', 50)
        self.kelly_max_clip = getattr(config, 'KELLY_MAX_CLIP_PCT', 0.15)
        
        # Trade history for Kelly calculation
        self.trade_results: List[float] = []  # List of PnL percentages
        
        # Inventory decay parameters 
        self.use_inventory_decay = getattr(config, 'USE_INVENTORY_DECAY', True)
        self.inventory_half_life = getattr(config, 'INVENTORY_HALF_LIFE_SECONDS', 120.0)  # 2 minutes
        self.position_entry_time: Optional[float] = None
        self.last_position_usd: float = 0.0
        
        # Latency metrics tracking 
        self.latency_metrics = LatencyMetrics()
        
        # Spread adjustment from maker ratio 
        self.maker_ratio_spread_mult = 1.0
        
        # Tracking
        self.last_analysis: Optional[OrderbookAnalysis] = None
        self.last_funding_rate: float = 0.0
        self.tick_size = 0.0001  # Default tick size (snapped in round_order)

    def generate_orders(
        self,
        regime: TradingRegime,
        metadata: dict,
        mid_price: float,
        bids: List[List[float]],  # [[price, size], ...]
        asks: List[List[float]],
        current_position_usd: float,
        volatility: float,
        funding_rate: float = 0.0
    ) -> StrategyDecision:
        """
        Generate order levels based on orderbook analysis.
        
        Args:
            regime: Current trading regime
            metadata: Regime metadata including skew_factor
            mid_price: Current mid price
            bids: Full bid orderbook [[price, size], ...]
            asks: Full ask orderbook [[price, size], ...]
            current_position_usd: Current position in USD
            volatility: Current volatility metric
            funding_rate: Current funding rate (positive = longs pay shorts)
            
        Returns:
            StrategyDecision with orders at optimal levels
        """
        # Store funding rate for use in strategy
        self.last_funding_rate = funding_rate
        
        # Handle HALTED state
        if regime == TradingRegime.HALTED:
            return StrategyDecision(
                buy_orders=[],
                sell_orders=[],
                should_cancel_all=True,
                reason="Regime HALTED - cancelling all orders"
            )
        
        # Perform orderbook analysis
        analysis = self.ob_analyzer.analyze(bids, asks, mid_price, volatility)
        self.last_analysis = analysis
        
        # ML Prediction (if enabled)
        if self.ml_engine:
            from types import SimpleNamespace
            dummy_md = SimpleNamespace(
                bids=bids, asks=asks, mid_price=mid_price,
                best_bid=bids[0][0] if bids else mid_price,
                best_ask=asks[0][0] if asks else mid_price,
                volatility=volatility
            )
            ml_signal, ml_conf = self.ml_engine.predict(dummy_md, analysis)
            if ml_conf > 0.0:
                 self.last_ml_signal = ml_signal
        
        # Check if market is stable enough to trade
        if not analysis.is_stable:
            return StrategyDecision(
                buy_orders=[],
                sell_orders=[],
                should_cancel_all=True,
                reason=f"Market unstable - spread:{analysis.current_spread_bps:.1f}bps, liq:{analysis.liquidity_score:.1%}",
                analysis_summary=self.ob_analyzer.get_summary(analysis)
            )
        
        # Route to appropriate strategy
        if regime in [TradingRegime.HUNTER_LONG, TradingRegime.HUNTER_SHORT]:
            return self._hunter_strategy(regime, analysis, mid_price, bids, asks, current_position_usd)
        else:
            return self._farmer_strategy(
                regime, metadata, analysis, mid_price, 
                bids, asks, current_position_usd
            )

    def calculate_position_size(self, account_value: float, volatility: float) -> float:
        """Scale position size inversely with volatility."""
        base_size = account_value * config.COMPOUND_CLIP_SIZE
        # Target 0.2% volatility impact
        # If vol is 0.2%, scalar is 1.0
        # If vol is 1.0%, scalar is 0.2 (smaller size)
        # If vol is 0.1%, scalar is 1.5 (larger size, capped)
        vol_scalar = max(0.3, min(1.5, 0.002 / (volatility + 0.00001)))
        return base_size * vol_scalar

    def _farmer_strategy(
        self,
        regime: TradingRegime,
        metadata: dict,
        analysis: OrderbookAnalysis,
        mid_price: float,
        bids: List[List[float]],
        asks: List[List[float]],
        current_position_usd: float
    ) -> StrategyDecision:
        """
        FARMER (Market Making) Strategy - Orderbook Driven.
        
        NOT a grid. We:
        1. Use adaptive spread from orderbook analysis
        2. Place orders at optimal levels (near walls if they exist)
        3. Skew quotes based on inventory AND orderbook pressure
        4. Apply funding rate bias
        5. Adjust spread based on time of day
        6. Use Kelly sizing if enabled
        7. Only 1 order per side (snipe the best level)
        """
        buy_orders = []
        sell_orders = []
        
        # Get optimal levels from analysis with time-of-day adjustment
        time_mult = self._get_time_of_day_multiplier(analysis)
        
        # Adjust spread based on time of day
        adjusted_spread_bps = analysis.adaptive_spread_bps * time_mult
        spread_adjustment = (adjusted_spread_bps - analysis.adaptive_spread_bps) / 10000 * mid_price
        
        optimal_bid = analysis.optimal_bid_price - (spread_adjustment / 2)
        optimal_ask = analysis.optimal_ask_price + (spread_adjustment / 2)
        
        # Calculate order size
        account_value = metadata.get('account_value', 1000.0)
        volatility = metadata.get('volatility', 0.005)
        
        # Try Kelly sizing first, fall back to compound sizing
        kelly_size = self._calculate_kelly_size(account_value)
        if kelly_size is not None and kelly_size > 20:
            dynamic_size_usd = kelly_size
            sizing_method = "Kelly"
        else:
            # Use volatility-adjusted sizing
            dynamic_size_usd = self.calculate_position_size(account_value, volatility)
            sizing_method = "VolAdjusted"
        
        # Clamp to reasonable limits
        dynamic_size_usd = max(20.0, min(2000.0, dynamic_size_usd))
        
        # Use simple moving average of size to prevent jitter? No, instant compounding is better.
        # Fallback to config if dynamic fails
        order_size_usd = dynamic_size_usd if dynamic_size_usd > 10 else self.order_size_usd
        
        order_size = order_size_usd / mid_price
        
        # Calculate inventory skew (now supports convex mode)
        inventory_skew = self._calculate_inventory_skew(current_position_usd, mid_price)
        
        # Calculate funding rate skew
        funding_skew = self._calculate_funding_skew(self.last_funding_rate)
        
        # Combine with orderbook pressure for final skew
        # If orderbook is bullish AND we're long, reduce bid more
        # If orderbook is bullish AND we're short, be more aggressive on bids
        pressure_adjustment = analysis.vwap_pressure * config.VWAP_PRESSURE_SCALER
        
        # Apply combined skew (inventory + pressure + funding)
        final_skew = inventory_skew + pressure_adjustment + funding_skew
        final_skew = max(-0.003, min(0.003, final_skew))  # Clamp to 0.3%
        
        # Skewed levels - shift CENTER by skew (Avellaneda-Stoikov style)
        # Positive skew (long inventory) -> shift BOTH quotes DOWN to encourage selling
        # This keeps spread constant while moving the center
        skew_offset = mid_price * final_skew
        skewed_bid = optimal_bid - skew_offset
        skewed_ask = optimal_ask - skew_offset
        
        # HIGH-LATENCY SAFETY BUFFER
        # With 2500ms latency, price can move during order submission
        # Push orders further from mid to avoid POST_ONLY_WOULD_TAKE rejections
        latency_buffer = mid_price * (getattr(config, 'LATENCY_SAFETY_BPS', 0.0) / 10000.0)
        skewed_bid -= latency_buffer  # Push bid down
        skewed_ask += latency_buffer  # Push ask up
        
        # Ensure we don't cross the book
        best_bid = bids[0][0] if bids else mid_price * 0.999
        best_ask = asks[0][0] if asks else mid_price * 1.001
        
        skewed_bid = min(skewed_bid, best_ask - (mid_price * 0.00005))
        skewed_ask = max(skewed_ask, best_bid + (mid_price * 0.00005))
        
        # Size adjustment based on regime
        bid_size = order_size
        ask_size = order_size
        
        if regime == TradingRegime.FARMER_SKEW_LONG:
            # We're long - reduce bids, increase asks
            bid_size *= 0.5
            ask_size *= 1.5
        elif regime == TradingRegime.FARMER_SKEW_SHORT:
            # We're short - increase bids, reduce asks
            bid_size *= 1.5
            ask_size *= 0.5
        
        # Additional pressure-based sizing
        # If orderbook heavily bullish, be more careful on asks
        if analysis.vwap_pressure > 0.3:
            ask_size *= 0.8  # Reduce ask size in bullish flow
        elif analysis.vwap_pressure < -0.3:
            bid_size *= 0.8  # Reduce bid size in bearish flow
        
        # Create orders with reasons
        bid_reason = f"Optimal bid (spread:{analysis.adaptive_spread_bps:.1f}bps"
        if analysis.bid_wall_price:
            bid_reason += f", wall@{analysis.bid_wall_price:.4f}"
        bid_reason += ")"
        
        ask_reason = f"Optimal ask (spread:{analysis.adaptive_spread_bps:.1f}bps"
        if analysis.ask_wall_price:
            ask_reason += f", wall@{analysis.ask_wall_price:.4f}"
        ask_reason += ")"
        
        buy_orders.append(OrderLevel(
            price=skewed_bid,
            size=bid_size,
            side="BUY",
            is_aggressive=False,
            reason=bid_reason
        ))
        
        sell_orders.append(OrderLevel(
            price=skewed_ask,
            size=ask_size,
            side="SELL",
            is_aggressive=False,
            reason=ask_reason
        ))
        
        # OPTIONAL: Add a second level if there's a wall to lean on
        if analysis.bid_wall_price and analysis.bid_wall_size > order_size * 3:
            # There's a big wall - place order just above it
            wall_bid = analysis.bid_wall_price + (mid_price * 0.00005)
            if wall_bid < skewed_bid * 0.998:  # Only if meaningfully different
                buy_orders.append(OrderLevel(
                    price=wall_bid,
                    size=order_size * 0.5,
                    side="BUY",
                    is_aggressive=False,
                    reason=f"Lean on wall ({analysis.bid_wall_size:.1f})"
                ))
        
        if analysis.ask_wall_price and analysis.ask_wall_size > order_size * 3:
            wall_ask = analysis.ask_wall_price - (mid_price * 0.00005)
            if wall_ask > skewed_ask * 1.002:
                sell_orders.append(OrderLevel(
                    price=wall_ask,
                    size=order_size * 0.5,
                    side="SELL",
                    is_aggressive=False,
                    reason=f"Lean on wall ({analysis.ask_wall_size:.1f})"
                ))
        
        # Build reason with all skew components
        skew_components = []
        if abs(inventory_skew) > 0.00001:
            skew_components.append(f"inv={inventory_skew*10000:.1f}")
        if abs(funding_skew) > 0.00001:
            skew_components.append(f"fund={funding_skew*10000:.1f}")
        if abs(pressure_adjustment) > 0.00001:
            skew_components.append(f"press={pressure_adjustment*10000:.1f}")
        
        skew_detail = "+".join(skew_components) if skew_components else "0"
        time_info = f", time={time_mult:.2f}x" if time_mult != 1.0 else ""
        sizing_info = f", {sizing_method}" if sizing_method == "Kelly" else ""
        
        return StrategyDecision(
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            reason=f"FARMER: spread={adjusted_spread_bps:.1f}bps, skew={final_skew*10000:.1f}bps ({skew_detail}){time_info}{sizing_info}",
            analysis_summary=self.ob_analyzer.get_summary(analysis)
        )

    def _hunter_strategy(
        self,
        regime: TradingRegime,
        analysis: OrderbookAnalysis,
        mid_price: float,
        bids: List[List[float]],
        asks: List[List[float]],
        current_position_usd: float
    ) -> StrategyDecision:
        """
        HUNTER (Trend Following) Strategy.
        
        Uses orderbook analysis for conviction:
        - Strong imbalance = more aggressive
        - Trade flow confirms = larger size
        """
        buy_orders = []
        sell_orders = []
        
        order_size = self.order_size_usd / mid_price
        
        # Calculate conviction based on orderbook
        imbalance_conviction = abs(analysis.depth_imbalance - 0.5) * 2  # 0 to 1
        flow_conviction = abs(analysis.trade_flow_bias)  # 0 to 1
        pressure_conviction = abs(analysis.vwap_pressure)  # 0 to 1
        
        # New: ML Signal Conviction
        ml_conviction = 0.0
        if getattr(self, 'last_ml_signal', 0) != 0:
            if regime == TradingRegime.HUNTER_LONG and self.last_ml_signal > 0.5:
                ml_conviction = 1.0 # High confidence boost
            elif regime == TradingRegime.HUNTER_SHORT and self.last_ml_signal < -0.5:
                ml_conviction = 1.0
        
        # Combined conviction (0 to 1)
        conviction = (imbalance_conviction + flow_conviction + pressure_conviction + ml_conviction) / 4
        
        # MARKET SHOCK ADJUSTMENT:
        # If imbalance is extreme (> 0.9 or < 0.1), boost conviction regardless of other metrics.
        # This catches "news crashes" where trade flow hasn't caught up but the book reflects panic.
        if imbalance_conviction > 0.8:
            conviction = max(conviction, 0.7) # Force above the 0.6 threshold
            
        if ml_conviction > 0:
            conviction = max(conviction, 0.8) # ML Overrides to high conviction
        
        # Size based on conviction
        hunt_size = order_size * (1 + conviction)  # 1x to 2x normal size
        
        best_bid = bids[0][0] if bids else mid_price * 0.999
        best_ask = asks[0][0] if asks else mid_price * 1.01
        
        # HIGH-LATENCY AGGRESSION for Hunter mode
        # In Hunter mode, we WANT to cross the spread or land at the very top.
        # We ADD the buffer to ensure we hit the price even if it moves away.
        latency_buffer = mid_price * (getattr(config, 'LATENCY_SAFETY_BPS', 0.0) / 10000.0)
        
        if regime == TradingRegime.HUNTER_LONG:
            # Bullish - aggressive bid
            base_agg = getattr(config, 'HUNTER_BASE_AGGRESSION', 0.0001)
            max_agg = getattr(config, 'HUNTER_MAX_AGGRESSION_BPS', 5.0) / 10000.0
            dynamic_range = max_agg - base_agg
            aggression = base_agg + (conviction * dynamic_range)
            
            # Conviction threshold: don't hunt unless we are sure
            if conviction < 0.6:
                return StrategyDecision(buy_orders=[], sell_orders=[], reason="Low conviction")

            # PUSH BID HIGHER but try to stay Maker (Post-Only)
            # If we cross mid_price, we are likely a taker. 
            # We use the latency_buffer as the MAX we are willing to push.
            hunt_price = best_bid + (mid_price * (aggression / 2)) # Use half aggression for price
            hunt_price = min(hunt_price, best_ask - self.tick_size) # Don't cross
            
            # EXTRA: If we have a SHORT position, flush it now! (Trend reversal)
            if current_position_usd < -5.0:
                flush_price = hunt_price # Use the aggressive hunt price
                buy_orders.append(OrderLevel(
                    price=flush_price,
                    size=abs(current_position_usd / mid_price),
                    side="BUY",
                    is_aggressive=True,
                    reduce_only=True,
                    reason="FLUSH SHORT (Trend Flip)"
                ))

            buy_orders.append(OrderLevel(
                price=hunt_price,
                size=hunt_size,
                side="BUY",
                is_aggressive=not getattr(config, 'MAKER_ONLY', True),
                reason=f"HUNT LONG (conviction:{conviction:.1%})"
            ))
            
            # Take-profit above (Gate: Only if we have a LONG position or just placed a BUY)
            # For HFT, we can place the TP alongside the entry if we omit reduce_only
            # or if we are sure the entry will hit.
            if current_position_usd > 5.0: # If we already have a position
                tp_price = mid_price * 1.002 - latency_buffer  # 0.2% profit, pulled closer for fill
                sell_orders.append(OrderLevel(
                    price=tp_price,
                    size=abs(current_position_usd / mid_price),
                    side="SELL",
                    reduce_only=True,
                    reason="Take-profit (Existing)"
                ))
        
        elif regime == TradingRegime.HUNTER_SHORT:
            # Bearish - aggressive ask
            base_agg = getattr(config, 'HUNTER_BASE_AGGRESSION', 0.0001)
            max_agg = getattr(config, 'HUNTER_MAX_AGGRESSION_BPS', 5.0) / 10000.0
            dynamic_range = max_agg - base_agg
            aggression = base_agg + (conviction * dynamic_range)
            
            # Conviction threshold
            if conviction < 0.6:
                return StrategyDecision(buy_orders=[], sell_orders=[], reason="Low conviction")

            # PUSH ASK LOWER
            hunt_price = best_ask - (mid_price * (aggression / 2))
            hunt_price = max(hunt_price, best_bid + self.tick_size) # Don't cross
            
            # EXTRA: If we have a LONG position, flush it now! (Trend reversal)
            if current_position_usd > 5.0:
                flush_price = hunt_price
                sell_orders.append(OrderLevel(
                    price=flush_price,
                    size=abs(current_position_usd / mid_price),
                    side="SELL",
                    is_aggressive=True,
                    reduce_only=True,
                    reason="FLUSH LONG (Trend Flip)"
                ))

            if not getattr(config, 'MAKER_ONLY', True):
                hunt_price -= latency_buffer

            sell_orders.append(OrderLevel(
                price=hunt_price,
                size=hunt_size,
                side="SELL",
                is_aggressive=not getattr(config, 'MAKER_ONLY', True),
                reason=f"HUNT SHORT (conviction:{conviction:.1%})"
            ))

            # Take-profit below (Gate: Only if we have a SHORT position)
            if current_position_usd < -5.0:
                tp_price = mid_price * (1 - 0.002) + latency_buffer
                buy_orders.append(OrderLevel(
                    price=tp_price,
                    size=abs(current_position_usd / mid_price),
                    side="BUY",
                    reduce_only=True,
                    reason="Take-profit (Existing)"
                ))

        return StrategyDecision(
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            reason=f"HUNTER: conviction={conviction:.1%}, pressure={analysis.vwap_pressure:+.2f}",
            analysis_summary=self.ob_analyzer.get_summary(analysis)
        )

    def _calculate_inventory_skew(
        self, 
        current_position_usd: float,
        mid_price: float
    ) -> float:
        """
        Calculate price skew based on current inventory.
        
        If long: skew bids down (less aggressive buying)
        If short: skew asks up (less aggressive selling)
        
        Supports both linear and convex (quadratic) skew modes.
        Convex skew penalizes large positions more aggressively.
        
        Now includes inventory decay : older positions
        are treated as smaller to encourage faster unwinding.
        
        Returns skew as decimal (e.g., 0.001 = 0.1%)
        """
        if abs(current_position_usd) < 10:  # Ignore tiny positions
            # Reset position tracking when flat
            self.position_entry_time = None
            self.last_position_usd = 0.0
            return 0.0
        
        # Track position entry time for decay calculation
        if self.position_entry_time is None or (
            np.sign(current_position_usd) != np.sign(self.last_position_usd)
        ):
            # New position or flipped sides
            self.position_entry_time = time.time()
        self.last_position_usd = current_position_usd
        
        # Apply inventory decay 
        effective_position = current_position_usd
        if self.use_inventory_decay and self.position_entry_time:
            time_held = time.time() - self.position_entry_time
            # Half-life decay: after half_life seconds, treat position as half its size
            decay_factor = 0.5 ** (time_held / self.inventory_half_life)
            # Invert decay for skew: older positions get MORE skew pressure to close
            # decay_factor goes 1.0 -> 0.5 -> 0.25 over time
            # We want skew to INCREASE, so use (2 - decay_factor) = 1.0 -> 1.5 -> 1.75
            urgency_factor = 2.0 - decay_factor
            effective_position = current_position_usd * urgency_factor
        
        max_position = getattr(config, 'MAX_POSITION_USD', 1000.0)
        
        if self.use_convex_skew:
            # Quadratic (convex) skew - penalizes large positions more
            # Formula: sign(inv) * (inv/max)^2 * max_skew
            normalized_inv = effective_position / max_position
            normalized_inv = max(-1.0, min(1.0, normalized_inv))  # Clamp to [-1, 1]
            
            # Quadratic penalty with sign preservation
            skew = np.sign(normalized_inv) * (normalized_inv ** 2) * (self.convex_max_skew_bps / 10000)
            
            return skew
        else:
            # Original linear skew
            # $100 position = 0.05% skew
            skew = (effective_position / 100) * self.inventory_skew_factor
            
            # Clamp
            max_skew = self.max_skew_bps / 10000
            return max(-max_skew, min(max_skew, skew))
    
    def update_maker_ratio_adjustment(self, should_widen: bool, multiplier: float):
        """
        Update spread multiplier based on maker ratio from OrderManager.
        
        Called by main loop when maker ratio falls below target.
        """
        if should_widen:
            self.maker_ratio_spread_mult = multiplier
            logger.debug(f" Widening spreads by {multiplier:.2f}x due to low maker ratio")
        else:
            self.maker_ratio_spread_mult = 1.0
    
    def get_latency_summary(self) -> dict:
        """Get latency metrics summary for monitoring."""
        return self.latency_metrics.get_summary()

    def _calculate_funding_skew(self, funding_rate: float) -> float:
        """
        Calculate additional skew based on funding rate.
        
        If funding > threshold (longs pay shorts):
            - Bias towards short to collect funding
            - Return positive skew (shift quotes down)
        If funding < -threshold (shorts pay longs):
            - Bias towards long to collect funding
            - Return negative skew (shift quotes up)
            
        Returns skew as decimal.
        """
        if not self.funding_bias_enabled:
            return 0.0
        
        if abs(funding_rate) < self.funding_threshold:
            return 0.0
        
        # Funding rate directly translates to skew
        # Positive funding = longs pay = we want to be short = positive skew
        skew = funding_rate * self.funding_skew_mult
        
        # Increase skew as we approach the hour (funding payment time)
        # Hyperliquid funding is paid hourly
        minutes_to_hour = 60 - datetime.now(timezone.utc).minute
        if minutes_to_hour < 15:
            # Ramp up skew in last 15 mins
            urgency = (15 - minutes_to_hour) / 15  # 0.0 to 1.0
            skew *= (1 + urgency)  # Up to 2x skew
        
        # Clamp to reasonable range (max 0.1% = 10bps)
        return max(-0.001, min(0.001, skew))

    def _get_time_of_day_multiplier(self, analysis: OrderbookAnalysis = None) -> float:
        """
        Get spread multiplier based on time of day AND volume.
        
        Low volume hours (e.g., Asia night): Widen spreads
        High volume hours (e.g., US open): Tighten spreads
        
        Returns multiplier (1.0 = no change).
        """
        multiplier = 1.0
        
        # 1. Time of Day Check
        if self.time_adjust_enabled:
            current_hour = datetime.now(timezone.utc).hour
            if current_hour in self.low_volume_hours:
                multiplier *= self.low_vol_spread_mult
            elif current_hour in self.high_volume_hours:
                multiplier *= self.high_vol_spread_mult
        
        # 2. Volume Check (Dynamic)
        if analysis and (analysis.recent_buy_volume + analysis.recent_sell_volume) > 0:
            # If we have volume data, use it to adjust spread
            # This is a relative check - needs baseline. 
            # For now, we use a simple heuristic: if volume is very low, widen spread
            # We can use liquidity_score as a proxy for volume/depth
            if analysis.liquidity_score < 0.3:
                multiplier *= 1.2  # Widen spread in illiquid markets
            elif analysis.liquidity_score > 0.8:
                multiplier *= 0.9  # Tighten spread in very liquid markets
                
        return multiplier

    def _calculate_kelly_size(self, account_value: float) -> Optional[float]:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly formula: f* = (p * b - q) / b
        Where:
            p = win probability
            b = avg win / avg loss ratio
            q = 1 - p = loss probability
            
        Returns position size in USD, or None if not enough data.
        """
        if not self.use_kelly_sizing:
            return None
        
        if len(self.trade_results) < self.kelly_min_trades:
            return None
        
        # Calculate win rate and ratios
        wins = [r for r in self.trade_results if r > 0]
        losses = [r for r in self.trade_results if r < 0]
        
        if not wins or not losses:
            return None
        
        win_rate = len(wins) / len(self.trade_results)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return None
        
        # Kelly formula
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        
        # Apply safety multiplier (half-Kelly)
        safe_fraction = kelly_fraction * self.kelly_fraction
        
        # Clamp to maximum
        safe_fraction = max(0, min(safe_fraction, self.kelly_max_clip))
        
        return account_value * safe_fraction

    def record_trade_result(self, pnl_pct: float):
        """Record a trade result for Kelly calculation."""
        self.trade_results.append(pnl_pct)
        # Keep last 500 trades
        if len(self.trade_results) > 500:
            self.trade_results = self.trade_results[-500:]

    def record_fill(self, side: str, price: float, size: float):
        """Record a fill for trade flow analysis."""
        self.ob_analyzer.record_trade(side, price, size)

    def get_flush_orders(
        self,
        current_position: float,
        mid_price: float,
        best_bid: float,
        best_ask: float
    ) -> StrategyDecision:
        """Generate orders to immediately close a position."""
        buy_orders = []
        sell_orders = []
        
        if abs(current_position) < 0.001:
            return StrategyDecision(
                buy_orders=[],
                sell_orders=[],
                reason="No position to flush"
            )
        
        if current_position > 0:
            flush_price = best_bid * 0.999
            sell_orders.append(OrderLevel(
                price=flush_price,
                size=abs(current_position),
                side="SELL",
                is_aggressive=True,
                reduce_only=True,
                reason="Emergency flush"
            ))
            reason = f"FLUSH: Selling {current_position:.4f} at {flush_price:.4f}"
        else:
            flush_price = best_ask * 1.001
            buy_orders.append(OrderLevel(
                price=flush_price,
                size=abs(current_position),
                side="BUY",
                is_aggressive=True,
                reason="Emergency flush"
            ))
            reason = f"FLUSH: Buying {abs(current_position):.4f} at {flush_price:.4f}"
        
        return StrategyDecision(
            buy_orders=buy_orders,
            sell_orders=sell_orders,
            should_cancel_all=True,
            should_flush_inventory=True,
            reason=reason
        )

    def get_analysis(self) -> Optional[OrderbookAnalysis]:
        """Get the last orderbook analysis."""
        return self.last_analysis

    def round_order(self, order: OrderLevel, coin: str) -> Optional[OrderLevel]:
        """Round an order to coin precision; returns None if it becomes invalid."""
        try:
            import math
            price_decimals = getattr(self, 'price_decimals', config.get_price_precision(coin))
            size_decimals = config.get_size_precision(coin)
            
            tick = getattr(self, 'tick_size', 10 ** (-price_decimals))
            maker_only = getattr(config, 'MAKER_ONLY', False)
            
            # Round size normally
            size = round(float(order.size), int(size_decimals))
            
            # Round price based on MAKER_ONLY rules
            # To stay maker, we MUST round AWAY from the mid price.
            # If we round a BUY up, we might hit the ask (taker).
            # If we round a SELL down, we might hit the bid (taker).
            raw_price = float(order.price)
            
            if maker_only and not order.reduce_only:
                if order.side == "BUY":
                    # Round DOWN to be safe
                    price = math.floor(raw_price / tick) * tick
                else:
                    # Round UP to be safe
                    price = math.ceil(raw_price / tick) * tick
            else:
                # Standard rounding for aggressive/exit orders
                price = round(raw_price / tick) * tick
            
            # Clean up floating point
            price = round(price, int(price_decimals))
            
        except Exception:
            return None

        if price <= 0 or size <= 0:
            return None

        return OrderLevel(
            price=price,
            size=size,
            side=order.side,
            is_aggressive=order.is_aggressive,
            reduce_only=order.reduce_only,
            reason=order.reason,
        )

