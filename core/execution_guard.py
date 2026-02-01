# execution_guard.py - checks if its safe to trade right now

import time
import logging
from collections import deque
from typing import Tuple

import config

logger = logging.getLogger("ExecutionGuard")


class ToxicFlowGuard:
    """checks if the market is dangerous right now (crash, pump, wide spread, etc)"""

    def __init__(self):
        # Thresholds from config
        self.imbalance_threshold = config.TOXIC_IMBALANCE_THRESHOLD
        self.max_short_term_vol = config.MAX_SHORT_TERM_VOLATILITY
        self.order_stale_seconds = config.ORDER_STALE_SECONDS
        self.min_spread = config.MIN_SPREAD_PCT
        
        # Price history for crash detection
        self.price_history = deque(maxlen=100)  # (timestamp, price)
        
        # === TICK-TO-TICK VELOCITY (Flash Crash Detection) ===
        # In crypto, a 5% crash can happen in 1 second
        # This provides sub-second protection
        self.tick_velocity_enabled = getattr(config, 'TICK_VELOCITY_ENABLED', True)
        self.tick_velocity_window_ms = getattr(config, 'TICK_VELOCITY_WINDOW_MS', 500)  # 500ms window
        self.tick_velocity_threshold = getattr(config, 'TICK_VELOCITY_THRESHOLD', 0.01)  # 1% in 500ms
        self.tick_history = deque(maxlen=50)  # High-res tick history (time_ns, price)
        self.last_velocity_check_ns = 0
        
        # State tracking
        self.last_check_time = 0.0
        self.is_toxic = False
        self.toxic_reason = ""
        
        # Cooldown after toxic detection
        self.toxic_cooldown = 10.0  # Wait 10 seconds after toxic clears
        self.toxic_cleared_time = 0.0

    def update_price(self, price: float):
        """Track price for crash/pump detection."""
        now = time.time()
        self.price_history.append((now, price))
        
        # Also track high-resolution ticks for velocity
        if self.tick_velocity_enabled:
            now_ns = time.time_ns()
            self.tick_history.append((now_ns, price))

    def is_safe_to_trade(
        self,
        side: str,
        imbalance: float,
        spread_pct: float,
        volatility_short: float
    ) -> Tuple[bool, str]:
        """
        Check if it's safe to place/maintain an order on the given side.
        
        Args:
            side: "BUY" or "SELL"
            imbalance: Orderbook imbalance (0.0=all sellers, 1.0=all buyers)
            spread_pct: Current spread as percentage
            volatility_short: Short-term volatility
            
        Returns:
            Tuple of (is_safe, reason)
        """
        current_time = time.time()
        self.last_check_time = current_time
        
        # Check 1: Orderbook Imbalance (Toxic Flow Detection)
        if side == "BUY":
            # We want to BUY - check if there's massive sell pressure
            if imbalance < (1 - self.imbalance_threshold):
                reason = f"Toxic: Sell pressure {(1-imbalance)*100:.0f}% - don't buy into this"
                return False, reason
        else:  # SELL
            # We want to SELL - check if there's massive buy pressure
            if imbalance > self.imbalance_threshold:
                reason = f"Toxic: Buy pressure {imbalance*100:.0f}% - don't sell here"
                return False, reason
        
        # Check 2: Crash/Pump Detection
        crash_safe, crash_reason = self._check_crash(side)
        if not crash_safe:
            self._set_toxic(crash_reason)
            return False, crash_reason
        
        # Check 2.5: Tick-to-Tick Velocity (Flash Crash - Sub-second detection)
        velocity_safe, velocity_reason = self._check_tick_velocity(side)
        if not velocity_safe:
            self._set_toxic(velocity_reason)
            return False, velocity_reason
        
        # Check 3: Spread Blowout
        if spread_pct > 0.005:  # > 0.5% spread
            reason = f"Toxic: Spread too wide ({spread_pct*100:.2f}%) - market makers fled"
            self._set_toxic(reason)
            return False, reason
        
        # NOTE: Minimum spread check REMOVED
        # A tight market spread is GOOD (liquid market). 
        # Profitability is ensured by FeeGuard which checks our QUOTED spread.
        
        # Check 5: High short-term volatility
        if volatility_short > self.max_short_term_vol:
            reason = f"Toxic: High volatility ({volatility_short*100:.2f}%) - unstable market"
            self._set_toxic(reason)
            return False, reason
        
        # Check cooldown if we were recently toxic
        if self.is_toxic:
            if current_time - self.toxic_cleared_time < self.toxic_cooldown:
                remaining = self.toxic_cooldown - (current_time - self.toxic_cleared_time)
                return False, f"Cooling down ({remaining:.1f}s remaining)"
            else:
                self.is_toxic = False
                self.toxic_reason = ""
        
        return True, "Safe"

    def _check_crash(self, side: str) -> Tuple[bool, str]:
        """
        Detect if price is crashing or pumping too fast.
        """
        if len(self.price_history) < 10:
            return True, "Not enough data"
        
        current_time = time.time()
        current_price = self.price_history[-1][1]
        
        # Look at price 10 seconds ago
        ten_sec_ago = None
        for ts, price in self.price_history:
            if current_time - ts >= 10:
                ten_sec_ago = price
            else:
                break
        
        if ten_sec_ago is None:
            return True, "OK"
        
        # Calculate percentage change
        pct_change = (current_price - ten_sec_ago) / ten_sec_ago
        
        if side == "BUY":
            # Don't buy if price is crashing
            if pct_change < -self.max_short_term_vol:
                return False, f"Falling knife! Price dropped {abs(pct_change)*100:.2f}% in 10s"
        else:  # SELL
            # Don't sell if price is pumping
            if pct_change > self.max_short_term_vol:
                return False, f"Rocket detected! Price pumped {pct_change*100:.2f}% in 10s"
        
        return True, "OK"

    def _check_tick_velocity(self, side: str) -> Tuple[bool, str]:
        """
        Check for extreme tick-to-tick price velocity (flash crash detection).
        
        Unlike _check_crash which uses a 10-second window, this looks at
        sub-second movements to catch flash crashes that can wipe out
        positions before the 10-second check reacts.
        
        In crypto, a 5% crash can happen in under 1 second.
        """
        if not self.tick_velocity_enabled:
            return True, "Velocity check disabled"
            
        if len(self.tick_history) < 5:
            return True, "Not enough tick data"
        
        now_ns = time.time_ns()
        window_ns = self.tick_velocity_window_ms * 1_000_000  # Convert ms to ns
        
        current_price = self.tick_history[-1][1]
        
        # Find the price at the start of our velocity window
        window_start_price = None
        window_start_ns = None
        for ts_ns, price in self.tick_history:
            if now_ns - ts_ns >= window_ns:
                window_start_price = price
                window_start_ns = ts_ns
            else:
                break
        
        if window_start_price is None or window_start_ns is None:
            return True, "OK"
        
        # Calculate velocity (% change in the time window)
        pct_change = (current_price - window_start_price) / window_start_price
        actual_window_ms = (now_ns - window_start_ns) / 1_000_000
        
        # Also calculate instantaneous velocity (last 2-3 ticks)
        if len(self.tick_history) >= 3:
            recent_ticks = list(self.tick_history)[-3:]
            instant_change = (recent_ticks[-1][1] - recent_ticks[0][1]) / recent_ticks[0][1]
            instant_window_ms = (recent_ticks[-1][0] - recent_ticks[0][0]) / 1_000_000
            
            # If 3 ticks show extreme move, that's a flash crash signal
            if instant_window_ms > 0 and instant_window_ms < 200:  # Within 200ms
                if abs(instant_change) > self.tick_velocity_threshold * 0.5:
                    # Extreme velocity in very short time
                    direction = "crashed" if instant_change < 0 else "pumped"
                    logger.warning(
                        f" FLASH MOVE: Price {direction} {abs(instant_change)*100:.3f}% "
                        f"in {instant_window_ms:.0f}ms (3 ticks)"
                    )
                    if side == "BUY" and instant_change < 0:
                        return False, f"Flash crash! {abs(instant_change)*100:.2f}% drop in {instant_window_ms:.0f}ms"
                    elif side == "SELL" and instant_change > 0:
                        return False, f"Flash pump! {instant_change*100:.2f}% rise in {instant_window_ms:.0f}ms"
        
        # Check the full velocity window
        if side == "BUY":
            if pct_change < -self.tick_velocity_threshold:
                return False, f"Velocity crash! {abs(pct_change)*100:.2f}% drop in {actual_window_ms:.0f}ms"
        else:  # SELL
            if pct_change > self.tick_velocity_threshold:
                return False, f"Velocity pump! {pct_change*100:.2f}% rise in {actual_window_ms:.0f}ms"
        
        return True, "OK"

    def get_velocity_metrics(self) -> dict:
        """Get current velocity metrics for monitoring."""
        if len(self.tick_history) < 2:
            return {"enabled": self.tick_velocity_enabled, "ticks": 0, "velocity_pct": 0.0}
        
        now_ns = time.time_ns()
        window_ns = self.tick_velocity_window_ms * 1_000_000
        
        current_price = self.tick_history[-1][1]
        window_start_price = None
        
        for ts_ns, price in self.tick_history:
            if now_ns - ts_ns >= window_ns:
                window_start_price = price
            else:
                break
        
        if window_start_price is None:
            return {"enabled": True, "ticks": len(self.tick_history), "velocity_pct": 0.0}
        
        velocity = (current_price - window_start_price) / window_start_price
        return {
            "enabled": True,
            "ticks": len(self.tick_history),
            "velocity_pct": velocity * 100,
            "threshold_pct": self.tick_velocity_threshold * 100,
            "window_ms": self.tick_velocity_window_ms
        }

    def _set_toxic(self, reason: str):
        """Mark market as toxic."""
        if not self.is_toxic:
            logger.warning(f" TOXIC FLOW DETECTED: {reason}")
        self.is_toxic = True
        self.toxic_reason = reason
        self.toxic_cleared_time = time.time()

    def is_order_stale(self, order_timestamp: float) -> bool:
        """
        Check if an order has been sitting too long.
        Stale orders are "dumb money" that HFTs will pick off.
        """
        age = time.time() - order_timestamp
        if age > self.order_stale_seconds:
            logger.debug(f" Order is stale ({age:.1f}s old)")
            return True
        return False

    def get_status(self) -> dict:
        """Get current guard status including velocity metrics."""
        status = {
            "is_toxic": self.is_toxic,
            "toxic_reason": self.toxic_reason if self.is_toxic else "",
            "price_history_len": len(self.price_history),
        }
        
        # Add velocity metrics if enabled
        if self.tick_velocity_enabled:
            status["velocity"] = self.get_velocity_metrics()
        
        return status


class FeeGuard:
    """makes sure we actually profit after paying fees"""

    def __init__(self):
        self.maker_fee = float(getattr(config, 'MAKER_FEE', 0.0) or 0.0)
        self.taker_fee = float(getattr(config, 'TAKER_FEE', 0.0) or 0.0)

        # Optional: auto-fetch user fee rates from Hyperliquid.
        # This reflects your current tier/discounts instead of a hard-coded estimate.
        if bool(getattr(config, 'AUTO_FETCH_FEES', False)):
            self._try_autofetch_fees()

        # Minimum expected NET profit (as decimal pct). Config uses MIN_PROFIT (e.g. 0.00001 = 0.001%).
        self.min_profit_threshold = float(getattr(config, 'MIN_PROFIT_THRESHOLD', 0.0) or 0.0)
        
        # Calculate hurdle rate (minimum move needed)
        # Entry (Maker) + Exit (Taker) + Desired Profit
        self.hurdle_rate = self.maker_fee + self.taker_fee + self.min_profit_threshold

    def _try_autofetch_fees(self) -> None:
        address = str(getattr(config, 'WALLET_ADDRESS', '') or '').strip()
        if not address:
            logger.warning("AUTO_FETCH_FEES=True but WALLET_ADDRESS is empty; using configured fees")
            return

        try:
            from hyperliquid.info import Info
            from hyperliquid.utils import constants

            api_url = constants.TESTNET_API_URL if getattr(config, 'USE_TESTNET', True) else constants.MAINNET_API_URL
            info = Info(api_url, skip_ws=True)
            data = info.user_fees(address)

            maker = float(data.get('userAddRate', self.maker_fee))
            taker = float(data.get('userCrossRate', self.taker_fee))

            # Sanity: avoid obviously broken values.
            if taker < 0 or taker > 0.05:
                raise ValueError(f"unexpected taker fee: {taker}")

            # If you don't have fee rebates, never treat maker as negative.
            if not bool(getattr(config, 'ALLOW_MAKER_REBATE', False)):
                maker = max(0.0, maker)

            self.maker_fee = maker
            self.taker_fee = taker
            logger.info(f" Fees auto-fetched: maker={self.maker_fee*100:.4f}% taker={self.taker_fee*100:.4f}%")
        except Exception as e:
            logger.warning(f"Fee auto-fetch failed; using configured fees (maker={self.maker_fee}, taker={self.taker_fee}). Error: {e}")

    def is_profitable_market_making(
        self,
        bid_price: float,
        ask_price: float,
        maker_only: bool = True,
    ) -> Tuple[bool, float]:
        """Evaluate profitability of a passive market-making quote pair.

        For a classic market-making round trip you expect to buy at your bid and
        sell at your ask (or vice versa), ideally both as maker when using post-only.

        Returns:
            (is_profitable, expected_net_profit_pct)
        """
        if bid_price <= 0 or ask_price <= 0:
            return False, 0.0
        if ask_price <= bid_price:
            return False, 0.0

        mid = (bid_price + ask_price) / 2
        if mid <= 0:
            return False, 0.0

        gross_capture = (ask_price - bid_price) / mid

        # Fee model:
        # - maker_only quotes: assume maker on both fills (most common for post-only quoting)
        # - otherwise: worst-case maker on entry, taker on exit
        total_fees = (2 * self.maker_fee) if maker_only else (self.maker_fee + self.taker_fee)
        net_profit = gross_capture - total_fees

        return net_profit >= self.min_profit_threshold, net_profit

    def is_profitable(
        self,
        entry_price: float,
        target_price: float,
        use_taker_entry: bool = False
    ) -> Tuple[bool, float]:
        """
        Check if a trade would be profitable after fees.
        
        Args:
            entry_price: Our entry price
            target_price: Expected exit price
            use_taker_entry: True if we're paying taker fee on entry (aggressive orders)
            
        Returns:
            Tuple of (is_profitable, expected_profit_pct)
        """
        # Calculate raw move percentage
        raw_move = abs(target_price - entry_price) / entry_price
        
        # Calculate fees
        if use_taker_entry:
            total_fees = self.taker_fee + self.taker_fee  # Taker both ways
        else:
            total_fees = self.maker_fee + self.taker_fee  # Maker entry, Taker exit
        
        # Net profit
        net_profit = raw_move - total_fees
        
        is_profitable = net_profit >= self.min_profit_threshold
        
        return is_profitable, net_profit

    def get_min_target_price(self, entry_price: float, side: str) -> float:
        """
        Calculate the minimum target price needed for profitability.
        
        Args:
            entry_price: Entry price
            side: "BUY" or "SELL"
            
        Returns:
            Minimum exit price for profitability
        """
        if side == "BUY":
            # We bought, need price to go UP
            return entry_price * (1 + self.hurdle_rate)
        else:
            # We sold, need price to go DOWN
            return entry_price * (1 - self.hurdle_rate)

    def should_take_signal(
        self,
        signal_strength: float,
        expected_move_pct: float
    ) -> Tuple[bool, str]:
        """
        Evaluate if a trading signal is worth taking.
        
        Args:
            signal_strength: 0.0 to 1.0 confidence in the signal
            expected_move_pct: Expected price move percentage
            
        Returns:
            Tuple of (should_trade, reason)
        """
        # Minimum expected move must cover fees
        if expected_move_pct < self.hurdle_rate:
            return False, f"Expected move ({expected_move_pct*100:.3f}%) < hurdle ({self.hurdle_rate*100:.3f}%)"
        
        # Weak signals with small moves = noise
        if signal_strength < 0.5 and expected_move_pct < self.hurdle_rate * 2:
            return False, "Signal too weak for the expected move"
        
        return True, "Signal passes fee guard"

    def get_summary(self) -> dict:
        """Get fee guard configuration."""
        return {
            "maker_fee": f"{self.maker_fee*100:.3f}%",
            "taker_fee": f"{self.taker_fee*100:.3f}%",
            "hurdle_rate": f"{self.hurdle_rate*100:.3f}%",
            "min_profit": f"{self.min_profit_threshold*100:.3f}%"
        }
