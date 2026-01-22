# regime_engine.py - decides if we should be farming (market making) or hunting (trend following)

import logging
import numpy as np
from collections import deque
from typing import Tuple, Optional
from enum import Enum

import config

logger = logging.getLogger("RegimeEngine")


class TradingRegime(Enum):
    """Trading regime states."""
    FARMER_NEUTRAL = "FARMER_NEUTRAL"      # Market making, balanced quotes
    FARMER_SKEW_LONG = "FARMER_SKEW_LONG"  # Market making, biased to sell (we're long)
    FARMER_SKEW_SHORT = "FARMER_SKEW_SHORT"  # Market making, biased to buy (we're short)
    HUNTER_LONG = "HUNTER_LONG"            # Aggressive buying (bull trend)
    HUNTER_SHORT = "HUNTER_SHORT"          # Aggressive selling (bear trend)
    HALTED = "HALTED"                      # Stop trading (risk limit or toxic flow)


class RegimeEngine:
    """figures out if market is calm (farm) or trending (hunt) and switches modes"""

    def __init__(self):
        # Thresholds from config
        self.volatility_threshold = config.VOLATILITY_THRESHOLD
        self.imbalance_bull_threshold = config.IMBALANCE_BULL_THRESHOLD
        self.imbalance_bear_threshold = config.IMBALANCE_BEAR_THRESHOLD
        self.hysteresis_multiplier = config.VOLATILITY_EXIT_MULTIPLIER
        self.imbalance_extreme_threshold = getattr(config, 'IMBALANCE_EXTREME_THRESHOLD', 0.80)
        
        # Current state
        self.current_regime = TradingRegime.FARMER_NEUTRAL
        self.previous_regime = TradingRegime.FARMER_NEUTRAL
        
        # Tracking
        self.regime_start_time = 0.0
        self.regime_switches = 0
        
        # Cooldown to prevent rapid switching
        self.min_regime_duration = config.MIN_REGIME_DURATION  # Minimum seconds in a regime before switching
        self.last_switch_time = 0.0
        
        # History for analysis
        self.regime_history = deque(maxlen=100)

    def evaluate(
        self,
        volatility: float,
        imbalance: float,
        current_position_usd: float,
        max_position_usd: float
    ) -> Tuple[TradingRegime, dict]:
        """
        Evaluate current market conditions and determine the trading regime.
        
        Args:
            volatility: Standard deviation of recent prices (as percentage of price)
            imbalance: Orderbook imbalance (0.0 = all sellers, 1.0 = all buyers)
            current_position_usd: Current position value in USD (positive = long, negative = short)
            max_position_usd: Maximum allowed position size
            
        Returns:
            Tuple of (TradingRegime, metadata dict)
        """
        import time
        current_time = time.time()
        
        # Calculate derived metrics
        inventory_ratio = current_position_usd / max_position_usd if max_position_usd > 0 else 0
        
        # Determine regime thresholds based on current state (hysteresis)
        if self.current_regime in [TradingRegime.HUNTER_LONG, TradingRegime.HUNTER_SHORT]:
            # We're in Hunter mode - use LOWER threshold to exit (easier to return to Farmer)
            vol_threshold = self.volatility_threshold * self.hysteresis_multiplier
        else:
            # We're in Farmer mode - use HIGHER threshold to enter Hunter (harder to switch)
            vol_threshold = self.volatility_threshold
        
        # Check cooldown
        time_in_regime = current_time - self.last_switch_time
        can_switch = time_in_regime >= self.min_regime_duration
        
        # Calculate dynamic thresholds
        bull_thresh, bear_thresh = self.calculate_dynamic_imbalance_threshold(volatility)
        
        # Determine new regime
        new_regime = self._calculate_regime(
            volatility, 
            imbalance, 
            vol_threshold,
            inventory_ratio,
            can_switch,
            bull_thresh,
            bear_thresh
        )
        
        # Track regime change
        if new_regime != self.current_regime:
            self.previous_regime = self.current_regime
            self.current_regime = new_regime
            self.last_switch_time = current_time
            self.regime_switches += 1
            
            # Log the switch
            self.regime_history.append({
                "time": current_time,
                "from": self.previous_regime.value,
                "to": new_regime.value,
                "volatility": volatility,
                "imbalance": imbalance
            })
            
            logger.info(f" REGIME SWITCH: {self.previous_regime.value} -> {new_regime.value}")
        
        # Build metadata for strategy use
        metadata = {
            "volatility": volatility,
            "imbalance": imbalance,
            "inventory_ratio": inventory_ratio,
            "time_in_regime": time_in_regime,
            "regime_switches": self.regime_switches,
            "can_switch": can_switch,
            "skew_factor": self._calculate_skew_factor(inventory_ratio)
        }
        
        # DEBUG: Trace regime decision
        logger.debug(f"REGIME DEBUG: vol={volatility:.5f} thr={vol_threshold:.5f} imb={imbalance:.2f} can={can_switch} -> {self.current_regime.value}")
        
        return self.current_regime, metadata

    def calculate_dynamic_imbalance_threshold(self, volatility: float) -> Tuple[float, float]:
        """Adjust Hunter entry thresholds based on market conditions."""
        base_bull = self.imbalance_bull_threshold
        base_bear = self.imbalance_bear_threshold
        
        # Require stronger conviction in volatile markets
        # If vol is 0.5% (0.005), adjustment is 0.005 * 20 = 0.10 (10%)
        vol_adjustment = min(0.15, volatility * 20.0)
        
        return (
            min(0.95, base_bull + vol_adjustment),
            max(0.05, base_bear - vol_adjustment)
        )

    def _calculate_regime(
        self,
        volatility: float,
        imbalance: float,
        vol_threshold: float,
        inventory_ratio: float,
        can_switch: bool,
        bull_threshold: float = None,
        bear_threshold: float = None
    ) -> TradingRegime:
        """
        Core regime calculation logic.
        """
        bull_thresh = bull_threshold if bull_threshold is not None else self.imbalance_bull_threshold
        bear_thresh = bear_threshold if bear_threshold is not None else self.imbalance_bear_threshold

        # --- HUNTER MODE CONDITIONS ---
        # High volatility OR Extreme Imbalance = Trend detected
        # We want to catch quiet trends (low vol, high imbalance)
        is_high_vol = volatility > vol_threshold
        # is_extreme_imbalance = imbalance > 0.80 or imbalance < 0.20
        is_extreme_imbalance = (imbalance > self.imbalance_extreme_threshold) or \
                               (imbalance < (1.0 - self.imbalance_extreme_threshold))
        
        # --- EMERGENCY SHOCK DETECTION ---
        # If the market is moving VIOLENTLY against us, ignore the 30s timer.
        # This prevents getting trapped in a long during a news crash.
        is_shock = False
        if self.current_regime == TradingRegime.HUNTER_LONG and imbalance < 0.2 and volatility > vol_threshold:
            is_shock = True
            logger.warning(f" MARKET SHOCK (BEAR): Imbalance={imbalance:.2f} Vol={volatility:.5f}. Overriding sticky timer!")
        elif self.current_regime == TradingRegime.HUNTER_SHORT and imbalance > 0.8 and volatility > vol_threshold:
            is_shock = True
            logger.warning(f" MARKET SHOCK (BULL): Imbalance={imbalance:.2f} Vol={volatility:.5f}. Overriding sticky timer!")

        if (is_high_vol or is_extreme_imbalance) and (can_switch or is_shock):
            # Bullish: Heavy buy pressure
            if imbalance > bull_thresh:
                return TradingRegime.HUNTER_LONG
            # Bearish: Heavy sell pressure
            elif imbalance < bear_thresh:
                return TradingRegime.HUNTER_SHORT
        
        # --- EXIT HUNTER CONDITIONS ---
        # We need strong evidence that the trend is over.
        # Just dipping to 0.49 is not enough (hysteresis).
        if self.current_regime == TradingRegime.HUNTER_LONG:
            # Exit if:
            # 1. Imbalance strongly reverses (< 0.3)
            # 2. Volatility dies AND imbalance is not even slightly bullish (< 0.6)
            should_exit = (imbalance < 0.3) or (volatility < (vol_threshold * 0.8) and imbalance < 0.6)
            # 3. Emergency Shock (already handled above but for completeness)
            should_exit_fast = (imbalance < 0.15)
            
            if (should_exit or should_exit_fast) and (can_switch or should_exit_fast):
                logger.info(f" Exiting HUNTER_LONG: imb={imbalance:.2f} vol={volatility:.5f}")
                return self._get_farmer_regime(inventory_ratio)
            return TradingRegime.HUNTER_LONG
                
        elif self.current_regime == TradingRegime.HUNTER_SHORT:
            # Exit if:
            # 1. Imbalance strongly reverses (> 0.7)
            # 2. Volatility dies AND imbalance is not even slightly bearish (> 0.4)
            should_exit = (imbalance > 0.7) or (volatility < (vol_threshold * 0.8) and imbalance > 0.4)
            # 3. Emergency Shock
            should_exit_fast = (imbalance > 0.85)

            if (should_exit or should_exit_fast) and (can_switch or should_exit_fast):
                logger.info(f" Exiting HUNTER_SHORT: imb={imbalance:.2f} vol={volatility:.5f}")
                return self._get_farmer_regime(inventory_ratio)
            return TradingRegime.HUNTER_SHORT
        
        # --- FARMER MODE (Default) ---
        return self._get_farmer_regime(inventory_ratio)

    def _get_farmer_regime(self, inventory_ratio: float) -> TradingRegime:
        """
        Determine the appropriate Farmer regime based on inventory.
        """
        # Significant long position - need to sell
        # Hysteresis: Use config thresholds
        if self.current_regime == TradingRegime.FARMER_SKEW_LONG:
             if inventory_ratio > config.SKEW_MODE_EXIT_THRESHOLD:
                 return TradingRegime.FARMER_SKEW_LONG
        elif inventory_ratio > config.SKEW_MODE_ENTRY_THRESHOLD:
            return TradingRegime.FARMER_SKEW_LONG

        # Significant short position - need to buy
        if self.current_regime == TradingRegime.FARMER_SKEW_SHORT:
             if inventory_ratio < -config.SKEW_MODE_EXIT_THRESHOLD:
                 return TradingRegime.FARMER_SKEW_SHORT
        elif inventory_ratio < -config.SKEW_MODE_ENTRY_THRESHOLD:
            return TradingRegime.FARMER_SKEW_SHORT
            
        # Balanced
        return TradingRegime.FARMER_NEUTRAL

    def _calculate_skew_factor(self, inventory_ratio: float) -> float:
        """
        Calculate how much to skew quotes based on inventory.
        
        Positive skew = We're long, lower prices to encourage selling to us and us selling to others
        Negative skew = We're short, raise prices
        
        Returns: Percentage to shift the "fair price"
        """
        # Linear skew based on inventory
        # At 100% inventory, skew by SKEW_FACTOR percentage
        return inventory_ratio * config.SKEW_FACTOR

    def is_hunter_mode(self) -> bool:
        """Check if currently in aggressive Hunter mode."""
        return self.current_regime in [TradingRegime.HUNTER_LONG, TradingRegime.HUNTER_SHORT]

    def is_farmer_mode(self) -> bool:
        """Check if currently in passive Farmer mode."""
        return self.current_regime in [
            TradingRegime.FARMER_NEUTRAL,
            TradingRegime.FARMER_SKEW_LONG,
            TradingRegime.FARMER_SKEW_SHORT
        ]

    def halt(self):
        """Force halt trading (called by risk manager)."""
        self.previous_regime = self.current_regime
        self.current_regime = TradingRegime.HALTED
        logger.warning(" Regime HALTED by risk manager")

    def resume(self):
        """Resume trading after halt."""
        if self.current_regime == TradingRegime.HALTED:
            self.current_regime = TradingRegime.FARMER_NEUTRAL
            logger.info(" Regime RESUMED to FARMER_NEUTRAL")

    def get_regime_icon(self) -> str:
        """Get an icon representing the current regime for logging."""
        icons = {
            TradingRegime.FARMER_NEUTRAL: "",
            TradingRegime.FARMER_SKEW_LONG: "",
            TradingRegime.FARMER_SKEW_SHORT: "",
            TradingRegime.HUNTER_LONG: "",
            TradingRegime.HUNTER_SHORT: "",
            TradingRegime.HALTED: ""
        }
        return icons.get(self.current_regime, "")

    def get_summary(self) -> dict:
        """Get a summary of regime engine state."""
        return {
            "current_regime": self.current_regime.value,
            "previous_regime": self.previous_regime.value,
            "regime_switches": self.regime_switches,
            "is_hunter": self.is_hunter_mode(),
            "is_farmer": self.is_farmer_mode(),
            "is_halted": self.current_regime == TradingRegime.HALTED
        }
