# risk_manager.py - stops the bot if we lose too much money

import json
import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Callable

import config

logger = logging.getLogger("RiskManager")


class RiskManager:
    """tracks pnl and stops trading if we hit our loss limit"""

    def __init__(
        self,
        max_daily_loss_pct: float = None,
        max_drawdown_pct: float = None,
        state_file: str = None
    ):
        # Configuration
        self.max_daily_loss_pct = max_daily_loss_pct or config.MAX_DAILY_LOSS_PCT
        self.max_drawdown_pct = max_drawdown_pct or config.MAX_DRAWDOWN_PCT
        self.state_file = state_file or config.RISK_STATE_FILE
        
        # State tracking
        self.starting_equity = 0.0
        self.current_equity = 0.0
        self.daily_high_equity = 0.0
        self.daily_low_equity = float('inf')
        
        # Kill switch
        self.is_killed = False
        self.kill_reason = ""
        self.kill_time: Optional[float] = None
        
        # Position tracking
        self.position_entry_time: Optional[float] = None
        self.position_entry_price = 0.0
        self.max_hold_time = config.MAX_HOLD_TIME_SECONDS
        
        # Daily tracking
        self.last_reset_date = datetime.now(timezone.utc).date()
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.weekly_starting_equity = 0.0
        self.monthly_starting_equity = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        
        # Callbacks
        self._on_kill_callbacks: list[Callable] = []
        
        # Load saved state
        self._load_state()

    def _load_state(self):
        """Load persisted state from disk."""
        if not os.path.exists(self.state_file):
            logger.info(" No saved risk state found, starting fresh")
            return
            
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # Check if state is from today
            saved_date = datetime.strptime(data['date'], "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()
            
            if saved_date == today:
                self.starting_equity = data.get('starting_equity', 0)
                self.weekly_starting_equity = data.get('weekly_starting_equity', 0)
                self.monthly_starting_equity = data.get('monthly_starting_equity', 0)
                self.daily_high_equity = data.get('daily_high_equity', 0)
                self.is_killed = data.get('is_killed', False)
                self.kill_reason = data.get('kill_reason', '')
                self.daily_pnl = data.get('daily_pnl', 0)
                self.daily_trades = data.get('daily_trades', 0)
                
                # Backwards compatibility for new features
                if self.weekly_starting_equity == 0 and self.starting_equity > 0:
                    self.weekly_starting_equity = self.starting_equity
                if self.monthly_starting_equity == 0 and self.starting_equity > 0:
                    self.monthly_starting_equity = self.starting_equity
                
                logger.info(f" Loaded risk state from today. PnL: ${self.daily_pnl:.2f}")
                
                if self.is_killed:
                    logger.warning(f" Bot was killed earlier: {self.kill_reason}")
            else:
                logger.info(" Previous state is from yesterday, starting fresh")
                # Try to recover weekly/monthly starts from previous state if valid
                # (This is tricky without full history, so we'll reset if date mismatch for now
                #  unless we want to implement complex date logic here. 
                #  Simpler: Let _daily_reset handle logic if we loaded yesterday's state? 
                #  No, _load_state discards it. 
                #  Better: Just start fresh. If user restarts bot across days, they lose history 
                #  unless we change how state is stored. 
                #  For now, we'll stick to the current pattern but initialize weekly/monthly.)
                pass
                
        except Exception as e:
            logger.error(f"Error loading risk state: {e}")

    def _save_state(self):
        """Persist state to disk."""
        data = {
            "date": str(datetime.now(timezone.utc).date()),
            "starting_equity": self.starting_equity,
            "weekly_starting_equity": self.weekly_starting_equity,
            "monthly_starting_equity": self.monthly_starting_equity,
            "daily_high_equity": self.daily_high_equity,
            "daily_low_equity": self.daily_low_equity if self.daily_low_equity != float('inf') else 0,
            "is_killed": self.is_killed,
            "kill_reason": self.kill_reason,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "daily_wins": self.daily_wins,
            "daily_losses": self.daily_losses
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")

    def update(self, current_equity: float, current_position_usd: float = 0) -> bool:
        """
        Update risk state with current account data.
        Call this on every tick.
        
        Args:
            current_equity: Current total account value in USD
            current_position_usd: Current position value (for time-based checks)
            
        Returns:
            True if trading is allowed, False if killed
        """
        current_time = time.time()
        today = datetime.now(timezone.utc).date()
        
        # Daily reset check
        if today != self.last_reset_date:
            self._daily_reset(current_equity)
        
        # Update current equity (CRITICAL: needed for _check_kill_conditions)
        self.current_equity = current_equity
        
        # Update current state
        # Initialize on first run
        if self.starting_equity == 0:
            self.starting_equity = current_equity
            self.weekly_starting_equity = current_equity
            self.monthly_starting_equity = current_equity
            self.daily_high_equity = current_equity
            self.daily_low_equity = current_equity
            self._save_state()
        
        # Update high/low watermarks
        if current_equity > self.daily_high_equity:
            self.daily_high_equity = current_equity
            self._save_state()
            
        if current_equity < self.daily_low_equity:
            self.daily_low_equity = current_equity
        
        # Calculate metrics
        self.daily_pnl = current_equity - self.starting_equity
        self.weekly_pnl = current_equity - self.weekly_starting_equity
        self.monthly_pnl = current_equity - self.monthly_starting_equity
        
        # Check kill conditions
        if not self.is_killed:
            self._check_kill_conditions()
        
        # Check position time (rotten fish)
        if abs(current_position_usd) > 10:  # Has meaningful position
            if self.position_entry_time is None:
                self.position_entry_time = current_time
        else:
            self.position_entry_time = None
        
        return self.can_trade()

    def _daily_reset(self, current_equity: float):
        """Reset daily counters at midnight UTC."""
        logger.info(" New trading day! Resetting risk metrics.")
        
        today = datetime.now(timezone.utc).date()
        
        # Weekly Reset (Monday)
        if today.isocalendar()[1] != self.last_reset_date.isocalendar()[1]:
             logger.info(" New trading week! Resetting weekly metrics.")
             self.weekly_starting_equity = current_equity
        
        # Monthly Reset (1st of month)
        if today.month != self.last_reset_date.month:
             logger.info(" New trading month! Resetting monthly metrics.")
             self.monthly_starting_equity = current_equity

        self.starting_equity = current_equity
        self.daily_high_equity = current_equity
        self.daily_low_equity = current_equity
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.is_killed = False
        self.kill_reason = ""
        
        self.last_reset_date = today
        self._save_state()

    def _check_kill_conditions(self):
        """Check if any kill conditions are met."""
        # Guard - don't check if not properly initialized
        if self.starting_equity == 0 or self.current_equity == 0:
            return
            
        # Rule 1: Daily Loss Limit
        loss_pct = (self.current_equity - self.starting_equity) / self.starting_equity
        if loss_pct < -self.max_daily_loss_pct:
            self._trigger_kill(
                f"Daily loss limit hit: {loss_pct*100:.2f}% "
                f"(limit: {self.max_daily_loss_pct*100:.1f}%)"
            )
            return
        
        # Rule 2: Drawdown from Daily High
        if self.daily_high_equity > 0:
            drawdown_pct = (self.current_equity - self.daily_high_equity) / self.daily_high_equity
            if drawdown_pct < -self.max_drawdown_pct:
                self._trigger_kill(
                    f"Max drawdown hit: {drawdown_pct*100:.2f}% from high "
                    f"(limit: {self.max_drawdown_pct*100:.1f}%)"
                )
                return

    def _trigger_kill(self, reason: str):
        """Activate the kill switch."""
        self.is_killed = True
        self.kill_reason = reason
        self.kill_time = time.time()
        
        logger.critical(f" KILL SWITCH ACTIVATED: {reason}")
        
        self._save_state()
        
        # Trigger callbacks
        for callback in self._on_kill_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Kill callback error: {e}")

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return not self.is_killed

    def check_position_time(self, current_pnl_pct: float) -> bool:
        """
        Check if position has been held too long (rotten fish rule).
        
        Args:
            current_pnl_pct: Current position PnL as percentage
            
        Returns:
            True if position should be closed due to time
        """
        if self.position_entry_time is None:
            return False
            
        time_held = time.time() - self.position_entry_time
        
        # If losing and held too long, should close
        if time_held > self.max_hold_time and current_pnl_pct < 0:
            logger.warning(
                f" Rotten fish! Position held {time_held/60:.1f} min with "
                f"{current_pnl_pct*100:.2f}% loss"
            )
            return True
            
        return False

    def record_trade(self, pnl: float):
        """Record a completed trade for statistics."""
        self.daily_trades += 1
        
        if pnl > 0:
            self.daily_wins += 1
        elif pnl < 0:
            self.daily_losses += 1
        
        self._save_state()

    def force_kill(self, reason: str):
        """Externally force a kill switch activation."""
        self._trigger_kill(f"Manual kill: {reason}")

    def reset_kill(self):
        """Reset the kill switch (use with caution!)."""
        if self.is_killed:
            logger.warning(" Kill switch manually reset!")
            self.is_killed = False
            self.kill_reason = ""
            self._save_state()

    def on_kill(self, callback: Callable):
        """Register a callback to be called when kill switch activates."""
        self._on_kill_callbacks.append(callback)

    def get_summary(self) -> dict:
        """Get current risk status summary."""
        pnl_pct = 0
        dd_pct = 0
        
        if self.starting_equity > 0:
            pnl_pct = (self.current_equity - self.starting_equity) / self.starting_equity
            
        if self.daily_high_equity > 0:
            dd_pct = (self.current_equity - self.daily_high_equity) / self.daily_high_equity
        
        win_rate = 0
        if self.daily_trades > 0:
            win_rate = (self.daily_wins / self.daily_trades) * 100
        
        return {
            "is_killed": self.is_killed,
            "kill_reason": self.kill_reason,
            "starting_equity": f"${self.starting_equity:.2f}",
            "current_equity": f"${self.current_equity:.2f}",
            "daily_pnl": f"${self.daily_pnl:.2f}",
            "daily_pnl_pct": f"{pnl_pct*100:.2f}%",
            "daily_high": f"${self.daily_high_equity:.2f}",
            "drawdown_pct": f"{dd_pct*100:.2f}%",
            "daily_trades": self.daily_trades,
            "win_rate": f"{win_rate:.1f}%",
            "can_trade": self.can_trade()
        }

    def get_status_line(self) -> str:
        """Get a one-line status for logging."""
        pnl_icon = "" if self.daily_pnl >= 0 else ""
        status = "" if self.can_trade() else ""
        
        return (
            f"{status} PnL: {pnl_icon}${self.daily_pnl:+.2f} | "
            f"Trades: {self.daily_trades} | "
            f"High: ${self.daily_high_equity:.2f}"
        )
