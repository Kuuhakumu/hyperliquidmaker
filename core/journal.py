# journal.py - logs all trades so we can look at them later

import json
import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

import config

logger = logging.getLogger("Journal")


@dataclass
class FillRecord:
    """one fill from the exchange"""
    fill_id: str
    timestamp: float
    datetime_str: str
    coin: str
    side: str  # "BUY" or "SELL"
    price: float  # Actual fill price
    size: float
    fee: float  # Fee paid in USD
    is_maker: bool  # True if maker, False if taker
    oid: str  # Original order ID
    notional_usd: float  # price * size


@dataclass
class MarketContext:
    """Snapshot of market conditions at trade time."""
    best_bid: float
    best_ask: float
    mid_price: float
    spread_pct: float
    imbalance: float
    volatility: float
    volatility_short: float
    micro_price: float
    trend: str
    regime: str
    hour_of_day: int
    day_of_week: int


@dataclass
class TradeRecord:
    """Complete trade record for analysis."""
    # Identification
    trade_id: str
    timestamp: float
    datetime_str: str
    coin: str
    
    # Trade details
    side: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    size: float
    
    # Results
    pnl_usd: float
    pnl_pct: float
    result: str  # "WIN", "LOSS", "BREAKEVEN"
    
    # Context at entry
    entry_context: Dict
    
    # Context at exit
    exit_context: Dict
    
    # Reason for trade
    entry_reason: str
    exit_reason: str
    
    # Duration
    hold_time_seconds: float


class TradeJournal:
    """saves trades to a json file with market data at the time"""

    def __init__(self, filename: str = None):
        self.filename = filename or config.JOURNAL_FILE
        self.fills_filename = self.filename.replace('.json', '_fills.json')
        self.trades: List[TradeRecord] = []
        self.fills: List[FillRecord] = []  # P1 fix: track individual fills
        
        # Current open trade tracking
        self.open_trade: Optional[Dict] = None
        
        # Statistics
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0.0
        
        # Fill statistics (P1 fix)
        self.total_fills = 0
        self.total_maker_fills = 0
        self.total_taker_fills = 0
        self.total_fees_paid = 0.0
        self.seen_fill_ids: set = set()  # Dedup fills
        
        # Load existing data
        self._load()

    def _load(self):
        """Load existing trade history."""
        if not os.path.exists(self.filename):
            return
            
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                
            # Load stats
            self.total_trades = data.get('total_trades', 0)
            self.total_wins = data.get('total_wins', 0)
            self.total_losses = data.get('total_losses', 0)
            self.total_pnl = data.get('total_pnl', 0.0)
            
            logger.info(f" Loaded journal: {self.total_trades} trades, ${self.total_pnl:.2f} PnL")
            
        except Exception as e:
            logger.error(f"Error loading journal: {e}")

    def _save(self):
        """Save trade history to disk."""
        try:
            # Load existing trades
            existing_trades = []
            if os.path.exists(self.filename):
                try:
                    with open(self.filename, 'r') as f:
                        data = json.load(f)
                        existing_trades = data.get('trades', [])
                except:
                    pass
            
            # Add new trades
            all_trades = existing_trades + [asdict(t) for t in self.trades]
            
            # Keep last 1000 trades to prevent file from growing too large
            if len(all_trades) > 1000:
                all_trades = all_trades[-1000:]
            
            data = {
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'total_losses': self.total_losses,
                'total_pnl': self.total_pnl,
                'total_fills': self.total_fills,
                'total_fees_paid': self.total_fees_paid,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'trades': all_trades
            }
            
            with open(self.filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save fills to separate file (P1 fix)
            self._save_fills()
                
            # Clear local buffer after saving
            self.trades = []
            
        except Exception as e:
            logger.error(f"Error saving journal: {e}")
    
    def _save_fills(self):
        """Save fill records to disk (P1 fix)."""
        if not self.fills:
            return
            
        try:
            # Load existing fills
            existing_fills = []
            if os.path.exists(self.fills_filename):
                try:
                    with open(self.fills_filename, 'r') as f:
                        data = json.load(f)
                        existing_fills = data.get('fills', [])
                except:
                    pass
            
            # Add new fills
            all_fills = existing_fills + [asdict(f) for f in self.fills]
            
            # Keep last 5000 fills
            if len(all_fills) > 5000:
                all_fills = all_fills[-5000:]
            
            data = {
                'total_fills': self.total_fills,
                'total_maker_fills': self.total_maker_fills,
                'total_taker_fills': self.total_taker_fills,
                'total_fees_paid': self.total_fees_paid,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'fills': all_fills
            }
            
            with open(self.fills_filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.fills = []
            
        except Exception as e:
            logger.error(f"Error saving fills: {e}")
    
    def record_fill(
        self,
        fill_id: str,
        coin: str,
        side: str,
        price: float,
        size: float,
        fee: float,
        is_maker: bool,
        oid: str,
        timestamp: float = None
    ) -> Optional[FillRecord]:
        """
        Record an individual fill from the exchange (P1 fix).
        
        This provides accurate HFT tracking vs position-delta approach.
        
        Args:
            fill_id: Unique fill identifier from exchange
            coin: Trading pair
            side: "BUY" or "SELL"
            price: Actual fill price
            size: Fill size
            fee: Fee paid in USD
            is_maker: True if maker order, False if taker
            oid: Original order ID
            timestamp: Fill timestamp (defaults to now)
            
        Returns:
            FillRecord if recorded, None if duplicate
        """
        # Deduplicate
        if fill_id in self.seen_fill_ids:
            return None
        self.seen_fill_ids.add(fill_id)
        
        # Keep seen_fill_ids bounded
        if len(self.seen_fill_ids) > 10000:
            # Remove oldest half
            self.seen_fill_ids = set(list(self.seen_fill_ids)[-5000:])
        
        now = datetime.now(timezone.utc)
        ts = timestamp or time.time()
        
        record = FillRecord(
            fill_id=fill_id,
            timestamp=ts,
            datetime_str=now.strftime("%Y-%m-%d %H:%M:%S"),
            coin=coin,
            side=side,
            price=price,
            size=size,
            fee=fee,
            is_maker=is_maker,
            oid=oid,
            notional_usd=price * size
        )
        
        # Update stats
        self.total_fills += 1
        self.total_fees_paid += fee
        if is_maker:
            self.total_maker_fills += 1
        else:
            self.total_taker_fills += 1
        
        # Add to buffer
        self.fills.append(record)
        
        # Auto-save every 20 fills
        if len(self.fills) >= 20:
            self._save_fills()
        
        icon = "" if is_maker else ""
        logger.debug(
            f"{icon} Fill: {side} {size:.4f} {coin} @ ${price:.4f} "
            f"({'maker' if is_maker else 'taker'}) fee=${fee:.4f}"
        )
        
        return record
    
    def get_fill_stats(self) -> Dict:
        """Get fill statistics for analysis."""
        maker_rate = 0
        if self.total_fills > 0:
            maker_rate = (self.total_maker_fills / self.total_fills) * 100
        
        return {
            'total_fills': self.total_fills,
            'maker_fills': self.total_maker_fills,
            'taker_fills': self.total_taker_fills,
            'maker_rate': f"{maker_rate:.1f}%",
            'total_fees': f"${self.total_fees_paid:.2f}",
            'avg_fee': f"${self.total_fees_paid / max(self.total_fills, 1):.4f}"
        }

    def open_position(
        self,
        coin: str,
        side: str,
        entry_price: float,
        size: float,
        context: MarketContext,
        reason: str
    ):
        """
        Record opening a new position.
        """
        self.open_trade = {
            'trade_id': f"{coin}_{int(time.time()*1000)}",
            'timestamp': time.time(),
            'coin': coin,
            'side': side,
            'entry_price': entry_price,
            'size': size,
            'entry_context': asdict(context),
            'entry_reason': reason
        }
        
        logger.info(f" Opened {side} {size:.4f} {coin} @ {entry_price:.4f} ({reason})")

    def close_position(
        self,
        exit_price: float,
        context: MarketContext,
        reason: str
    ) -> Optional[TradeRecord]:
        """
        Record closing a position.
        
        Returns:
            TradeRecord if position was tracked, None otherwise
        """
        if self.open_trade is None:
            logger.warning("No open trade to close")
            return None
        
        # Calculate results
        entry_price = self.open_trade['entry_price']
        size = self.open_trade['size']
        side = self.open_trade['side']
        
        if side == "BUY":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
            
        pnl_usd = pnl_pct * size * entry_price
        
        # Determine result
        if pnl_usd > 0.01:
            result = "WIN"
            self.total_wins += 1
        elif pnl_usd < -0.01:
            result = "LOSS"
            self.total_losses += 1
        else:
            result = "BREAKEVEN"
        
        # Build trade record
        now = datetime.now(timezone.utc)
        record = TradeRecord(
            trade_id=self.open_trade['trade_id'],
            timestamp=time.time(),
            datetime_str=now.strftime("%Y-%m-%d %H:%M:%S"),
            coin=self.open_trade['coin'],
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            result=result,
            entry_context=self.open_trade['entry_context'],
            exit_context=asdict(context),
            entry_reason=self.open_trade['entry_reason'],
            exit_reason=reason,
            hold_time_seconds=time.time() - self.open_trade['timestamp']
        )
        
        # Update stats
        self.total_trades += 1
        self.total_pnl += pnl_usd
        
        # Add to buffer
        self.trades.append(record)
        
        # Clear open trade
        self.open_trade = None
        
        # Save periodically
        if len(self.trades) >= 5:
            self._save()
        
        icon = "" if result == "WIN" else "" if result == "LOSS" else ""
        logger.info(
            f"{icon} Closed {side} @ {exit_price:.4f} | "
            f"PnL: ${pnl_usd:+.2f} ({pnl_pct*100:+.2f}%) | "
            f"Held: {record.hold_time_seconds:.1f}s | {reason}"
        )
        
        return record

    def has_open_trade(self) -> bool:
        """Check if there's an open trade being tracked."""
        return self.open_trade is not None

    def get_open_trade_pnl(self, current_price: float) -> float:
        """Calculate current PnL on open trade."""
        if self.open_trade is None:
            return 0.0
            
        entry_price = self.open_trade['entry_price']
        size = self.open_trade['size']
        side = self.open_trade['side']
        
        if side == "BUY":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        return pnl_pct * size * entry_price

    def force_save(self):
        """Force save any buffered trades."""
        if self.trades:
            self._save()

    def get_recent_trades(self, count: int = 20) -> List[Dict]:
        """Get recent trades for analysis."""
        try:
            with open(self.filename, 'r') as f:
                data = json.load(f)
                trades = data.get('trades', [])
                return trades[-count:]
        except:
            return []

    def get_losses_for_analysis(self, count: int = 10) -> List[Dict]:
        """Get recent losing trades for AI analysis."""
        trades = self.get_recent_trades(100)
        losses = [t for t in trades if t.get('result') == 'LOSS']
        return losses[-count:]

    def get_wins_for_analysis(self, count: int = 10) -> List[Dict]:
        """Get recent winning trades for AI analysis."""
        trades = self.get_recent_trades(100)
        wins = [t for t in trades if t.get('result') == 'WIN']
        return wins[-count:]

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        win_rate = 0
        if self.total_trades > 0:
            win_rate = (self.total_wins / self.total_trades) * 100
            
        return {
            'total_trades': self.total_trades,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': f"{win_rate:.1f}%",
            'total_pnl': f"${self.total_pnl:.2f}",
            'avg_pnl': f"${self.total_pnl/max(self.total_trades,1):.2f}"
        }

    def export_for_analysis(self) -> str:
        """
        Export trade data in a format suitable for AI analysis.
        Returns a JSON string.
        """
        losses = self.get_losses_for_analysis(10)
        wins = self.get_wins_for_analysis(10)
        stats = self.get_stats()
        
        export = {
            'summary': stats,
            'recent_losses': losses,
            'recent_wins': wins
        }
        
        return json.dumps(export, indent=2)


def create_market_context(
    market_data,
    regime: str
) -> MarketContext:
    """
    Helper function to create a MarketContext from market data manager.
    """
    now = datetime.now(timezone.utc)
    
    return MarketContext(
        best_bid=market_data.best_bid,
        best_ask=market_data.best_ask,
        mid_price=market_data.mid_price,
        spread_pct=market_data.spread_pct,
        imbalance=market_data.imbalance,
        volatility=market_data.volatility,
        volatility_short=market_data.volatility_short,
        micro_price=market_data.get_micro_price(),
        trend=market_data.get_trend_direction(),
        regime=regime,
        hour_of_day=now.hour,
        day_of_week=now.weekday()
    )
