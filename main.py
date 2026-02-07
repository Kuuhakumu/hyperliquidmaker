# main.py - runs the trading bot
# usage: python main.py --dry-run  (for testing)
#        python main.py            (for real trading)

import asyncio
import argparse
import functools
import logging
import signal
import sys
import time
from collections import deque
from datetime import datetime, timezone

import config
from core import (
    MarketDataManager,
    RegimeEngine,
    TradingRegime,
    StrategyEngine,
    ToxicFlowGuard,
    FeeGuard,
    OrderManager,
    RiskManager,
    TradeJournal,
    create_market_context,
    get_notifier,
    AlertLevel,
)
from core.recorder import DataRecorder
from core.sim_venue import SimulatedVenue, compute_taker_fill_price, OrderSide, FillEvent, RejectEvent

# LOGGING SETUP
def setup_logging():
    """Configure logging with color-coded output."""
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler (for detailed analysis)
    import os
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/bot.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)

logger = logging.getLogger("Main")


# THE BOT
class HyperLiquidBot:
    """
    The main trading bot that orchestrates all modules.
    
    Components:
    - MarketData: Real-time orderbook from websocket
    - RegimeEngine: Decides HUNTER vs FARMER mode
    - StrategyEngine: Generates order levels
    - ToxicFlowGuard: Protects against adverse selection
    - FeeGuard: Ensures profitability after fees
    - OrderManager: Places/cancels orders with diffing
    - RiskManager: Kill switch and PnL tracking
    - Journal: Records trades for AI analysis
    - Notifier: Discord alerts
    - Recorder: Captures raw data for replay (optional)
    """

    def __init__(self, dry_run: bool = False, capture_data: bool = False):
        self.dry_run = dry_run
        self.capture_data = capture_data
        self.running = False
        self.coin = config.COIN

        if not self.dry_run:
            if not getattr(config, 'PRIVATE_KEY', ''):
                raise ValueError("Missing PRIVATE_KEY in environment (.env) for LIVE mode")
        
        # Initialize components
        logger.info(f" Initializing HyperLiquid HFT Bot for {self.coin}...")
        logger.info(f"   Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info(f"   Network: {'TESTNET' if config.USE_TESTNET else 'MAINNET'}")
        
        # Market Data
        self.market_data = MarketDataManager(
            coin=self.coin,
            use_testnet=config.USE_TESTNET
        )
        
        # Data Recorder
        self.recorder = None
        if self.capture_data:
            self.recorder = DataRecorder(self.coin)
            logger.info("   VideoCapture: ENABLED ")
        
        # Strategy Components
        from core.ml_engine import MLEngine # Local import to avoid circular dependency
        self.ml_engine = MLEngine()
        
        self.regime_engine = RegimeEngine()
        self.strategy_engine = StrategyEngine(ml_engine=self.ml_engine)
        self.toxic_guard = ToxicFlowGuard()
        self.fee_guard = FeeGuard()
        
        # Execution
        if not dry_run:
            self.order_manager = OrderManager(
                private_key=config.PRIVATE_KEY,
                coin=self.coin,
                use_testnet=config.USE_TESTNET
            )
        else:
            self.order_manager = None
        
        # Risk & Tracking
        self.risk_manager = RiskManager()
        
        # Initialize Risk Manager with Sim Cash if Dry Run
        if self.dry_run:
            # In dry run, we need to tell RiskManager our starting "fake" equity
            # otherwise it sees 0 and thinks we lost 100%
            self.risk_manager.update(current_equity=1000.0, current_position_usd=0.0)
            
        self.journal = TradeJournal()
        self.notifier = get_notifier()
        
        # State tracking
        self.last_regime = TradingRegime.FARMER_NEUTRAL
        self.last_position = 0.0
        self.last_entry_price = 0.0 # Track entry price for PnL calc
        self.tick_count = 0
        self.hunter_entry_price = None  # Track entry price for Hunter stop loss
        
        # Simulation state (Paper Trading)
        self.sim_position_size = 0.0
        self.sim_entry_price = 0.0
        self.sim_cash = 1000.0  # Start with $1000 fake money
        self.sim_starting_cash = self.sim_cash
        self.sim_realized_pnl = 0.0
        self.sim_fees_paid = 0.0
        self.sim_fill_count = 0

        # Realistic Simulated Venue (dry-run only)
        self.sim_venue = None
        if self.dry_run and getattr(config, 'USE_REALISTIC_SIM', True):
            seed = getattr(config, 'SIM_RNG_SEED', None)
            self.sim_venue = SimulatedVenue(seed=seed)
            logger.info("   SimVenue: REALISTIC MODE (queue/latency/partials)")

        self.last_status_log = 0
        self.status_log_interval = 60  # Log status every 60s
        
        # Discord PnL Update Interval (every 15 mins)
        self.last_discord_update = 0
        self.discord_update_interval = float(getattr(config, "DISCORD_STATUS_UPDATE_INTERVAL_SECONDS", 900.0))

        # Rolling fill-rate tracking (live/testnet only)
        self._fill_timestamps = deque()
        
        # Spread capture tracking 
        self._last_entry_side = None
        self._last_entry_price = 0.0
        
        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(" Shutdown signal received...")
        self.running = False

    async def _run_blocking(self, func, *args, **kwargs):
        """Run blocking SDK/HTTP calls off the asyncio event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    def _notify_bg(self, func, *args, **kwargs):
        """Fire-and-forget notifications so network I/O can't stall trading."""
        try:
            asyncio.create_task(self._run_blocking(func, *args, **kwargs))
        except RuntimeError:
            # No running event loop (shouldn't happen in normal operation)
            pass

    def _apply_position_clamp(self, decision, position_usd: float):
        """Clamp orders so we never increase exposure beyond MAX_POSITION_USD."""
        cap = float(getattr(config, 'MAX_POSITION_USD', 0) or 0)
        if cap <= 0:
            return decision

        def allowed(order, side: str) -> bool:
            notional = float(order.price) * float(order.size)
            delta = notional if side == "BUY" else -notional
            new_exposure = position_usd + delta

            # If we're already over the cap, only allow exposure-reducing orders.
            if abs(position_usd) >= cap:
                return abs(new_exposure) < abs(position_usd)

            # Normal mode: allow if it stays within cap, or reduces exposure.
            return abs(new_exposure) <= cap or abs(new_exposure) < abs(position_usd)

        before_buys = len(decision.buy_orders)
        before_sells = len(decision.sell_orders)

        decision.buy_orders = [o for o in decision.buy_orders if allowed(o, "BUY")]
        decision.sell_orders = [o for o in decision.sell_orders if allowed(o, "SELL")]

        # If fully capped, don't quote the risk-increasing side at all.
        if abs(position_usd) >= cap:
            if position_usd > 0:
                decision.buy_orders = []
            elif position_usd < 0:
                decision.sell_orders = []

        if (before_buys, before_sells) != (len(decision.buy_orders), len(decision.sell_orders)):
            logger.warning(
                f" Position clamp: pos=${position_usd:+.2f} cap=${cap:.0f} "
                f"(buys {before_buys}->{len(decision.buy_orders)}, sells {before_sells}->{len(decision.sell_orders)})"
            )

        return decision

    async def start(self):
        """Start the bot."""
        logger.info(" Starting bot...")
        
        try:
            # Connect to market data
            await self.market_data.connect()
            
            # Log connection mode
            if self.market_data.is_websocket_mode():
                logger.info(" Market data: WEBSOCKET mode (low-latency)")
            else:
                logger.info(" Market data: REST polling mode")
            
            # Get initial account state
            if not self.dry_run:
                account_value = await self._run_blocking(self.order_manager.get_account_value)
                logger.info(f" Account Value: ${account_value:.2f}")
                
                if account_value < 10:
                    logger.error(" Account value too low. Need at least $10 to trade.")
                    return
            
            # Notify start
            self._notify_bg(self.notifier.bot_started, self.coin, "HYBRID")
            
            # Start Recorder
            if self.recorder:
                self.recorder.start()
            
            # Run main loop
            self.running = True
            await self._main_loop()
            
        except Exception as e:
            logger.error(f" Bot failed to start: {e}")
            self._notify_bg(self.notifier.error, str(e), "Startup")
            raise
        finally:
            # Stop Recorder
            if self.recorder:
                self.recorder.stop()
            await self._shutdown()

    async def _shutdown(self):
        """Graceful shutdown sequence."""
        logger.info(" Shutting down...")
        
        # Stop recorder if running
        if self.recorder:
            self.recorder.stop()
            
        if not self.dry_run and self.order_manager:
            try:
                # Cancel all orders
                logger.info("   Cancelling all open orders...")
                cancelled = await self._run_blocking(self.order_manager.cancel_all)
                logger.info(f"    Cancelled {cancelled} order(s)")
            except Exception as e:
                logger.error(f"    Error cancelling orders: {e}")
                # Try a second time with direct API call
                try:
                    logger.info("   Retrying order cancellation...")
                    cancelled = await self._run_blocking(self.order_manager.cancel_all)
                    logger.info(f"    Cancelled {cancelled} order(s) on retry")
                except Exception as e2:
                    logger.error(f"    Failed to cancel orders: {e2}")
                
        self._notify_bg(self.notifier.bot_stopped, "Shutdown")
        logger.info(" Goodbye!")

    async def _main_loop(self):
        """
        The main trading loop.
        Runs every LOOP_INTERVAL seconds.
        """
        logger.info(f" Entering main loop (interval: {config.LOOP_INTERVAL}s)")
        
        while self.running:
            try:
                loop_start = time.time()
                self.tick_count += 1
                
                # Latency tracking
                tick_start_time = time.time()
                
                # =============================================
                # STEP 1: UPDATE MARKET DATA
                # =============================================
                if not await self.market_data.update():
                    logger.warning(" Market data update failed, retrying...")
                    await asyncio.sleep(1)
                    continue

                # Optional: simulate small network/venue latency.
                sim_ms = float(getattr(config, 'SIMULATED_LATENCY_MS', 0.0) or 0.0)
                if sim_ms > 0:
                    await asyncio.sleep(sim_ms / 1000.0)
                
                # Update toxic guard with price
                self.toxic_guard.update_price(self.market_data.mid_price)
                
                # =============================================
                # STEP 1b: RECORD SNAPSHOT (FIX: Record before Risk Check)
                # =============================================
                # We record here to capture data even if the bot is halted by Risk Manager.
                if self.recorder:
                    # Regime might be stale here if we haven't evaluated it yet, 
                    # but for raw data capture (Market Data) it's fine.
                    # We pass the PREVIOUS regime to keep it simple, or "ANALYZING"
                    self.recorder.snapshot(
                        market_data=self.market_data,
                        regime=self.last_regime.value if self.last_regime else "INIT",
                        strategy_decision=None # No decision yet
                    )

                # =============================================
                # STEP 2: GET CURRENT STATE
                # =============================================
                if not self.dry_run:
                    position = await self._run_blocking(self.order_manager.get_position)
                    position_size = position['size']
                    current_entry_price = float(position.get('entryPx', 0.0))
                    position_usd = position_size * self.market_data.mid_price
                    account_value = await self._run_blocking(self.order_manager.get_account_value)
                    unrealized = float(position.get('unrealized_pnl', 0.0))
                else:
                    # PAPER TRADING MODE
                    position_size = self.sim_position_size
                    current_entry_price = self.sim_entry_price
                    position_usd = position_size * self.market_data.mid_price
                    
                    # Calculate Unrealized PnL
                    if abs(position_size) > 0:
                        if position_size > 0:
                            unrealized = (self.market_data.mid_price - self.sim_entry_price) * position_size
                        else:
                            unrealized = (self.sim_entry_price - self.market_data.mid_price) * abs(position_size)
                        
                        # DEDUCT FEES: "Mark to Market" requires assuming we close NOW (Taker Fee)
                        # This avoids the "green until I close" illusion.
                        exit_fee = abs(position_usd) * self.fee_guard.taker_fee
                        unrealized -= exit_fee
                    else:
                        unrealized = 0.0
                        
                    account_value = self.sim_cash + unrealized
                    
                    # Log sim status occasionally
                    if self.tick_count % 100 == 0:
                        pass # removed hack
                
                # =============================================
                # STEP 3: RISK CHECK
                # =============================================
                can_trade = self.risk_manager.update(account_value, position_usd)
                
                if not can_trade:
                    logger.warning(" Risk manager says NO TRADE")
                    
                    if not self.dry_run:
                        await self._run_blocking(self.order_manager.cancel_all)
                        await self._run_blocking(self.order_manager.market_close_position)
                    
                    await asyncio.sleep(60)  # Long sleep when killed
                    continue
                
                # =============================================
                # STEP 3b: ROTTEN FISH CHECK 
                # =============================================
                # Check if holding a losing position too long
                if not self.dry_run and abs(position_usd) > 10:
                    unrealized_pnl = position.get('unrealized_pnl', 0)
                    pnl_pct = unrealized_pnl / abs(position_usd) if position_usd != 0 else 0
                    
                    if self.risk_manager.check_position_time(pnl_pct):
                        logger.warning(" Rotten fish! Closing stale losing position...")
                        await self._run_blocking(self.order_manager.cancel_all)
                        await self._run_blocking(self.order_manager.market_close_position)
                        self._notify_bg(
                            self.notifier.send,
                            " Rotten fish closed (forced close)",
                            level=AlertLevel.WARNING,
                            title="Risk: Rotten Fish",
                            fields={"Reason": "Held losing position too long"},
                            key="risk_rotten_fish",
                            urgent=True,
                        )
                        await asyncio.sleep(2)  # Brief pause after forced close
                        continue
                
                # =============================================
                # STEP 4: EVALUATE REGIME
                # =============================================
                regime, metadata = self.regime_engine.evaluate(
                    volatility=self.market_data.volatility,
                    imbalance=getattr(self.market_data, 'depth_imbalance', self.market_data.imbalance),
                    current_position_usd=position_usd,
                    max_position_usd=config.MAX_POSITION_USD
                )

                # Hard safety: don't trade into a synthetic/one-sided book.
                # If we have open orders, cancel them so we don't get picked off.
                if getattr(self.market_data, 'is_synthetic_book', False):
                    logger.warning(" One-sided/synthetic book detected; cancelling orders and skipping tick")
                    if not self.dry_run:
                        await self._run_blocking(self.order_manager.cancel_all)
                    await asyncio.sleep(config.LOOP_INTERVAL)
                    continue

                # Provide equity context for sizing logic (used by StrategyEngine compounding).
                metadata["account_value"] = float(account_value)
                
                # Log regime changes
                if regime != self.last_regime:
                    self._notify_bg(self.notifier.regime_change, self.last_regime.value, regime.value)
                    
                    # Flush inventory on regime switch (Hunter -> Farmer)
                    if self.last_regime in [TradingRegime.HUNTER_LONG, TradingRegime.HUNTER_SHORT]:
                        if regime in [TradingRegime.FARMER_NEUTRAL, TradingRegime.FARMER_SKEW_LONG, TradingRegime.FARMER_SKEW_SHORT]:
                            if abs(position_usd) > 50:  # Significant position
                                logger.info(" Flushing inventory on regime switch...")
                                if not self.dry_run:
                                    await self._run_blocking(self.order_manager.market_close_position)
                    
                    self.last_regime = regime
                
                # =============================================
                # STEP 4b: HUNTER STOP LOSS CHECK
                # =============================================
                if regime in [TradingRegime.HUNTER_LONG, TradingRegime.HUNTER_SHORT]:
                    # Initialize entry price if just entered
                    if self.hunter_entry_price is None:
                        self.hunter_entry_price = self.market_data.mid_price
                    
                    # Calculate PnL %
                    pnl_pct = (self.market_data.mid_price - self.hunter_entry_price) / self.hunter_entry_price
                    if regime == TradingRegime.HUNTER_SHORT:
                        pnl_pct = -pnl_pct
                    
                    # Check stop loss (-0.5%)
                    if pnl_pct < -0.005:
                        logger.warning(f" Hunter stop loss triggered! PnL: {pnl_pct*100:.2f}%")
                        if not self.dry_run:
                            await self._run_blocking(self.order_manager.market_close_position)
                        self.hunter_entry_price = None
                        await asyncio.sleep(5)
                        continue
                else:
                    self.hunter_entry_price = None

                # =============================================
                # STEP 5: TOXIC FLOW CHECK
                # =============================================
                # Check if safe to trade each side
                buy_safe, buy_reason = self.toxic_guard.is_safe_to_trade(
                    "BUY",
                    self.market_data.imbalance,
                    self.market_data.spread_pct,
                    self.market_data.volatility_short
                )
                
                sell_safe, sell_reason = self.toxic_guard.is_safe_to_trade(
                    "SELL",
                    self.market_data.imbalance,
                    self.market_data.spread_pct,
                    self.market_data.volatility_short
                )
                
                # =============================================
                # STEP 5b: FETCH FUNDING RATE (Periodic)
                # =============================================
                # Fetch funding rate every ~60 ticks (6 seconds at 100ms loop)
                funding_rate = 0.0
                if getattr(config, 'FUNDING_BIAS_ENABLED', True) and self.tick_count % 60 == 0:
                    try:
                        funding_rate = await self.market_data.fetch_funding_rate()
                    except Exception as e:
                        logger.debug(f"Failed to fetch funding rate: {e}")
                else:
                    funding_rate = getattr(self.market_data, 'funding_rate', 0.0)
                
                # =============================================
                # STEP 5c: UPDATE QUEUE POSITIONS
                # =============================================
                if not self.dry_run and self.order_manager:
                    # Sync exchange metadata to strategy engine for accurate rounding
                    self.strategy_engine.tick_size = getattr(self.order_manager, 'tick_size', 0.0001)
                    self.strategy_engine.price_decimals = getattr(self.order_manager, 'price_decimals', 2)
                    
                    self.order_manager.update_queue_positions(
                        self.market_data.bids,
                        self.market_data.asks
                    )

                # =============================================
                # STEP 6: GENERATE STRATEGY (Orderbook-Driven)
                # =============================================
                # Pass full L2 orderbook for deep analysis
                decision = self.strategy_engine.generate_orders(
                    regime=regime,
                    metadata=metadata,
                    mid_price=self.market_data.mid_price,
                    bids=self.market_data.bids,  # Full L2 orderbook
                    asks=self.market_data.asks,  # Full L2 orderbook
                    current_position_usd=position_usd,
                    volatility=self.market_data.volatility,
                    funding_rate=funding_rate  # NEW: Pass funding rate for skew
                )
                
                # Log orderbook analysis if available
                if self.tick_count % 10 == 0 and decision.analysis_summary:
                    logger.info(decision.analysis_summary)
                
                # =============================================
                # STEP 6b: FEE GUARD CHECK 
                # =============================================
                # Filter orders that won't be profitable after fees.
                # NOTE: Passive market-making should be evaluated as a *pair* (our quoted bid/ask)
                # with maker fees on both legs (post-only), not as maker-entry + taker-exit vs current BBO.
                best_bid = self.market_data.best_bid
                best_ask = self.market_data.best_ask

                # In HUNTER mode, the entry order is paired with a reduce-only TP.
                # FeeGuard should evaluate the entry against that TP target (not the current BBO),
                # and we should never drop reduce-only exit orders.
                hunter_tp_sell = next((o for o in decision.sell_orders if getattr(o, 'reduce_only', False)), None)
                hunter_tp_buy = next((o for o in decision.buy_orders if getattr(o, 'reduce_only', False)), None)

                is_farmer = regime in [
                    TradingRegime.FARMER_NEUTRAL,
                    TradingRegime.FARMER_SKEW_LONG,
                    TradingRegime.FARMER_SKEW_SHORT,
                ]

                allow_one_sided_flat = bool(
                    getattr(config, 'ALLOW_ONE_SIDED_QUOTES_WHEN_FLAT', False)
                    or (self.dry_run and getattr(config, 'ALLOW_ONE_SIDED_QUOTES_IN_DRY_RUN', True))
                )
                
                profitable_buys = []
                for order in decision.buy_orders:
                    if getattr(order, 'reduce_only', False):
                        profitable_buys.append(order)
                        continue

                    # In FARMER mode, we validate profitability at the *pair* level below.
                    if is_farmer and not getattr(order, 'is_aggressive', False):
                        profitable_buys.append(order)
                        continue

                    # Buy order: we enter at order.price, exit at best_ask
                    target_price = hunter_tp_sell.price if getattr(order, 'is_aggressive', False) and hunter_tp_sell else best_ask
                    is_profitable, expected_pnl = self.fee_guard.is_profitable(
                        entry_price=order.price,
                        target_price=target_price,
                        use_taker_entry=getattr(order, 'is_aggressive', False)
                    )
                    if is_profitable:
                        profitable_buys.append(order)
                    elif self.tick_count % 50 == 0:
                        logger.debug(f"FeeGuard rejected buy @ ${order.price:.4f} (pnl: {expected_pnl*100:.3f}%)")
                
                profitable_sells = []
                for order in decision.sell_orders:
                    if getattr(order, 'reduce_only', False):
                        profitable_sells.append(order)
                        continue

                    # In FARMER mode, we validate profitability at the *pair* level below.
                    if is_farmer and not getattr(order, 'is_aggressive', False):
                        profitable_sells.append(order)
                        continue

                    # Sell order: we enter at order.price, exit at best_bid
                    target_price = hunter_tp_buy.price if getattr(order, 'is_aggressive', False) and hunter_tp_buy else best_bid
                    is_profitable, expected_pnl = self.fee_guard.is_profitable(
                        entry_price=order.price,
                        target_price=target_price,
                        use_taker_entry=getattr(order, 'is_aggressive', False)
                    )
                    if is_profitable:
                        profitable_sells.append(order)
                    elif self.tick_count % 50 == 0:
                        logger.debug(f"FeeGuard rejected sell @ ${order.price:.4f} (pnl: {expected_pnl*100:.3f}%)")
                
                decision.buy_orders = profitable_buys
                decision.sell_orders = profitable_sells

                # FARMER mode: Pair-level profitability check for passive quotes.
                if is_farmer:
                    passive_buys = [o for o in decision.buy_orders if not getattr(o, 'is_aggressive', False) and not getattr(o, 'reduce_only', False)]
                    passive_sells = [o for o in decision.sell_orders if not getattr(o, 'is_aggressive', False) and not getattr(o, 'reduce_only', False)]

                    if passive_buys and passive_sells:
                        # Evaluate best (top) pair; keep/remove all passive quotes together.
                        bid_px = max(o.price for o in passive_buys)
                        ask_px = min(o.price for o in passive_sells)
                        ok, net = self.fee_guard.is_profitable_market_making(
                            bid_price=bid_px,
                            ask_price=ask_px,
                            maker_only=getattr(config, 'MAKER_ONLY', True),
                        )
                        if not ok:
                            decision.buy_orders = [o for o in decision.buy_orders if getattr(o, 'is_aggressive', False) or getattr(o, 'reduce_only', False)]
                            decision.sell_orders = [o for o in decision.sell_orders if getattr(o, 'is_aggressive', False) or getattr(o, 'reduce_only', False)]
                            decision.reason = (decision.reason or "").rstrip() + f" | FeeGuardMM net={net*100:.3f}%"
                            if self.tick_count % 50 == 0:
                                logger.debug(
                                    f"FeeGuard rejected passive MM quotes (bid={bid_px:.4f}, ask={ask_px:.4f}) net={net*100:.3f}%"
                                )
                    else:
                        # If we only have one passive side while flat, don't open a directional position.
                        if (not allow_one_sided_flat) and abs(position_usd) < 1.0:
                            decision.buy_orders = [o for o in decision.buy_orders if getattr(o, 'reduce_only', False)]
                            decision.sell_orders = [o for o in decision.sell_orders if getattr(o, 'reduce_only', False)]
                
                # Filter out unsafe orders (toxic flow)
                if not buy_safe:
                    # Deep-wick bid logic: if enabled, allow a single bid far below mid even if toxic
                    allow_deep_wick = getattr(config, 'ALLOW_DEEP_WICK_BIDS', False)
                    deep_wick_offset_bps = getattr(config, 'DEEP_WICK_BID_OFFSET_BPS', 30.0)
                    
                    if allow_deep_wick and decision.buy_orders:
                        # Only keep the lowest bid, move it deep below mid
                        min_bid = min(decision.buy_orders, key=lambda o: o.price)
                        deep_wick_price = self.market_data.mid_price * (1 - deep_wick_offset_bps / 10000)
                        min_bid.price = round(deep_wick_price, 4)
                        min_bid.reason += f" | deep-wick toxic bid ({deep_wick_offset_bps}bps below mid)"
                        decision.buy_orders = [min_bid]
                        decision.reason = (decision.reason or "").rstrip() + f" | BUY deep-wick: {buy_reason}"
                    else:
                        # NOTE: Preserve reduce_only orders even if toxic (to allow closing positions)
                        decision.buy_orders = [o for o in decision.buy_orders if getattr(o, 'reduce_only', False)]
                        decision.reason = (decision.reason or "").rstrip() + f" | BUY blocked (non-exit): {buy_reason}"
                
                if not sell_safe:
                    # NOTE: Always allow reduce_only sells during crash/pump
                    decision.sell_orders = [o for o in decision.sell_orders if getattr(o, 'reduce_only', False)]
                    decision.reason = (decision.reason or "").rstrip() + f" | SELL blocked (non-exit): {sell_reason}"

                # If toxic flow removed one side and we're flat, avoid one-sided quoting (directional exposure).
                if is_farmer and (not allow_one_sided_flat) and abs(position_usd) < 1.0:
                    if decision.buy_orders and not decision.sell_orders:
                        decision.buy_orders = [o for o in decision.buy_orders if getattr(o, 'reduce_only', False)]
                    if decision.sell_orders and not decision.buy_orders:
                        decision.sell_orders = [o for o in decision.sell_orders if getattr(o, 'reduce_only', False)]

                # =============================================
                # STEP 6c: HARD POSITION CLAMP (P0)
                # =============================================
                decision = self._apply_position_clamp(decision, position_usd)
                
                # =============================================
                # STEP 6d: HFT IMPROVEMENTS - Maker Ratio & Adverse Selection
                # =============================================
                if not self.dry_run:
                    # Check maker ratio and adjust spreads if needed
                    should_widen, mult = self.order_manager.should_widen_spread()
                    self.strategy_engine.update_maker_ratio_adjustment(should_widen, mult)
                    
                    # Cancel orders at risk of adverse selection (every tick)
                    adverse_cancelled = await self._run_blocking(
                        self.order_manager.cancel_adverse_orders,
                        self.market_data.mid_price
                    )
                    
                    # Handle partial fills (every 10 ticks)
                    if self.tick_count % 10 == 0:
                        await self._run_blocking(
                            self.order_manager.handle_partial_fills,
                            self.market_data.bids,
                            self.market_data.asks,
                            self.market_data.mid_price
                        )
                
                # =============================================
                # STEP 7: EXECUTE ORDERS
                # =============================================
                if self.dry_run:
                    # Just log what we would do
                    self._log_dry_run(regime, decision)
                    # Simulate Fills (Paper Trading)
                    self._simulate_fills(decision)
                else:
                    # Round orders to valid precision
                    buy_orders = [o for o in (self.strategy_engine.round_order(x, self.coin) for x in decision.buy_orders) if o]
                    sell_orders = [o for o in (self.strategy_engine.round_order(x, self.coin) for x in decision.sell_orders) if o]
                    
                    # Track strategy calculation time
                    strategy_calc_end = time.time()
                    self.strategy_engine.latency_metrics.record(
                        'strategy_calc_ms', 
                        (strategy_calc_end - tick_start_time) * 1000
                    )
                    
                    # Execute with diffing
                    order_submit_start = time.time()
                    placed, cancelled = await self._run_blocking(
                        self.order_manager.execute_strategy,
                        buy_orders=buy_orders,
                        sell_orders=sell_orders,
                        cancel_all=decision.should_cancel_all
                    )
                    order_submit_end = time.time()
                    self.strategy_engine.latency_metrics.record(
                        'order_submit_ms',
                        (order_submit_end - order_submit_start) * 1000
                    )
                    
                    # Latency check - full tick latency
                    tick_latency = (time.time() - tick_start_time) * 1000
                    self.strategy_engine.latency_metrics.record('tick_to_trade_ms', tick_latency)
                    
                    # Use config threshold (default 500ms for testnet)
                    latency_threshold = getattr(config, 'LATENCY_WARN_TICK_TO_TRADE_MS', 500.0)
                    if tick_latency > latency_threshold:
                        logger.warning(f" High tick-to-order latency: {tick_latency:.0f}ms")
                    elif self.tick_count % 100 == 0:
                        logger.info(f" Tick latency: {tick_latency:.0f}ms")
                
                # =============================================
                # STEP 8: TRACK POSITION CHANGES & NOTIFY
                # =============================================
                new_position = position_size
                
                # Check for changes (Notifications)
                if abs(new_position - self.last_position) > 1e-6:
                    change = new_position - self.last_position
                    price = self.market_data.mid_price
                    
                    # Helper to calc PnL
                    def calc_pnl(size, entry, exit_px, side):
                        if entry == 0: return 0.0
                        if side == "LONG":
                            return (exit_px - entry) * size
                        else:
                            return (entry - exit_px) * size

                    # Case 1: Full Close (X -> 0)
                    if abs(new_position) < 1e-6:
                        closed_size = abs(self.last_position)
                        side = "LONG" if self.last_position > 0 else "SHORT"
                        pnl = calc_pnl(closed_size, self.last_entry_price, price, side)
                        pnl_pct = (pnl / (closed_size * self.last_entry_price)) if (closed_size * self.last_entry_price) else 0
                        
                        status = self._get_status_snapshot(regime.value, account_value)
                        self.notifier.trade_closed(
                            coin=self.coin,
                            side=side,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason="Full Exit",
                            status=status
                        )
                    
                    # Case 2: Flip (X -> -Y)
                    elif (self.last_position > 0 and new_position < 0) or (self.last_position < 0 and new_position > 0):
                        # Close Old
                        closed_size = abs(self.last_position)
                        side = "LONG" if self.last_position > 0 else "SHORT"
                        pnl = calc_pnl(closed_size, self.last_entry_price, price, side)
                        pnl_pct = (pnl / (closed_size * self.last_entry_price)) if (closed_size * self.last_entry_price) else 0
                        
                        status = self._get_status_snapshot(regime.value, account_value)
                        self.notifier.trade_closed(
                            coin=self.coin,
                            side=side,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason="Position Flip",
                            status=status
                        )
                        
                        # Open New
                        self.notifier.trade_opened(
                            coin=self.coin,
                            side="BUY" if new_position > 0 else "SELL",
                            price=price,
                            size=abs(new_position),
                            reason=regime.value
                        )

                    # Case 3: Adjustment (X -> X+delta)
                    else:
                        # Opened or Adjusted
                        action = "BUY" if change > 0 else "SELL"
                        self.notifier.trade_opened(
                            coin=self.coin,
                            side=action,
                            price=price,
                            size=abs(change),
                            reason=regime.value
                        )

                # Journaling Logic (Live Only)
                if not self.dry_run:
                    # Position opened - track entry for spread capture
                    if abs(self.last_position) < 0.001 and abs(new_position) > 0.001:
                        context = create_market_context(self.market_data, regime.value)
                        side = "BUY" if new_position > 0 else "SELL"
                        self._last_entry_side = side
                        self._last_entry_price = self.market_data.mid_price
                        self.journal.open_position(
                            coin=self.coin,
                            side=side,
                            entry_price=self.market_data.mid_price,
                            size=abs(new_position),
                            context=context,
                            reason=regime.value
                        )
                    
                    # Position closed - record spread capture
                    elif abs(self.last_position) > 0.001 and abs(new_position) < 0.001:
                        context = create_market_context(self.market_data, regime.value)
                        record = self.journal.close_position(
                            exit_price=self.market_data.mid_price,
                            context=context,
                            reason="Position closed"
                        )
                        if record:
                            self.risk_manager.record_trade(record.pnl_usd)
                            # Record spread capture for analysis
                            if hasattr(self, '_last_entry_side') and hasattr(self, '_last_entry_price'):
                                self.order_manager.record_spread_capture(
                                    entry_side=self._last_entry_side,
                                    entry_price=self._last_entry_price,
                                    exit_price=self.market_data.mid_price,
                                    size=abs(self.last_position)
                                )
                
                # Update State
                self.last_position = new_position
                self.last_entry_price = current_entry_price
                
                # =============================================
                # STEP 8b: TRACK FILLS (P1 Fix - Fill-based journaling)
                # =============================================
                if not self.dry_run and self.tick_count % 100 == 0:  # Every ~10 seconds
                    await self._record_recent_fills()
                
                # =============================================
                # STEP 9: PERIODIC LOGGING
                # =============================================
                if self.tick_count % 30 == 0:  # Every ~60 seconds
                    self._log_status(regime, position_usd, account_value)
                
                # HFT metrics logging (every 5 minutes)
                if not self.dry_run and self.tick_count % 300 == 0:
                    self._log_hft_metrics()
                
                # =============================================
                # STEP 10: SLEEP
                # =============================================
                elapsed = time.time() - loop_start
                
                # Minimum sleep reduced from 0.1 to 0.01 for HFT responsiveness
                sleep_time = max(0.01, config.LOOP_INTERVAL - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f" Loop error: {e}")
                self._notify_bg(self.notifier.error, str(e), "Main loop")
                await asyncio.sleep(5)

    def _log_dry_run(self, regime: TradingRegime, decision):
        """Log what we would do in dry run mode."""
        icon = self.regime_engine.get_regime_icon()
        
        # Format orders with reasons
        buy_str = ", ".join([f"${o.price:.4f}" for o in decision.buy_orders[:2]])
        sell_str = ", ".join([f"${o.price:.4f}" for o in decision.sell_orders[:2]])
        
        logger.info(
            f"{icon} {regime.value} | "
            f"Mid: ${self.market_data.mid_price:.4f} | "
            f"Bids: [{buy_str}] | "
            f"Asks: [{sell_str}]"
        )

        # If we have no orders, show the reason at INFO (common in dry-run).
        if (not decision.buy_orders and not decision.sell_orders) and getattr(decision, 'reason', ''):
            logger.info(f"    No orders: {decision.reason}")
        
        # Show orderbook analysis every few ticks
        if decision.analysis_summary and self.tick_count % 5 == 0:
            logger.info(f"   {decision.analysis_summary}")
        
        # Show order reasons
        if decision.reason:
            logger.debug(f"   Strategy: {decision.reason}")

    def _simulate_fills(self, decision):
        """
        Simulate order fills for Paper Trading.

        If USE_REALISTIC_SIM is enabled, uses the SimulatedVenue which models:
        - Queue position and consumption-based fills
        - Order ack / cancel latency
        - Partial fills
        - Taker depth-walk slippage
        - Adverse selection penalty
        - Stochastic rejections

        Otherwise falls back to the legacy random-fill logic.
        """
        now = time.time()
        best_ask = self.market_data.best_ask
        best_bid = self.market_data.best_bid
        bids = self.market_data.bids
        asks = self.market_data.asks

        # --------------------------------------------------------
        # Realistic Venue Mode
        # --------------------------------------------------------
        if self.sim_venue is not None:
            # 1. Update venue with new L2 snapshot (generates fills from queue)
            fills, rejects = self.sim_venue.on_tick(
                bids=bids,
                asks=asks,
                mid_price=self.market_data.mid_price,
                now=now,
            )

            # 2. Submit new desired orders (may generate immediate taker fills)
            margin_avail = self.sim_cash  # Simplified margin check
            new_fills, new_rejects = self.sim_venue.submit_orders(
                buy_orders=decision.buy_orders,
                sell_orders=decision.sell_orders,
                now=now,
                margin_available=margin_avail,
            )
            fills.extend(new_fills)
            rejects.extend(new_rejects)

            # 3. Check for taker orders (crossing spread) and fill them via depth-walk
            for order in decision.buy_orders:
                if order.price >= best_ask:
                    slip_bps = getattr(config, 'SIM_TAKER_SLIPPAGE_EXTRA_BPS', 0.5)
                    fill_px, filled_sz = compute_taker_fill_price("BUY", order.size, bids, asks, slip_bps)
                    if filled_sz > 0:
                        fills.append(FillEvent(
                            oid="taker",
                            side=OrderSide.BUY,
                            price=fill_px,
                            size=filled_sz,
                            remaining=order.size - filled_sz,
                            is_maker=False,
                            timestamp=now,
                        ))

            for order in decision.sell_orders:
                if order.price <= best_bid:
                    slip_bps = getattr(config, 'SIM_TAKER_SLIPPAGE_EXTRA_BPS', 0.5)
                    fill_px, filled_sz = compute_taker_fill_price("SELL", order.size, bids, asks, slip_bps)
                    if filled_sz > 0:
                        fills.append(FillEvent(
                            oid="taker",
                            side=OrderSide.SELL,
                            price=fill_px,
                            size=filled_sz,
                            remaining=order.size - filled_sz,
                            is_maker=False,
                            timestamp=now,
                        ))

            # 4. Apply fills to sim state
            for fill in fills:
                self._apply_sim_fill(fill)

            # 5. Log rejects
            for rej in rejects:
                logger.warning(f" SIM REJECT: {rej.reason.value}")

            return

        # --------------------------------------------------------
        # Legacy Random Fill Mode (fallback)
        # --------------------------------------------------------
        self._simulate_fills_legacy(decision)

    def _apply_sim_fill(self, fill: FillEvent):
        """Apply a FillEvent to sim_position, sim_cash, sim_realized_pnl."""
        fill_price = fill.price
        fill_size = fill.size
        is_buy = fill.side == OrderSide.BUY
        is_maker = fill.is_maker

        fee_rate = self.fee_guard.maker_fee if is_maker else self.fee_guard.taker_fee
        fee = fill_price * fill_size * fee_rate

        # Apply adverse selection penalty to effective fill price
        if is_maker and fill.adverse_penalty_bps > 0:
            penalty = fill_price * (fill.adverse_penalty_bps / 10000.0)
            if is_buy:
                fill_price += penalty  # Worse for buyer
            else:
                fill_price -= penalty  # Worse for seller

        if is_buy:
            # Buying
            if self.sim_position_size >= 0:
                # Adding to long
                total_cost = (self.sim_position_size * self.sim_entry_price) + (fill_size * fill_price)
                total_size = self.sim_position_size + fill_size
                self.sim_entry_price = total_cost / total_size if total_size > 0 else 0
            else:
                # Closing short
                closing_part = min(abs(self.sim_position_size), fill_size)
                if closing_part > 0:
                    pnl = (self.sim_entry_price - fill_price) * closing_part
                    self.sim_realized_pnl += pnl
                    self.sim_cash += pnl
                    self.risk_manager.record_trade(pnl)
                if fill_size > abs(self.sim_position_size):
                    self.sim_entry_price = fill_price

            self.sim_position_size += fill_size
        else:
            # Selling
            if self.sim_position_size <= 0:
                # Adding to short
                total_cost = (abs(self.sim_position_size) * self.sim_entry_price) + (fill_size * fill_price)
                total_size = abs(self.sim_position_size) + fill_size
                self.sim_entry_price = total_cost / total_size if total_size > 0 else 0
            else:
                # Closing long
                closing_part = min(self.sim_position_size, fill_size)
                if closing_part > 0:
                    pnl = (fill_price - self.sim_entry_price) * closing_part
                    self.sim_realized_pnl += pnl
                    self.sim_cash += pnl
                    self.risk_manager.record_trade(pnl)
                if fill_size > self.sim_position_size:
                    self.sim_entry_price = fill_price

            self.sim_position_size -= fill_size

        self.sim_cash -= fee
        self.sim_fees_paid += fee
        self.sim_fill_count += 1

        fill_type = "MAKER" if is_maker else "TAKER"
        side_icon = "" if is_buy else ""
        side_str = "BUY" if is_buy else "SELL"
        logger.info(
            f"{side_icon} SIM FILL ({fill_type}): {side_str} {fill_size:.4f} @ ${fill.price:.4f} "
            f"| Pos: {self.sim_position_size:.4f} | Adverse: {fill.adverse_penalty_bps:.1f}bps"
        )

    def _simulate_fills_legacy(self, decision):
        """
        Legacy random-fill simulation (fallback when USE_REALISTIC_SIM=False).
        """
        best_ask = self.market_data.best_ask
        best_bid = self.market_data.best_bid

        for order in decision.buy_orders:
            fill_price = 0.0

            # Crossing the spread (Taker)
            if order.price >= best_ask:
                fill_price = best_ask

            if fill_price > 0:
                cost = fill_price * order.size
                fee = cost * self.fee_guard.taker_fee

                if self.sim_position_size >= 0:
                    total_cost = (self.sim_position_size * self.sim_entry_price) + (order.size * fill_price)
                    total_size = self.sim_position_size + order.size
                    self.sim_entry_price = total_cost / total_size if total_size > 0 else 0
                else:
                    closing_part = min(abs(self.sim_position_size), order.size)
                    if closing_part > 0:
                        pnl = (self.sim_entry_price - fill_price) * closing_part
                        self.sim_realized_pnl += pnl
                        self.sim_cash += pnl
                        self.risk_manager.record_trade(pnl)
                    if order.size > abs(self.sim_position_size):
                        self.sim_entry_price = fill_price

                self.sim_position_size += order.size
                self.sim_cash -= fee
                self.sim_fees_paid += fee
                self.sim_fill_count += 1
                logger.info(f" SIM FILL (TAKER): BUY {order.size} @ ${fill_price:.2f} | Pos: {self.sim_position_size:.4f}")
                break

            # Passive Fill Probability (Maker)
            if best_bid > 0 and best_ask > 0 and order.price < best_ask:
                diff_pct = abs(order.price - best_bid) / best_bid
                if diff_pct <= (config.SIM_MAKER_THRESHOLD_BPS / 10000):
                    import random
                    if random.random() < config.SIM_MAKER_FILL_PROB:
                        fill_price = order.price

                        if self.sim_position_size >= 0:
                            total_cost = (self.sim_position_size * self.sim_entry_price) + (order.size * fill_price)
                            total_size = self.sim_position_size + order.size
                            self.sim_entry_price = total_cost / total_size if total_size > 0 else 0
                        else:
                            closing_part = min(abs(self.sim_position_size), order.size)
                            if closing_part > 0:
                                pnl = (self.sim_entry_price - fill_price) * closing_part
                                self.sim_realized_pnl += pnl
                                self.sim_cash += pnl
                                self.risk_manager.record_trade(pnl)
                            if order.size > abs(self.sim_position_size):
                                self.sim_entry_price = fill_price

                        self.sim_position_size += order.size
                        fee = fill_price * order.size * self.fee_guard.maker_fee
                        self.sim_cash -= fee
                        self.sim_fees_paid += fee
                        self.sim_fill_count += 1
                        logger.info(f" SIM FILL (MAKER): BUY {order.size} @ ${fill_price:.2f} | Pos: {self.sim_position_size:.4f}")
                        break

        for order in decision.sell_orders:
            fill_price = 0.0

            if order.price <= best_bid:
                fill_price = best_bid

            if fill_price > 0:
                cost = fill_price * order.size
                fee = cost * self.fee_guard.taker_fee

                if self.sim_position_size <= 0:
                    total_cost = (abs(self.sim_position_size) * self.sim_entry_price) + (order.size * fill_price)
                    total_size = abs(self.sim_position_size) + order.size
                    self.sim_entry_price = total_cost / total_size if total_size > 0 else 0
                else:
                    closing_part = min(self.sim_position_size, order.size)
                    if closing_part > 0:
                        pnl = (fill_price - self.sim_entry_price) * closing_part
                        self.sim_realized_pnl += pnl
                        self.sim_cash += pnl
                        self.risk_manager.record_trade(pnl)
                    if order.size > self.sim_position_size:
                        self.sim_entry_price = fill_price

                self.sim_position_size -= order.size
                self.sim_cash -= fee
                self.sim_fees_paid += fee
                self.sim_fill_count += 1
                logger.info(f" SIM FILL (TAKER): SELL {order.size} @ ${fill_price:.2f} | Pos: {self.sim_position_size:.4f}")
                break

            if best_ask > 0 and best_bid > 0 and order.price > best_bid:
                diff_pct = abs(order.price - best_ask) / best_ask
                if diff_pct <= (config.SIM_MAKER_THRESHOLD_BPS / 10000):
                    import random
                    if random.random() < config.SIM_MAKER_FILL_PROB:
                        fill_price = order.price

                        if self.sim_position_size <= 0:
                            total_cost = (abs(self.sim_position_size) * self.sim_entry_price) + (order.size * fill_price)
                            total_size = abs(self.sim_position_size) + order.size
                            self.sim_entry_price = total_cost / total_size if total_size > 0 else 0
                        else:
                            closing_part = min(self.sim_position_size, order.size)
                            if closing_part > 0:
                                pnl = (fill_price - self.sim_entry_price) * closing_part
                                self.sim_realized_pnl += pnl
                                self.sim_cash += pnl
                                self.risk_manager.record_trade(pnl)
                            if order.size > self.sim_position_size:
                                self.sim_entry_price = fill_price

                        self.sim_position_size -= order.size
                        fee = fill_price * order.size * self.fee_guard.maker_fee
                        self.sim_cash -= fee
                        self.sim_fees_paid += fee
                        self.sim_fill_count += 1
                        logger.info(f" SIM FILL (MAKER): SELL {order.size} @ ${fill_price:.2f} | Pos: {self.sim_position_size:.4f}")
                        break

    def _get_status_snapshot(self, regime_val: str, account_value: float) -> dict:
        """Get current status snapshot for notifications."""
        if self.dry_run:
            realized = self.sim_realized_pnl
            pos_size = self.sim_position_size
            entry = self.sim_entry_price
            fees_paid = self.sim_fees_paid
        else:
            realized = self.risk_manager.daily_pnl
            pos_size = self.last_position
            entry = self.risk_manager.position_entry_price
            fees_paid = 0.0

        # Floating PnL
        floating = 0.0
        if abs(pos_size) > 0 and entry > 0:
            mid = self.market_data.mid_price
            if pos_size > 0:
                floating = (mid - entry) * pos_size
            else:
                floating = (entry - mid) * abs(pos_size)
            
            if self.dry_run:
                floating -= (abs(pos_size) * mid * self.fee_guard.taker_fee)

        total_pnl = realized + floating - fees_paid
        
        return {
            "regime": regime_val,
            "account": f"${account_value:.2f}",
            "realized": f"${realized:+.2f}",
            "floating": f"${floating:+.2f}",
            "fees": f"${fees_paid:.2f}",
            "total_pnl": f"${total_pnl:+.2f}",
            "daily_pnl": f"${self.risk_manager.daily_pnl:+.2f}",
            "weekly_pnl": f"${self.risk_manager.weekly_pnl:+.2f}",
            "monthly_pnl": f"${self.risk_manager.monthly_pnl:+.2f}"
        }

    def _log_status(self, regime: TradingRegime, position_usd: float, account_value: float):
        """Log periodic status update."""
        icon = self.regime_engine.get_regime_icon()
        risk_line = self.risk_manager.get_status_line()
        
        # Calculate PnL Breakdown
        if self.dry_run:
            realized = self.sim_realized_pnl
            pos_size = self.sim_position_size
            entry = self.sim_entry_price
            fees_paid = self.sim_fees_paid
        else:
            realized = self.risk_manager.daily_pnl
            pos_size = self.last_position # Approximate from last update
            entry = self.risk_manager.position_entry_price
            fees_paid = 0.0

        # Floating (Unrealized) PnL (Dry-run uses liquidation value: includes estimated exit taker fee)
        floating = 0.0
        exit_fee_est = 0.0
        if abs(pos_size) > 0 and entry > 0:
            if pos_size > 0:  # Long
                floating = (self.market_data.mid_price - entry) * pos_size
            else:  # Short
                floating = (entry - self.market_data.mid_price) * abs(pos_size)

            if self.dry_run:
                exit_fee_est = abs(position_usd) * self.fee_guard.taker_fee
                floating -= exit_fee_est

        total_pnl = realized + floating - fees_paid
        
        logger.info(
            f" STATUS "
        )
        logger.info(
            f"{icon} Regime: {regime.value} | "
            f"Pos: ${position_usd:+.2f} | "
            f"Account: ${account_value:.2f}"
        )
        pnl_icon = "" if total_pnl >= 0 else ""
        logger.info(
            f"   {pnl_icon} PnL: Realized ${realized:+.2f} | "
            f"Floating ${floating:+.2f} | "
            f"Total ${total_pnl:+.2f}"
        )
        logger.info(
            f"    PnL: Day ${self.risk_manager.daily_pnl:+.2f} | "
            f"Week ${self.risk_manager.weekly_pnl:+.2f} | "
            f"Month ${self.risk_manager.monthly_pnl:+.2f}"
        )
        if self.dry_run:
            logger.info(
                f"   FeesPaid ${fees_paid:.2f} | ExitFeeEst ${exit_fee_est:.2f} | Fills {self.sim_fill_count}"
            )
        logger.info(f"   {risk_line}")
        
        if not self.dry_run:
            stats = self.order_manager.get_stats()
            # Rolling fills/hour (based on fill timestamps recorded in _record_recent_fills)
            now = time.time()
            while self._fill_timestamps and (now - self._fill_timestamps[0]) > 3600:
                self._fill_timestamps.popleft()
            fills_last_hour = len(self._fill_timestamps)
            
            # Maker/taker ratio
            maker_ratio = self.order_manager.get_fill_ratio()
            
            logger.info(
                f"   Orders: {stats['active_orders']} active | "
                f"API: {stats['rate_limit_pct']:.0f}% used | "
                f"Maker: {maker_ratio:.0%}"
            )
            logger.info(f"   Fills: {fills_last_hour} / hr")
            
            # Log funding rate if enabled
            funding_rate = getattr(self.market_data, 'funding_rate', 0.0)
            if abs(funding_rate) > 0.00001:
                funding_pct = funding_rate * 100
                funding_icon = "" if funding_rate > 0 else ""
                logger.info(f"   {funding_icon} Funding: {funding_pct:+.4f}% (longs {'pay' if funding_rate > 0 else 'receive'})")
        logger.info(f"")

        # Periodic Discord Update
        current_time = time.time()
        if current_time - self.last_discord_update > self.discord_update_interval:
            daily_pnl = self.risk_manager.daily_pnl if not self.dry_run else self.sim_realized_pnl
            trades_count = self.journal.total_trades if not self.dry_run else 0 # Sim trades not meant for main journal yet
            
            # Use sim trades count if dry run (hacky but useful)
            if self.dry_run:
                # We don't easily track sim trade count here, just show 0 or estimated
                pass 

            status = self._get_status_snapshot(regime.value, account_value)
            self.notifier.status_update(
                pnl=daily_pnl,
                trades=trades_count,
                account_value=account_value,
                regime=regime.value,
                status=status,
            )
            self.last_discord_update = current_time
    
    def _log_hft_metrics(self):
        """Log detailed HFT performance metrics (every 5 minutes)."""
        logger.info(" HFT METRICS ")
        
        # Latency metrics
        latency_summary = self.strategy_engine.get_latency_summary()
        logger.info(
            f" Latency (avg/p99): "
            f"Strategy={latency_summary['strategy_calc_avg_ms']:.1f}/{latency_summary['strategy_calc_p99_ms']:.1f}ms | "
            f"Submit={latency_summary['order_submit_avg_ms']:.1f}/{latency_summary['order_submit_p99_ms']:.1f}ms | "
            f"Total={latency_summary['tick_to_trade_avg_ms']:.1f}/{latency_summary['tick_to_trade_p99_ms']:.1f}ms"
        )
        
        # Spread capture stats
        spread_stats = self.order_manager.get_spread_capture_stats()
        if spread_stats['count'] > 0:
            logger.info(
                f" Spread Capture: {spread_stats['count']} trades | "
                f"Avg={spread_stats['avg_spread_bps']:.2f}bps | "
                f"Total=${spread_stats['total_spread_usd']:.2f} | "
                f"Win%={spread_stats['positive_pct']:.0f}%"
            )
        
        # Maker ratio
        maker_ratio = self.order_manager.get_current_maker_ratio()
        target_ratio = self.order_manager.target_maker_ratio
        ratio_icon = "" if maker_ratio >= target_ratio else ""
        logger.info(
            f"{ratio_icon} Maker Ratio: {maker_ratio:.1%} (target: {target_ratio:.0%}) | "
            f"Maker Fills: {self.order_manager.maker_fills} | "
            f"Taker Fills: {self.order_manager.taker_fills}"
        )
        
        # Queue positions (if any active orders)
        if self.order_manager.queue_positions:
            avg_queue = sum(self.order_manager.queue_positions.values()) / len(self.order_manager.queue_positions)
            logger.info(f" Avg Queue Depth: {avg_queue:.2f} units ahead")
        
        logger.info("")
    
    async def _record_recent_fills(self):
        """
        Fetch and record recent fills for accurate tracking (P1 fix).
        This provides fill-based journaling instead of position-delta tracking.
        Also feeds fills into strategy engine for trade flow analysis.
        """
        try:
            fills = await self._run_blocking(self.order_manager.get_recent_fills, limit=50)
            
            for fill in fills:
                # Extract fill data from Hyperliquid format
                fill_id = fill.get('tid', str(fill.get('time', '')))
                side = "BUY" if fill.get('side') == 'B' else "SELL"
                price = float(fill.get('px', 0))
                size = float(fill.get('sz', 0))
                
                # Record to journal
                record = self.journal.record_fill(
                    fill_id=fill_id,
                    coin=fill.get('coin', self.coin),
                    side=side,
                    price=price,
                    size=size,
                    fee=float(fill.get('fee', 0)),
                    is_maker=fill.get('crossed', True) == False,  # crossed=False means maker
                    oid=fill.get('oid', ''),
                    timestamp=float(fill.get('time', 0)) / 1000 if fill.get('time') else None
                )

                if record is not None:
                    ts = float(record.timestamp or time.time())
                    self._fill_timestamps.append(ts)
                
                # Feed into strategy engine for trade flow analysis
                self.strategy_engine.record_fill(side, price, size)
                
        except Exception as e:
            logger.debug(f"Error recording fills: {e}")

    async def _shutdown(self):
        """Clean shutdown."""
        logger.info(" Shutting down...")
        
        # Save journal
        self.journal.force_save()
        
        # Cancel all orders
        if not self.dry_run and self.order_manager:
            try:
                await self._run_blocking(self.order_manager.cancel_all)
            except:
                pass
        
        # Send summary
        try:
            summary = self.risk_manager.get_summary()
            self._notify_bg(self.notifier.daily_summary, summary)
            self._notify_bg(self.notifier.bot_stopped, "Shutdown")
        except:
            pass
        
        logger.info(" Goodbye!")


# ENTRY POINT
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='HyperLiquid HFT Market Making Bot')
    parser.add_argument('--dry-run', action='store_true', help='Run without placing real orders')
    parser.add_argument('--analyze', action='store_true', help='Run AI analysis only')
    parser.add_argument('--capture', action='store_true', help='Capture market data for optimization')
    args = parser.parse_args()
    
    setup_logging()
    
    # Print banner
    print("""
    
             HYPERLIQUID HFT MARKET MAKING BOT                 
                                                               
       Strategy: Hybrid (Hunter + Farmer)                      
       Features: Regime Switching, Toxic Flow Protection       
                                                               
    
    """)
    
    if args.analyze:
        # Run analysis only
        from core.analyst import run_daily_analysis
        run_daily_analysis()
        return
    
    # Validate config
    if not args.dry_run:
        if not config.PRIVATE_KEY:
            logger.error(" PRIVATE_KEY not set in .env file!")
            logger.error("   Create a .env file with your API wallet private key")
            sys.exit(1)
    
    # Run bot
    try:
        bot = HyperLiquidBot(dry_run=args.dry_run, capture_data=args.capture)
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
