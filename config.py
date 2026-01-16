# config.py - all the settings for the bot
# change these in .env file or here directly

import os

from dotenv import load_dotenv

# Load environment variables from `.env` before reading any settings.
load_dotenv()

# If toxic flow blocks normal bids, allow quoting a "deep-wick" bid far below mid
# (for inventory/market presence). Offset is in bps (1 bps = 0.01%).
ALLOW_DEEP_WICK_BIDS = os.getenv("ALLOW_DEEP_WICK_BIDS", "False").lower() == "true"
DEEP_WICK_BID_OFFSET_BPS = float(os.getenv("DEEP_WICK_BID_OFFSET_BPS", "30.0"))

# API CONFIGURATION
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "")

# Network: Use TESTNET for testing, MAINNET for real trading
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"

# Use websocket for low-latency market data (recommended for HFT)
# Set to False to use REST polling instead
USE_WEBSOCKET = os.getenv("USE_WEBSOCKET", "True").lower() == "true"

# TRADING CONFIGURATION
# The coin to trade (e.g., "HYPE", "SOL", "BTC", "ETH")
COIN = os.getenv("COIN", "HYPE")

# Enforce maker-only (post-only) limit orders for normal quoting.
# This is strongly recommended for a market-making bot; if your order would cross,
# it should be rejected rather than paying taker fees.
MAKER_ONLY = os.getenv("MAKER_ONLY", "True").lower() == "true"

# Time-in-force for limit orders.
# Hyperliquid supports "Alo" (add liquidity only / post-only) and "Gtc".
MAKER_TIF = os.getenv("MAKER_TIF", "Alo")
DEFAULT_TIF = os.getenv("DEFAULT_TIF", "Gtc")

# Order sizing
ORDER_SIZE_USD = float(os.getenv("ORDER_SIZE_USD", "40"))  # Start at $40 (will compound)
MAX_POSITION_USD = float(os.getenv("MAX_POSITION_USD", "1000"))  # Allow 5x leverage on $200 account

# Leverage (1-50x on Hyperliquid)
LEVERAGE = int(os.getenv("LEVERAGE", "5"))

# When one side is blocked (e.g., by toxic flow), the bot can end up with only bids
# or only asks. By default we avoid one-sided quoting while flat to prevent taking
# accidental directional exposure. For debugging/dry-run it can be useful to allow.
ALLOW_ONE_SIDED_QUOTES_WHEN_FLAT = os.getenv("ALLOW_ONE_SIDED_QUOTES_WHEN_FLAT", "False").lower() == "true"
ALLOW_ONE_SIDED_QUOTES_IN_DRY_RUN = os.getenv("ALLOW_ONE_SIDED_QUOTES_IN_DRY_RUN", "True").lower() == "true"

# HFT MODE - 100+ trades per hour target
# Set to True for aggressive HFT (tight spreads, fast updates)
# Set to False for conservative mode (wider spreads, fewer trades)
HFT_MODE = os.getenv("HFT_MODE", "True").lower() == "true"

# STRATEGY PARAMETERS

# --- REGIME ENGINE (Brain) ---
# Volatility threshold to switch from Farmer to Hunter mode
# Increased to 0.0030 to prevent noise-based churning
VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", "0.0030"))  # 0.30%

# Imbalance thresholds for Hunter mode
# Require stronger conviction to enter taking modes
IMBALANCE_BULL_THRESHOLD = float(os.getenv("IMBALANCE_BULL_THRESHOLD", "0.75"))  # >75% = Bull
IMBALANCE_BEAR_THRESHOLD = float(os.getenv("IMBALANCE_BEAR_THRESHOLD", "0.25"))  # <25% = Bear

# Hysteresis: Need higher vol to ENTER hunter, lower to EXIT
VOLATILITY_EXIT_MULTIPLIER = float(os.getenv("VOL_EXIT_MULT", "0.7"))  # Exit at 70% of entry threshold

# Extreme imbalance overrides volatility check (default 0.80)
# Higher = stricter, requires stronger conviction to enter Hunter mode
IMBALANCE_EXTREME_THRESHOLD = float(os.getenv("IMBALANCE_EXTREME_THRESHOLD", "0.90"))

# --- FARMER MODE (Market Making - Orderbook Driven) ---
# Base spread is just afallback - actual spread is calculated from orderbook
# For 100+ trades/hour, we need TIGHT spreads to get filled frequently
# NOTE: Maker fee is 1bp. Round trip is 2bps. Spread MUST be > 2bps to profit.
BASE_SPREAD = float(os.getenv("BASE_SPREAD", "0.0003"))  # 3 bps (Aggressive HFT)
SKEW_FACTOR = float(os.getenv("SKEW_FACTOR", "0.0050"))  # Price shift per $100 inventory (Start stronger)

# NOTE: We do NOT use FARMER_LEVELS anymore - we place 1-2 orders at optimal levels
# determined by orderbook analysis, not a grid

# DYNAMIC CONFIG OVERRIDES (Auto-Optimizer)
# Load best parameters from the optimizer if they exist
import json
try:
    PARAMS_FILE = "data/best_params.json"
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, "r") as f:
            overrides = json.load(f)
            
            # Apply overrides with safety checks
            if 'vol_thresh' in overrides:
                VOLATILITY_THRESHOLD = float(overrides['vol_thresh'])
            
            if 'spread' in overrides:
                BASE_SPREAD = float(overrides['spread'])
                
            if 'skew' in overrides:
                SKEW_FACTOR = float(overrides['skew'])
                
            if 'imba_thresh' in overrides:
                it = float(overrides['imba_thresh'])
                IMBALANCE_BULL_THRESHOLD = it
                IMBALANCE_BEAR_THRESHOLD = 1.0 - it

            if 'ob_decay' in overrides:
                OB_OIMB_DECAY = float(overrides['ob_decay'])

            if 'press_scale' in overrides:
                VWAP_PRESSURE_SCALER = float(overrides['press_scale'])

            if 'skew_ent' in overrides:
                SKEW_MODE_ENTRY_THRESHOLD = float(overrides['skew_ent'])
                SKEW_MODE_EXIT_THRESHOLD = SKEW_MODE_ENTRY_THRESHOLD / 2

            if 'vol_exit_mult' in overrides:
                VOLATILITY_EXIT_MULTIPLIER = float(overrides['vol_exit_mult'])
                
        # print(f" Loaded optimized config: {overrides}") # Can't print in config
except Exception as e:
    pass # Silent fail, use defaults

# --- ORDERBOOK ANALYZER ---
# Wall detection: Level is a "wall" if size > WALL_THRESHOLD * average size
WALL_THRESHOLD = float(os.getenv("WALL_THRESHOLD", "3.0"))  # 3x average = wall

# Spread bounds (in basis points, 1 bps = 0.01%)
# Fee reality: maker 1bp + taker 3.5bp = 4.5bp round-trip minimum
# For profitable quoting, spread must exceed ~5-6 bps
MIN_SPREAD_BPS = float(os.getenv("MIN_SPREAD_BPS", "5.0"))   # 5.0bps minimum (Safer spread)
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "20.0"))  # Maximum 0.20% (tighter for HFT)
BASE_SPREAD_BPS = float(os.getenv("BASE_SPREAD_BPS", "3.5"))  # Target 3.5bps baseline

# HIGH-LATENCY SAFETY BUFFER
# With 2500ms API latency, price can move 5-15bps during roundtrip
# This buffer pushes orders further from mid to avoid POST_ONLY rejections
# Set to 0 for low-latency setups, 10+ for high-latency
LATENCY_SAFETY_BPS = float(os.getenv("LATENCY_SAFETY_BPS", "0.0"))

# Depth levels to analyze
ORDERBOOK_DEPTH = int(os.getenv("ORDERBOOK_DEPTH", "10"))  # Analyze top 10 levels

# --- HUNTER MODE (Trend Chasing) ---
# Aggression is now conviction-based (1-4 bps based on orderbook analysis)
HUNTER_BASE_AGGRESSION = float(os.getenv("HUNTER_AGGRESSION", "0.0001"))  # 0.01% (1 bps) base

# RISK MANAGEMENT
# Daily loss limit (percentage of account)
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))  # 5% (Safer - institutional standard)

# Max drawdown from daily high
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "0.10"))  # 10% (Hard Stop - tighter)

# Stop loss per trade (percentage)
# DISABLED for HFT (0.0). We use inventory management and skewed quoting instead.
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.0"))  # Disabled

# Rotten fish rule: Max time to hold a losing position (seconds)
MAX_HOLD_TIME_SECONDS = int(os.getenv("MAX_HOLD_TIME", "300"))  # 5 minutes


# FEE CONFIGURATION (Hyperliquid Tier 0)
# If True, FeeGuard will attempt to fetch your current user fee rates from the
# Hyperliquid API using WALLET_ADDRESS (fallbacks to values below on failure).
AUTO_FETCH_FEES = os.getenv("AUTO_FETCH_FEES", "False").lower() == "true"

# If True, allow maker fee to be negative (rebate). Most users without tier
# should keep this False.
ALLOW_MAKER_REBATE = os.getenv("ALLOW_MAKER_REBATE", "False").lower() == "true"

# Fallback/default fee rates (decimal pct). Can be overridden via env.
MAKER_FEE = float(os.getenv("MAKER_FEE", "0.0001"))   # 0.01%
TAKER_FEE = float(os.getenv("TAKER_FEE", "0.00035"))  # 0.035%

# Minimum spread required to trade (must cover fees)
MIN_SPREAD_PCT = float(os.getenv("MIN_SPREAD_PCT", "0.0003"))  # 0.03%

# Minimum expected NET profit (after fees) to keep an order.
# For HFT market-making, this must be small (a few bps) or you'll never quote.
# Example: 6 bps gross spread - 2 bps maker fees = ~4 bps net.
MIN_PROFIT_THRESHOLD = float(os.getenv("MIN_PROFIT", "0.00001"))  # 0.1bps net (Volume/Rebate focus)

# TOXIC FLOW PROTECTION
# Imbalance threshold to detect toxic flow
TOXIC_IMBALANCE_THRESHOLD = float(os.getenv("TOXIC_IMBALANCE", "0.75"))  # 75% (Relaxed for HFT)

# Max volatility in short window before pausing (crash detection)
MAX_SHORT_TERM_VOLATILITY = float(os.getenv("MAX_VOL_SHORT", "0.005"))  # 0.5% in 10 seconds

# Order staleness (rotten fish)
ORDER_STALE_SECONDS = int(os.getenv("ORDER_STALE_SEC", "10"))  # Cancel after 10s

# TICK-TO-TICK VELOCITY (Flash Crash Detection)
# In crypto, a 5% crash can happen in under 1 second. These settings provide
# sub-second protection that the 10-second crash detection can't catch.
TICK_VELOCITY_ENABLED = os.getenv("TICK_VELOCITY_ENABLED", "True").lower() == "true"
TICK_VELOCITY_WINDOW_MS = int(os.getenv("TICK_VELOCITY_WINDOW_MS", "500"))  # Check over 500ms
TICK_VELOCITY_THRESHOLD = float(os.getenv("TICK_VELOCITY_THRESHOLD", "0.01"))  # 1% move in window triggers

# WALL FLICKER PROTECTION (Anti-Spoofing)
# Spoofing: Large orders placed briefly to manipulate other traders.
# These settings require walls to persist before the bot trusts them.
WALL_FLICKER_ENABLED = os.getenv("WALL_FLICKER_ENABLED", "True").lower() == "true"
WALL_MIN_PERSISTENCE_SEC = float(os.getenv("WALL_MIN_PERSISTENCE_SEC", "2.0"))  # Wall must exist 2s
WALL_MIN_UPDATES = int(os.getenv("WALL_MIN_UPDATES", "3"))  # Wall must be seen in 3 snapshots

# LATENCY MONITORING
# Warning thresholds for latency monitoring (milliseconds)
# Note: API processing can spike to 2-3s during high volume periods
# Only warn on extreme outliers (>3s) since occasional spikes are normal
LATENCY_WARN_TICK_TO_TRADE_MS = float(os.getenv("LATENCY_WARN_TICK_MS", "3000.0"))
LATENCY_WARN_ORDER_SUBMIT_MS = float(os.getenv("LATENCY_WARN_ORDER_MS", "3000.0"))

# API RATE LIMITING
# Hyperliquid allows 1200 weight per minute
MAX_WEIGHT_PER_MINUTE = 1200
WEIGHT_WARNING_THRESHOLD = 1000  # Start slowing down at 1000

# Main loop sleep time (seconds)
# For 100+ trades/hour, we need fast updates
LOOP_INTERVAL = float(os.getenv("LOOP_INTERVAL", "0.1"))  # 10 Hz updates

# Optional simulated latency (milliseconds) to approximate network/venue latency in dry-run/tests.
# Example: SIMULATED_LATENCY_MS=15 simulates ~15ms extra delay.
SIMULATED_LATENCY_MS = float(os.getenv("SIMULATED_LATENCY_MS", "0"))

# LOGGING & NOTIFICATIONS
# Discord webhook for notifications (optional)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Discord notification throttling (to avoid webhook overload)
# Target: 1–5 notifications/minute.
DISCORD_MIN_INTERVAL_SECONDS = float(os.getenv("DISCORD_MIN_INTERVAL_SECONDS", "12"))
DISCORD_MAX_QUEUE_SIZE = int(os.getenv("DISCORD_MAX_QUEUE_SIZE", "500"))
DISCORD_DROP_NOISY_WHEN_BACKLOG = os.getenv("DISCORD_DROP_NOISY_WHEN_BACKLOG", "True").lower() == "true"

# Only send "important" alerts to Discord (recommended)
# Important alerts include: status updates, trade closed, risk/kill switch, errors, bot lifecycle.
DISCORD_IMPORTANT_ONLY = os.getenv("DISCORD_IMPORTANT_ONLY", "True").lower() == "true"

# Per-category toggles
DISCORD_NOTIFY_STATUS = os.getenv("DISCORD_NOTIFY_STATUS", "True").lower() == "true"
DISCORD_NOTIFY_TRADE_CLOSED = os.getenv("DISCORD_NOTIFY_TRADE_CLOSED", "True").lower() == "true"
DISCORD_NOTIFY_TRADE_OPENED = os.getenv("DISCORD_NOTIFY_TRADE_OPENED", "False").lower() == "true"
DISCORD_NOTIFY_RISK = os.getenv("DISCORD_NOTIFY_RISK", "True").lower() == "true"
DISCORD_NOTIFY_ERRORS = os.getenv("DISCORD_NOTIFY_ERRORS", "True").lower() == "true"
DISCORD_NOTIFY_LIFECYCLE = os.getenv("DISCORD_NOTIFY_LIFECYCLE", "True").lower() == "true"

# Periodic status push interval (seconds)
DISCORD_STATUS_UPDATE_INTERVAL_SECONDS = float(os.getenv("DISCORD_STATUS_UPDATE_INTERVAL_SECONDS", "900"))

# Coalesce high-frequency fills into periodic summaries.
DISCORD_FILL_SUMMARY_INTERVAL_SECONDS = float(os.getenv("DISCORD_FILL_SUMMARY_INTERVAL_SECONDS", "30"))
DISCORD_RISK_WARNING_COOLDOWN_SECONDS = float(os.getenv("DISCORD_RISK_WARNING_COOLDOWN_SECONDS", "60"))
DISCORD_FILL_SUMMARY_COOLDOWN_SECONDS = float(os.getenv("DISCORD_FILL_SUMMARY_COOLDOWN_SECONDS", "0"))

# Trade journal file
JOURNAL_FILE = "trade_history.json"

# Risk state persistence file
RISK_STATE_FILE = "risk_state.json"

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# AI ANALYST (Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANALYST_RUN_HOUR = int(os.getenv("ANALYST_HOUR", "0"))  # Run analysis at midnight UTC

# MACHINE LEARNING ENGINE
# Use trained scikit-learn model for signal confirmation
USE_ML_MODEL = os.getenv("USE_ML_MODEL", "False").lower() == "true"
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/direction_classifier.joblib")
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE", "0.65"))  # Min prob to act

# FUNDING RATE INTEGRATION
# Enable funding rate bias in strategy (skew towards collecting funding)
FUNDING_BIAS_ENABLED = os.getenv("FUNDING_BIAS_ENABLED", "True").lower() == "true"
# Minimum funding rate to trigger bias (0.0001 = 0.01%)
FUNDING_THRESHOLD = float(os.getenv("FUNDING_THRESHOLD", "0.0001"))
# Multiplier for funding-based skew adjustment
FUNDING_SKEW_MULTIPLIER = float(os.getenv("FUNDING_SKEW_MULT", "10.0"))
# Cache funding rate for N seconds (avoid excessive API calls)
FUNDING_CACHE_SECONDS = int(os.getenv("FUNDING_CACHE_SEC", "60"))

# CONVEX INVENTORY PENALTY (Quadratic Skew)
# Use quadratic (convex) skew instead of linear - penalizes large positions more
USE_CONVEX_SKEW = os.getenv("USE_CONVEX_SKEW", "True").lower() == "true"
# Maximum skew in basis points at full position
CONVEX_MAX_SKEW_BPS = float(os.getenv("CONVEX_MAX_SKEW_BPS", "25.0"))

# KELLY CRITERION POSITION SIZING
# Enable Kelly-based position sizing (requires trade history)
USE_KELLY_SIZING = os.getenv("USE_KELLY_SIZING", "False").lower() == "true"
# Kelly fraction multiplier (0.5 = half-Kelly for safety)
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.5"))
# Minimum trades required before Kelly kicks in
KELLY_MIN_TRADES = int(os.getenv("KELLY_MIN_TRADES", "50"))
# Maximum clip size even with Kelly (safety cap)
KELLY_MAX_CLIP_PCT = float(os.getenv("KELLY_MAX_CLIP", "0.15"))

# TIME-OF-DAY ADJUSTMENTS
# Enable time-based spread adjustments
TIME_ADJUST_ENABLED = os.getenv("TIME_ADJUST_ENABLED", "True").lower() == "true"
# Low volume hours (UTC) - widen spreads
LOW_VOLUME_HOURS = [int(h) for h in os.getenv("LOW_VOLUME_HOURS", "2,3,4,5,6,7").split(",")]
LOW_VOLUME_SPREAD_MULT = float(os.getenv("LOW_VOL_SPREAD_MULT", "1.3"))
# High volume hours (UTC) - tighten spreads  
HIGH_VOLUME_HOURS = [int(h) for h in os.getenv("HIGH_VOLUME_HOURS", "13,14,15,16").split(",")]
HIGH_VOLUME_SPREAD_MULT = float(os.getenv("HIGH_VOL_SPREAD_MULT", "0.9"))

# SNIPER PROTECTION
# Add random jitter to order placement timing (anti-front-run)
ORDER_JITTER_ENABLED = os.getenv("ORDER_JITTER_ENABLED", "True").lower() == "true"
ORDER_JITTER_MIN_MS = float(os.getenv("ORDER_JITTER_MIN_MS", "5.0"))
ORDER_JITTER_MAX_MS = float(os.getenv("ORDER_JITTER_MAX_MS", "50.0"))


# PRICE PRECISION (Fallback - auto-fetched from API)
# NOTE: These are FALLBACK values only. The OrderManager now auto-fetches 
# szDecimals from HyperLiquid's meta API and calculates the correct tick size
# using the 5 significant figures rule.
#
# HyperLiquid price precision rules:
# - Max 5 significant figures
# - Max decimals = 6 - szDecimals (for perps)
# - For ETH (szDecimals=3) at ~$3000: tick size = 0.10

PRICE_DECIMALS = {
    "BTC": 0,    # At $90,000+, tick = 1.0
    "ETH": 1,    # At $3,000+, tick = 0.10
    "SOL": 2,    # At $200+, tick = 0.01
    "HYPE": 2,   # At $20-30, tick = 0.001 but limited by 5 sig figs
    "DOGE": 5,
    "PEPE": 8,
    "WIF": 4,
}

SIZE_DECIMALS = {
    "BTC": 5,    # szDecimals from testnet API
    "ETH": 4,    # szDecimals from testnet API (was 3)
    "SOL": 2,    # szDecimals from testnet API
    "HYPE": 2,   # szDecimals from testnet API
    "DOGE": 0,
    "PEPE": 0,
    "WIF": 1,
}

def get_price_precision(coin: str) -> int:
    """Get price decimal precision for a coin (fallback values)."""
    return PRICE_DECIMALS.get(coin, 2)


def get_size_precision(coin: str) -> int:
    """Get size decimal precision for a coin (fallback values)."""
    return SIZE_DECIMALS.get(coin, 2)

# OPTIMIZATION PARAMETERS (Advanced / Internal)
# These allow the optimizer to tune the "physics" of the bot.

# Regime Engine
SKEW_MODE_ENTRY_THRESHOLD = float(os.getenv("SKEW_ENTRY", "0.5"))    # Enter skew mode at 50% position (Relaxed)
SKEW_MODE_EXIT_THRESHOLD = float(os.getenv("SKEW_EXIT", "0.25"))     # Exit skew mode at 25% position
MIN_REGIME_DURATION = float(os.getenv("MIN_REGIME_DURATION", "30.0")) # 30s minimum in regime to prevent churning

# Strategy Engine
HUNTER_MAX_AGGRESSION_BPS = float(os.getenv("HUNTER_AGG_BPS", "5.0")) # More aggressive (5bps)
VWAP_PRESSURE_SCALER = float(os.getenv("PRESSURE_SCALER", "0.0003"))  # Higher pressure sensitivity
COMPOUND_CLIP_SIZE = float(os.getenv("CLIP_SIZE", "0.10"))           # % of account per order (10% - safer)

# Orderbook Analyzer (Market Physics)
OB_OIMB_DECAY = float(os.getenv("OB_OIMB_DECAY", "1.5"))             # Stronger focus on top of book
OB_PRESSURE_DIST_FACTOR = float(os.getenv("OB_PRESS_DIST", "100.0")) # Keep default
OB_VOL_ADJUST_SCALER = float(os.getenv("OB_VOL_SCALER", "2000.0"))   # Less fearful of volatility (was 3000)
OB_LIQ_DEPTH_DIVISOR = float(os.getenv("OB_LIQ_DIV", "500.0"))       

# Simulation Accuracy (Backtesting)
SIM_MAKER_FILL_PROB = float(os.getenv("SIM_FILL_PROB", "0.15"))      # Optimistic fill prob (15%)
SIM_MAKER_THRESHOLD_BPS = float(os.getenv("SIM_FILL_BPS", "5.0"))    # Proximity required

# HFT IMPROVEMENTS (Advanced)
# Adverse price move cancellation - cancel orders at risk of adverse selection
ADVERSE_CANCEL_ENABLED = os.getenv("ADVERSE_CANCEL_ENABLED", "True").lower() == "true"
ADVERSE_CANCEL_BPS = float(os.getenv("ADVERSE_CANCEL_BPS", "2.0"))  # Cancel if price moved 2bps against

# Inventory decay - older positions get more aggressive skew
USE_INVENTORY_DECAY = os.getenv("USE_INVENTORY_DECAY", "True").lower() == "true"
INVENTORY_HALF_LIFE_SECONDS = float(os.getenv("INV_HALF_LIFE", "120.0"))  # 2 minutes

# Target maker fill ratio - widen spreads if ratio too low
TARGET_MAKER_RATIO = float(os.getenv("TARGET_MAKER_RATIO", "0.90"))  # 90% maker target

# Partial fill handling
PARTIAL_FILL_REPLENISH = os.getenv("PARTIAL_FILL_REPLENISH", "True").lower() == "true"
PARTIAL_FILL_THRESHOLD = float(os.getenv("PARTIAL_FILL_THRESHOLD", "0.5"))  # Replenish when 50% filled

# DATA MAINTENANCE
MAX_DATA_RETENTION_FILES = int(os.getenv("MAX_DATA_RETENTION", "10")) # Keep last 10 capture files

# SIMULATED VENUE (Realistic Dry-Run)
# Enable the realistic simulated venue (queue model, latency, partials, etc.)
USE_REALISTIC_SIM = os.getenv("USE_REALISTIC_SIM", "True").lower() == "true"

# Order ack latency (ms) - time before order is confirmed resting
SIM_ORDER_ACK_MS = float(os.getenv("SIM_ORDER_ACK_MS", "80"))

# Cancel ack latency (ms) - orders can still fill during this window
SIM_CANCEL_ACK_MS = float(os.getenv("SIM_CANCEL_ACK_MS", "120"))

# Extra taker slippage beyond VWAP depth-walk (bps)
SIM_TAKER_SLIPPAGE_EXTRA_BPS = float(os.getenv("SIM_TAKER_SLIPPAGE_EXTRA_BPS", "0.5"))

# Adverse selection penalty applied to maker fills (bps)
SIM_ADVERSE_SELECT_BPS = float(os.getenv("SIM_ADVERSE_SELECT_BPS", "0.5"))

# Adverse selection mode: "immediate" (mark penalty) or "drift" (next-N-ticks)
SIM_ADVERSE_SELECT_MODE = os.getenv("SIM_ADVERSE_SELECT_MODE", "immediate")

# Partial fill chunk size (fraction of remaining order per tick)
SIM_PARTIAL_CHUNK_PCT = float(os.getenv("SIM_PARTIAL_CHUNK_PCT", "0.5"))

# Stochastic rejection probabilities (per order)
SIM_REJECT_PROB_RATE_LIMIT = float(os.getenv("SIM_REJECT_PROB_RATE_LIMIT", "0.002"))
SIM_REJECT_PROB_MARGIN = float(os.getenv("SIM_REJECT_PROB_MARGIN", "0.001"))

# Rate limit budget (orders per minute, token bucket)
SIM_RATE_LIMIT_PER_MIN = float(os.getenv("SIM_RATE_LIMIT_PER_MIN", "1200"))

# RNG seed for reproducible backtests (None = non-deterministic)
SIM_RNG_SEED = os.getenv("SIM_RNG_SEED", None)
if SIM_RNG_SEED is not None:
    SIM_RNG_SEED = int(SIM_RNG_SEED)
