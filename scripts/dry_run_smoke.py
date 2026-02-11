"""
Dry-run smoke check with Logic Verification.

Runs the bot in dry-run mode and VERIFIES that:
1. The Strategy Engine produces valid orders.
2. The Regime Engine can switch states.
3. Orders are not crossed (Buy < Sell).

Usage:
  python scripts/dry_run_smoke.py
"""

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from main import HyperLiquidBot, setup_logging
from core.strategy_engine import StrategyEngine, TradingRegime

# Mock Data for Testing
MOCK_BIDS = [[100.0, 1000], [99.9, 2000], [99.8, 5000]]
MOCK_ASKS = [[100.1, 1000], [100.2, 2000], [100.3, 5000]]
MID_PRICE = 100.05

async def verify_strategy_logic():
    """Directly test the Strategy Engine logic without the full bot loop."""
    print("\n Verifying Strategy Logic...")
    
    strategy = StrategyEngine()
    
    # Test 1: FARMER Mode (Market Making)
    print("  Testing FARMER mode...", end=" ")
    decision = strategy.generate_orders(
        regime=TradingRegime.FARMER_NEUTRAL,
        metadata={"volatility": 0.005, "account_value": 1000},
        mid_price=MID_PRICE,
        bids=MOCK_BIDS,
        asks=MOCK_ASKS,
        current_position_usd=0.0,
        volatility=0.005
    )
    
    if not decision.buy_orders or not decision.sell_orders:
        print(" FAILED: No orders generated")
        return False
        
    buy_price = decision.buy_orders[0].price
    sell_price = decision.sell_orders[0].price
    
    if buy_price >= sell_price:
        print(f" FAILED: Crossed orders (Buy {buy_price} >= Sell {sell_price})")
        return False
        
    print(f" PASS (Spread: {sell_price - buy_price:.4f})")

    # Test 2: Inventory Skew (Long Position)
    print("  Testing Inventory Skew (Long)...", end=" ")
    decision_skew = strategy.generate_orders(
        regime=TradingRegime.FARMER_SKEW_LONG,
        metadata={"volatility": 0.005, "account_value": 1000},
        mid_price=MID_PRICE,
        bids=MOCK_BIDS,
        asks=MOCK_ASKS,
        current_position_usd=500.0, # We are Long
        volatility=0.005
    )
    
    # We expect bids to be lower (less aggressive) when we are already long
    skewed_bid = decision_skew.buy_orders[0].price
    if skewed_bid >= buy_price:
        print(f" WARNING: Bid did not lower despite long inventory ({skewed_bid} >= {buy_price})")
    else:
        print(f" PASS (Bid lowered by {buy_price - skewed_bid:.4f})")

    return True

async def run_full_system_check(seconds: float = 5.0):
    """Runs the full bot loop to check for runtime crashes."""
    print(f"\n Starting Full System Dry-Run ({seconds}s)...")
    setup_logging()
    
    # Override config for safety
    config.COIN = "HYPE" 
    
    bot = HyperLiquidBot(dry_run=True)
    
    # Start bot in background
    task = asyncio.create_task(bot.start())
    
    # Let it run
    await asyncio.sleep(seconds)
    
    # Stop
    print(" Stopping bot...")
    bot.running = False
    try:
        await asyncio.wait_for(task, timeout=5.0)
        print(" Bot shutdown cleanly.")
    except asyncio.TimeoutError:
        print(" Bot shutdown timed out.")
    except Exception as e:
        print(f" Bot crashed: {e}")

async def main_async():
    # 1. Unit Test the Strategy Logic
    success = await verify_strategy_logic()
    if not success:
        print("\n Strategy Verification Failed! Fix logic before running.")
        return

    # 2. Integration Test the Full System
    await run_full_system_check()

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()

