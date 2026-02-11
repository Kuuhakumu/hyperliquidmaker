import asyncio
import time
import logging
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.order_manager import OrderManager
import config

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LatencyTest")

async def test_latency():
    # Initialize OrderManager (using live settings but we won't place orders)
    logger.info("Initializing OrderManager...")
    om = OrderManager(config.PRIVATE_KEY, config.COIN, use_testnet=config.USE_TESTNET)
    
    logger.info(f"Connected to {'Testnet' if config.USE_TESTNET else 'Mainnet'}")
    logger.info(f"Targeting: {config.COIN}")
    
    # 1. Test raw REST Latency (Position/State)
    logger.info("\n--- Measuring REST Latency (User State) ---")
    latencies = []
    for i in range(10):
        start = time.time()
        # We manually call user_state to bypass the 1s cache for this test
        om.info.user_state(om.address)
        end = time.time()
        ms = (end - start) * 1000
        latencies.append(ms)
        logger.info(f"Tick {i+1}: {ms:.2f}ms")
        await asyncio.sleep(0.5)
        
    avg = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    logger.info(f"REST Stats - Avg: {avg:.2f}ms, P95: {p95:.2f}ms")

    # 2. Test Market Data Refresh (Info API)
    logger.info("\n--- Measuring REST Latency (L2 Snapshot) ---")
    latencies = []
    for i in range(10):
        start = time.time()
        om.info.l2_snapshot(config.COIN)
        end = time.time()
        ms = (end - start) * 1000
        latencies.append(ms)
        logger.info(f"Tick {i+1}: {ms:.2f}ms")
        await asyncio.sleep(0.5)
        
    avg = sum(latencies) / len(latencies)
    logger.info(f"L2 Snapshot Stats - Avg: {avg:.2f}ms")

    logger.info("\n Latency check complete.")

if __name__ == "__main__":
    asyncio.run(test_latency())
