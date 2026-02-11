import asyncio
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from core.market_data import MarketDataManager


async def main():
    md = MarketDataManager(coin=config.COIN, use_testnet=config.USE_TESTNET)
    await md.connect()

    t0 = md.last_update
    await asyncio.sleep(1.0)
    ok1 = await md.update()
    t1 = md.last_update
    await asyncio.sleep(1.0)
    ok2 = await md.update()
    t2 = md.last_update

    if md._ws_task:
        md._ws_task.cancel()
        try:
            await md._ws_task
        except Exception:
            pass

    print(
        {
            "ok1": ok1,
            "ok2": ok2,
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "age_sec": round(time.time() - t2, 3),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
