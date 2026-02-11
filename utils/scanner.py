# scanner.py - scans coins to find good ones to trade
import time
import numpy as np
from hyperliquid.info import Info
from hyperliquid.utils import constants


def scan_markets():
    """Scan all Hyperliquid markets and rank them for trading."""
    
    print("\n" + "="*60)
    print(" HYPERLIQUID MARKET SCANNER")
    print("="*60 + "\n")
    
    # Connect to Hyperliquid (no auth needed for scanning)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    print(" Fetching market data...\n")
    
    # Get all available markets
    meta = info.meta()
    universe = meta.get('universe', [])
    
    if not universe:
        print(" Failed to fetch market data")
        return
    
    scored_coins = []
    
    print(f"Analyzing {len(universe)} markets...\n")
    
    for i, coin_data in enumerate(universe[:30]):  # Top 30 by market cap
        coin = coin_data['name']
        
        try:
            # Get recent candles for volatility
            candles = info.candles_snapshot(coin, "15m", 100)
            
            if not candles:
                continue
            
            # Extract data
            closes = np.array([float(c['c']) for c in candles])
            volumes = np.array([float(c['v']) * float(c['c']) for c in candles])
            
            # Calculate metrics
            # 1. VOLUME (24h approximation from ~25 hours of 15m candles)
            total_volume = np.sum(volumes)
            
            # Skip low volume coins
            if total_volume < 5_000_000:
                print(f"  {coin}: Skipped (Vol < $5M)")
                continue
            
            # 2. VOLATILITY (Standard deviation of returns)
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * 100  # As percentage
            
            # 3. PRICE CHANGE (Trendiness)
            price_change = abs(closes[-1] - closes[0]) / closes[0] * 100
            
            # 4. Get current spread
            l2 = info.l2_snapshot(coin)
            if l2 and l2['levels'][0] and l2['levels'][1]:
                best_bid = float(l2['levels'][0][0]['px'])
                best_ask = float(l2['levels'][1][0]['px'])
                spread = (best_ask - best_bid) / best_bid * 100
            else:
                spread = 0.1  # Default
            
            # SCORING
            # We want: High volatility (movement) + High volume (liquidity) + Reasonable spread
            score = (volatility * 2) + (price_change * 1.5) - (spread * 5)
            
            # Normalize volume for display
            vol_m = total_volume / 1_000_000
            
            scored_coins.append({
                "coin": coin,
                "volume_m": vol_m,
                "volatility": volatility,
                "price_change": price_change,
                "spread": spread,
                "score": score,
                "current_price": closes[-1]
            })
            
            print(f" {coin}: Vol=${vol_m:.1f}M, Volat={volatility:.3f}%, Spread={spread:.4f}%")
            
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            print(f"  {coin}: Error - {e}")
            continue
    
    if not scored_coins:
        print("\n No suitable coins found")
        return
    
    # Sort by score
    scored_coins.sort(key=lambda x: x['score'], reverse=True)
    
    # Print results
    print("\n" + "="*60)
    print(" TOP 5 COINS FOR TRADING TODAY")
    print("="*60 + "\n")
    
    for i, coin in enumerate(scored_coins[:5], 1):
        # Determine recommended mode
        if coin['volatility'] > 1.0:
            mode = "HUNTER (High Volatility)"
            mode_icon = ""
        elif coin['volatility'] > 0.5:
            mode = "HYBRID"
            mode_icon = ""
        else:
            mode = "FARMER (Low Volatility)"
            mode_icon = ""
        
        print(f"{i}. {coin['coin']}")
        print(f"    Volume: ${coin['volume_m']:.1f}M/day")
        print(f"    Volatility: {coin['volatility']:.3f}%")
        print(f"    24h Change: {coin['price_change']:.2f}%")
        print(f"    Spread: {coin['spread']:.4f}%")
        print(f"   {mode_icon} Recommended: {mode}")
        print(f"    Current Price: ${coin['current_price']:.4f}")
        print()
    
    # Show warnings for risky coins
    print("="*60)
    print("  COINS TO AVOID (Low Liquidity)")
    print("="*60 + "\n")
    
    for coin in scored_coins[-3:]:
        if coin['volume_m'] < 10:
            print(f" {coin['coin']}: Only ${coin['volume_m']:.1f}M volume - Hard to exit positions")
    
    print("\n" + "="*60)
    print(" RECOMMENDATION")
    print("="*60)
    
    top_pick = scored_coins[0]
    print(f"\nToday's best pick: {top_pick['coin']}")
    print(f"Update your .env: COIN={top_pick['coin']}")
    print("\n")


def get_coin_details(coin: str):
    """Get detailed analysis of a specific coin."""
    
    print(f"\n Detailed Analysis: {coin}\n")
    
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    # Get orderbook
    l2 = info.l2_snapshot(coin)
    
    if not l2:
        print(f" Could not fetch data for {coin}")
        return
    
    bids = l2['levels'][0][:10]
    asks = l2['levels'][1][:10]
    
    print(" ORDER BOOK (Top 10)")
    print("-" * 50)
    print(f"{'BIDS':^25}|{'ASKS':^25}")
    print("-" * 50)
    
    for i in range(min(10, len(bids), len(asks))):
        bid_px = float(bids[i]['px'])
        bid_sz = float(bids[i]['sz'])
        ask_px = float(asks[i]['px'])
        ask_sz = float(asks[i]['sz'])
        
        print(f"${bid_px:>10.4f} ({bid_sz:>6.2f}) | ${ask_px:>10.4f} ({ask_sz:>6.2f})")
    
    # Calculate imbalance
    bid_vol = sum(float(b['sz']) for b in bids[:5])
    ask_vol = sum(float(a['sz']) for a in asks[:5])
    total = bid_vol + ask_vol
    imbalance = bid_vol / total if total > 0 else 0.5
    
    print("-" * 50)
    print(f"Bid Volume (Top 5): {bid_vol:.2f}")
    print(f"Ask Volume (Top 5): {ask_vol:.2f}")
    print(f"Imbalance: {imbalance:.2%} ({'BULLISH' if imbalance > 0.6 else 'BEARISH' if imbalance < 0.4 else 'NEUTRAL'})")
    
    # Spread
    spread = float(asks[0]['px']) - float(bids[0]['px'])
    mid = (float(asks[0]['px']) + float(bids[0]['px'])) / 2
    spread_pct = spread / mid * 100
    
    print(f"Spread: ${spread:.4f} ({spread_pct:.4f}%)")
    
    # Recommendation
    print("\n TRADING RECOMMENDATION")
    print("-" * 50)
    
    if spread_pct < 0.03:
        print("  Spread is very tight - hard to profit as market maker")
        print("   Consider HUNTER mode only")
    elif spread_pct > 0.1:
        print("  Spread is wide - low liquidity, be careful")
        print("   Use small position sizes")
    else:
        print(" Spread is optimal for market making")
    
    if imbalance > 0.6:
        print(" Order book shows BUY pressure - consider long bias")
    elif imbalance < 0.4:
        print(" Order book shows SELL pressure - consider short bias")
    
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Analyze specific coin
        get_coin_details(sys.argv[1].upper())
    else:
        # Scan all markets
        scan_markets()
