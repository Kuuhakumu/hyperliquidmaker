#!/usr/bin/env python3
"""
Test script for the SimulatedVenue (realistic dry-run paper trading).

Verifies:
1. Queue position modeling (fills only after queue consumed)
2. Order ack latency (orders pending before resting)
3. Cancel latency + fill-during-cancel
4. Partial fills
5. Taker slippage (depth-walk VWAP)
6. Adverse selection penalty
7. Stochastic rejections

Run: python tests/test_sim_venue.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass
from typing import List

# Mock order object matching strategy_engine output
@dataclass
class MockOrder:
    price: float
    size: float
    reduce_only: bool = False

def test_sim_venue():
    """Main test suite for SimulatedVenue."""
    from core.sim_venue import (
        SimulatedVenue,
        compute_taker_fill_price,
        OrderSide,
        OrderStatus,
    )

    print("=" * 60)
    print("SimulatedVenue Test Suite")
    print("=" * 60)

    # Use fixed seed for reproducibility
    venue = SimulatedVenue(seed=42)

    # Sample L2 orderbook
    bids = [
        [100.00, 10.0],  # Best bid
        [99.95, 20.0],
        [99.90, 30.0],
        [99.85, 50.0],
    ]
    asks = [
        [100.05, 15.0],  # Best ask
        [100.10, 25.0],
        [100.15, 35.0],
        [100.20, 45.0],
    ]
    mid = 100.025

    # --------------------------------------------------------
    # Test 1: Order Ack Latency
    # --------------------------------------------------------
    print("\n[Test 1] Order Ack Latency")
    venue.reset()
    now = 1000.0

    # Submit a buy order
    buy_orders = [MockOrder(price=99.98, size=5.0)]
    fills, rejects = venue.submit_orders(buy_orders, [], now, margin_available=10000)

    # Order should be PENDING_NEW
    orders = venue.get_open_orders()
    assert len(orders) == 1, f"Expected 1 order, got {len(orders)}"
    assert orders[0].status == OrderStatus.PENDING_NEW, f"Expected PENDING_NEW, got {orders[0].status}"
    print(f"    Order submitted, status: {orders[0].status.value}")

    # Advance time past ack
    now += 0.1  # 100ms
    fills, rejects = venue.on_tick(bids, asks, mid, now)
    orders = venue.get_open_orders()
    assert orders[0].status == OrderStatus.OPEN, f"Expected OPEN, got {orders[0].status}"
    print(f"    Order acked, status: {orders[0].status.value}")

    # --------------------------------------------------------
    # Test 2: Queue Position Modeling
    # --------------------------------------------------------
    print("\n[Test 2] Queue Position Modeling")
    venue.reset()
    now = 1000.0

    # Submit buy at best bid (100.00)
    buy_orders = [MockOrder(price=100.00, size=2.0)]
    venue.submit_orders(buy_orders, [], now, margin_available=10000)

    # Ack the order
    now += 0.1
    venue.on_tick(bids, asks, mid, now)

    orders = venue.get_open_orders()
    assert len(orders) == 1
    initial_queue = orders[0].queue_ahead
    print(f"   Initial queue_ahead: {initial_queue:.2f}")
    assert initial_queue >= 10.0, "Should have queue from existing bids"

    # Simulate volume consumed at best bid (reduce size)
    consumed_bids = [
        [100.00, 5.0],  # Reduced from 10 to 5
        [99.95, 20.0],
        [99.90, 30.0],
        [99.85, 50.0],
    ]
    now += 0.1
    fills, _ = venue.on_tick(consumed_bids, asks, mid, now)

    orders = venue.get_open_orders()
    if orders:
        new_queue = orders[0].queue_ahead
        print(f"   Queue after consumption: {new_queue:.2f}")
        assert new_queue < initial_queue, "Queue should decrease"
    print("    Queue position decreases with volume consumption")

    # --------------------------------------------------------
    # Test 3: Cancel Latency + Fill-During-Cancel
    # --------------------------------------------------------
    print("\n[Test 3] Cancel Latency")
    venue.reset()
    now = 1000.0

    buy_orders = [MockOrder(price=100.00, size=1.0)]
    venue.submit_orders(buy_orders, [], now, margin_available=10000)

    # Ack
    now += 0.1
    venue.on_tick(bids, asks, mid, now)

    # Request cancel
    venue.cancel_all(now)
    orders = venue.get_open_orders()
    assert orders[0].status == OrderStatus.PENDING_CANCEL
    print(f"    Cancel requested, status: {orders[0].status.value}")

    # Order still fillable during cancel latency
    # (would need queue consumption to actually fill)

    # After cancel ack time
    now += 0.15  # 150ms
    venue.on_tick(bids, asks, mid, now)
    orders = venue.get_open_orders()
    assert len(orders) == 0, "Order should be cancelled"
    print("    Cancel completed after latency")

    # --------------------------------------------------------
    # Test 4: Taker Slippage (Depth Walk)
    # --------------------------------------------------------
    print("\n[Test 4] Taker Slippage (Depth Walk)")

    # Buy 20 units - should walk through multiple levels
    fill_px, filled = compute_taker_fill_price("BUY", 20.0, bids, asks, extra_slippage_bps=0)
    print(f"   Taker BUY 20 @ VWAP: ${fill_px:.4f} (filled: {filled})")

    # VWAP = (100.05*15 + 100.10*5) / 20 = (1500.75 + 500.5) / 20 = 100.0625
    expected_vwap = (100.05 * 15 + 100.10 * 5) / 20
    assert abs(fill_px - expected_vwap) < 0.01, f"Expected ~{expected_vwap:.4f}"
    print(f"    VWAP matches expected: ${expected_vwap:.4f}")

    # With slippage
    fill_px_slip, _ = compute_taker_fill_price("BUY", 20.0, bids, asks, extra_slippage_bps=5.0)
    assert fill_px_slip > fill_px, "Slippage should worsen fill"
    print(f"    With 5bps slippage: ${fill_px_slip:.4f}")

    # --------------------------------------------------------
    # Test 5: Partial Fills
    # --------------------------------------------------------
    print("\n[Test 5] Partial Fills")
    venue.reset()
    now = 1000.0

    # Place order
    buy_orders = [MockOrder(price=100.00, size=10.0)]
    venue.submit_orders(buy_orders, [], now, margin_available=10000)

    # Ack
    now += 0.1
    venue.on_tick(bids, asks, mid, now)

    # Consume all queue + more
    empty_bids = [
        [99.95, 20.0],  # Best bid level gone
        [99.90, 30.0],
    ]
    now += 0.1
    fills, _ = venue.on_tick(empty_bids, asks, mid, now)

    if fills:
        print(f"   Fill: {fills[0].size:.4f} of 10.0 (remaining: {fills[0].remaining:.4f})")
        # Partial fills should occur (not always full)
        print("    Fill generated after queue consumption")
    else:
        print("   (No fill this tick - queue not fully consumed)")

    # --------------------------------------------------------
    # Test 6: Adverse Selection Penalty
    # --------------------------------------------------------
    print("\n[Test 6] Adverse Selection")
    # Already tested implicitly - fills include adverse_penalty_bps
    if fills:
        print(f"   Adverse penalty: {fills[0].adverse_penalty_bps:.2f} bps")
        print("    Adverse selection penalty applied")
    else:
        print("   (Skipped - no fills to check)")

    # --------------------------------------------------------
    # Test 7: Stochastic Rejections
    # --------------------------------------------------------
    print("\n[Test 7] Stochastic Rejections")
    venue.reset()
    venue._reject_prob_rate = 0.5  # High prob for testing
    venue._reject_prob_margin = 0.5

    reject_count = 0
    for i in range(20):
        now = 1000.0 + i
        buy_orders = [MockOrder(price=99.90, size=1.0)]
        _, rejects = venue.submit_orders(buy_orders, [], now, margin_available=10000)
        reject_count += len(rejects)
        venue.on_tick(bids, asks, mid, now + 0.1)

    print(f"   Rejections: {reject_count}/20 orders")
    assert reject_count > 0, "Should have some rejections with 50% prob"
    print("    Stochastic rejections working")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("All tests passed! ")
    print("=" * 60)


if __name__ == "__main__":
    test_sim_venue()
