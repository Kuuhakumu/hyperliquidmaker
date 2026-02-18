
import pytest
from core.strategy_engine import StrategyEngine, StrategyDecision
from core.regime_engine import TradingRegime

class TestStrategyEngine:
    def setup_method(self):
        self.engine = StrategyEngine()

    def test_farmer_strategy_basic(self):
        # Standard stable market
        bids = [[99.99, 100.0]]
        asks = [[100.01, 100.0]]
        mid = 100.0
        
        decision = self.engine.generate_orders(
            regime=TradingRegime.FARMER_NEUTRAL,
            metadata={"skew_factor": 0},
            mid_price=mid,
            bids=bids,
            asks=asks,
            current_position_usd=0.0,
            volatility=0.0
        )
        
        assert len(decision.buy_orders) > 0
        assert len(decision.sell_orders) > 0
        # Should be inside the spread or at BBO
        assert decision.buy_orders[0].price <= 99.99
        assert decision.sell_orders[0].price >= 100.01

    def test_hunter_strategy_bullish(self):
        # Bullish setup
        bids = [[99.99, 100.0]]
        asks = [[100.01, 100.0]]
        mid = 100.0
        
        decision = self.engine.generate_orders(
            regime=TradingRegime.HUNTER_LONG,
            metadata={},
            mid_price=mid,
            bids=bids,
            asks=asks,
            current_position_usd=0.0,
            volatility=0.002
        )
        
        # Should have aggressive buy and reduce-only sell
        assert len(decision.buy_orders) > 0
        assert decision.buy_orders[0].is_aggressive is True
        
        assert len(decision.sell_orders) > 0
        assert decision.sell_orders[0].reduce_only is True
