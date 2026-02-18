
import pytest
from core.orderbook_analyzer import OrderbookAnalyzer
from core.regime_engine import TradingRegime

class TestOrderbookAnalyzer:
    def setup_method(self):
        self.analyzer = OrderbookAnalyzer()

    def test_analyze_empty_book(self):
        analysis = self.analyzer.analyze([], [], 100.0)
        assert analysis.current_spread_bps == 0
        assert analysis.is_stable is False

    def test_analyze_normal_book(self):
        # Mid price 100
        # Bids: 99.99, 99.98...
        # Asks: 100.01, 100.02...
        bids = [[99.99 - (i*0.01), 100.0] for i in range(10)]
        asks = [[100.01 + (i*0.01), 100.0] for i in range(10)]
        
        analysis = self.analyzer.analyze(bids, asks, 100.0)
        
        # Spread should be (100.01 - 99.99) / 100 = 0.0002 = 2 bps
        assert 1.99 < analysis.current_spread_bps < 2.01
        assert analysis.is_stable is True

    def test_wall_detection(self):
        # Create a wall at 99.95
        bids = [[99.99 - (i*0.01), 100.0] for i in range(10)]
        bids[4][1] = 5000.0  # Big wall at index 4 (99.95)
        
        asks = [[100.01 + (i*0.01), 100.0] for i in range(10)]
        
        analysis = self.analyzer.analyze(bids, asks, 100.0)
        
        assert analysis.bid_wall_price == pytest.approx(99.95)
        assert analysis.bid_wall_size == 5000.0

    def test_imbalance_calculation(self):
        # Heavy buy side
        bids = [[99.99, 1000.0]]
        asks = [[100.01, 100.0]]
        
        analysis = self.analyzer.analyze(bids, asks, 100.0)
        
        assert analysis.top_imbalance > 0.9  # > 90% buyers
