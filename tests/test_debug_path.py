
import sys
import pytest

def test_path():
    print("\nSYS PATH:")
    for p in sys.path:
        print(p)
    
    import hyperliquid
    print("\nHYPERLIQUID FILE:", hyperliquid.__file__)
