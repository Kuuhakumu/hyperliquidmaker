# HyperLiquid Market Making Bot

a trading bot for hyperliquid perp futures. it does market making (placing buy and sell orders around the current price to earn spread + rebates).

## how it works

basically the bot looks at the orderbook and figures out where to place orders. it has 2 modes:

- **farmer mode** - normal market making, place orders on both sides and try to capture the spread
- **hunter mode** - when the market is trending hard, it tries to follow the trend instead

the bot also has risk management stuff like:
- daily loss limits so you dont blow up
- volatility detection (if market goes crazy it widens spreads or stops trading)
- inventory management (if you hold too much it tries to get rid of it)

## setup

1. install python packages
```
pip install -r requirements.txt
```

2. copy `.env.example` to `.env` and put your private key in there

3. run the bot
```
python main.py --dry-run
```

remove `--dry-run` when you want to trade for real (careful with this lol)

## project structure

- `main.py` - main loop that runs everything
- `config.py` - all the settings
- `core/` - all the important stuff (strategy, risk, orders, etc)
- `utils/` - helper stuff
- `scripts/` - automation scripts
- `tests/` - tests

## warning

this is for learning purposes. trading crypto futures is risky and you can lose money. dont trade with money you cant afford to lose
