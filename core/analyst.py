# analyst.py - analyzes trading data
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger("Analyst")

# Try to import Gemini SDK (google-genai)
try:
    from google import genai  # type: ignore
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google GenAI SDK not installed. Run: pip install google-genai")


def _extract_text(resp) -> str:
    """Best-effort extraction of text from google-genai responses."""
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(resp, "candidates", None)
    if not candidates:
        return ""

    try:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            part0 = parts[0]
            part_text = getattr(part0, "text", None)
            if isinstance(part_text, str):
                return part_text
    except Exception:
        return ""

    return ""


class _GenAIModel:
    """Small shim so the rest of the code can call .generate_content(prompt)."""

    def __init__(self, api_key: str, model_name: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model_name

    def generate_content(self, prompt: str):
        resp = self._client.models.generate_content(model=self._model, contents=prompt)

        # Mimic the old SDK shape (response.text)
        class _R:
            pass

        r = _R()
        r.text = _extract_text(resp)
        return r


class TradingAnalyst:
    """
    AI-powered trading analyst using Google Gemini.
    
    Analyzes:
    - Losing trade patterns
    - Winning trade patterns
    - Regime effectiveness
    - Market condition correlations
    - Suggested improvements
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = None
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                # Default to a fast/cheap model; can be changed later.
                self.model = _GenAIModel(api_key=self.api_key, model_name="gemini-2.0-flash")
                logger.info(" Gemini AI initialized (google-genai)")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")

    def is_available(self) -> bool:
        """Check if AI analysis is available."""
        return self.model is not None

    def analyze_performance(self, journal_data: str) -> Optional[str]:
        """
        Analyze trading performance and generate insights.
        
        Args:
            journal_data: JSON string from TradeJournal.export_for_analysis()
            
        Returns:
            Analysis report as string, or None if failed
        """
        if not self.is_available():
            logger.warning("Gemini not available for analysis")
            return None
        
        try:
            data = json.loads(journal_data)
            
            # Build the analysis prompt
            prompt = self._build_prompt(data)
            
            logger.info(" Sending trade data to Gemini for analysis...")
            
            response = self.model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return None

    def _build_prompt(self, data: dict) -> str:
        """Build the analysis prompt for Gemini."""
        summary = data.get('summary', {})
        losses = data.get('recent_losses', [])
        wins = data.get('recent_wins', [])
        
        prompt = f"""
You are a Senior Quantitative Trader reviewing a junior trading bot's performance.
Your job is to find patterns in the losses and suggest specific code improvements.

=== OVERALL STATS ===
Total Trades: {summary.get('total_trades', 0)}
Win Rate: {summary.get('win_rate', 'N/A')}
Total PnL: {summary.get('total_pnl', '$0')}
Average PnL per Trade: {summary.get('avg_pnl', '$0')}

=== RECENT LOSING TRADES ({len(losses)} trades) ===
Analyze the 'entry_context' and 'exit_context' to find common patterns.

{json.dumps(losses, indent=2)}

=== RECENT WINNING TRADES ({len(wins)} trades) ===
Compare these to losses to see what works.

{json.dumps(wins, indent=2)}

=== ANALYSIS TASKS ===

1. **Pattern Detection**: 
   - Look at 'imbalance', 'volatility', 'spread_pct', 'regime', 'hour_of_day' in the contexts
   - Find the common factor in losing trades (e.g., "losses happen when imbalance < 0.4")
   
2. **Regime Analysis**:
   - Is HUNTER mode profitable or losing money?
   - Is FARMER mode profitable or losing money?
   - Should we adjust when to switch regimes?

3. **Time Analysis**:
   - Are losses concentrated in certain hours?
   - Should we avoid trading at certain times?

4. **Specific Recommendations**:
   Provide exactly ONE specific code change for each problem found.
   Format: "Change [variable] from [current] to [new] because [reason]"
   
   Example: "Change IMBALANCE_BULL_THRESHOLD from 0.65 to 0.70 because most losses occurred when imbalance was between 0.6-0.7"

=== OUTPUT FORMAT ===
Be brutal and mathematical. No fluff.

## Key Findings
- [Finding 1]
- [Finding 2]

## Winning Pattern
[What conditions are present when we win]

## Losing Pattern  
[What conditions are present when we lose]

## Code Changes (Priority Order)
1. [Most important change]
2. [Second change]
3. [Third change]

## Risk Warning
[Any concerning patterns that could blow up the account]
"""
        return prompt

    def generate_daily_report(self, journal_data: str, risk_summary: dict) -> Optional[str]:
        """
        Generate a comprehensive daily report.
        
        Args:
            journal_data: JSON string from TradeJournal
            risk_summary: Dict from RiskManager.get_summary()
            
        Returns:
            Daily report as string
        """
        if not self.is_available():
            return self._generate_basic_report(journal_data, risk_summary)
        
        try:
            data = json.loads(journal_data)
            
            prompt = f"""
Generate a concise daily trading report for a Hyperliquid HFT bot.

=== PERFORMANCE ===
{json.dumps(data.get('summary', {}), indent=2)}

=== RISK STATUS ===
{json.dumps(risk_summary, indent=2)}

=== RECENT TRADES ===
{len(data.get('recent_losses', []))} losses, {len(data.get('recent_wins', []))} wins

=== REPORT FORMAT ===
Keep it under 200 words. Be direct.

 DAILY SUMMARY
- PnL: [amount]
- Trades: [count] ([win rate]%)
- Best trade: [details]
- Worst trade: [details]

 CONCERNS
- [Any issues to watch]

 TOMORROW
- [One suggestion for improvement]
"""
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Daily report failed: {e}")
            return self._generate_basic_report(journal_data, risk_summary)

    def _generate_basic_report(self, journal_data: str, risk_summary: dict) -> str:
        """Generate a basic report without AI."""
        try:
            data = json.loads(journal_data)
            summary = data.get('summary', {})
            
            report = f"""
 DAILY TRADING REPORT
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

=== PERFORMANCE ===
Total Trades: {summary.get('total_trades', 0)}
Win Rate: {summary.get('win_rate', 'N/A')}
Total PnL: {summary.get('total_pnl', '$0.00')}

=== RISK STATUS ===
Can Trade: {' Yes' if risk_summary.get('can_trade', True) else ' No'}
Daily PnL: {risk_summary.get('daily_pnl', '$0.00')}
Drawdown: {risk_summary.get('drawdown_pct', '0%')}

=== NOTE ===
AI analysis unavailable. Configure GEMINI_API_KEY for detailed insights.
"""
            return report
            
        except Exception as e:
            return f"Report generation failed: {e}"


def run_daily_analysis():
    """
    Standalone function to run daily analysis.
    Can be called from a scheduler or manually.
    """
    from core.journal import TradeJournal
    from core.risk_manager import RiskManager
    
    print("\n" + "="*60)
    print(" DAILY TRADING ANALYSIS")
    print("="*60 + "\n")
    
    # Load journal
    journal = TradeJournal()
    journal_data = journal.export_for_analysis()
    
    # Load risk state
    risk = RiskManager()
    risk_summary = risk.get_summary()
    
    # Initialize analyst
    analyst = TradingAnalyst()
    
    if not analyst.is_available():
        print(" Gemini API not configured. Set GEMINI_API_KEY in .env")
        print("\nBasic Statistics:")
        print(journal_data)
        return
    
    # Run analysis
    print("Analyzing trading patterns...\n")
    
    analysis = analyst.analyze_performance(journal_data)
    
    if analysis:
        print(analysis)
    else:
        print("Analysis failed. Check logs for details.")
    
    print("\n" + "="*60)
    print("Analysis complete.")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run analysis when script is called directly
    logging.basicConfig(level=logging.INFO)
    run_daily_analysis()
