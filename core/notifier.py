# notifier.py - sends messages to discord so you know whats happening

import time
import logging
import requests
import threading
import queue
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum

import config

logger = logging.getLogger("Notifier")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DiscordNotifier:
    """sends discord messages using webhook, runs in background thread"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or config.DISCORD_WEBHOOK_URL
        self.enabled = bool(self.webhook_url)

        # Category gating (reduce spam)
        self.important_only = bool(getattr(config, "DISCORD_IMPORTANT_ONLY", True))
        self.notify_status = bool(getattr(config, "DISCORD_NOTIFY_STATUS", True))
        self.notify_trade_closed = bool(getattr(config, "DISCORD_NOTIFY_TRADE_CLOSED", True))
        self.notify_trade_opened = bool(getattr(config, "DISCORD_NOTIFY_TRADE_OPENED", False))
        self.notify_risk = bool(getattr(config, "DISCORD_NOTIFY_RISK", True))
        self.notify_errors = bool(getattr(config, "DISCORD_NOTIFY_ERRORS", True))
        self.notify_lifecycle = bool(getattr(config, "DISCORD_NOTIFY_LIFECYCLE", True))

        # Keys allowed through when important_only=True (unless urgent=True)
        self._important_key_prefixes = (
            "status_",
            "trade_closed",
            "risk_",
            "kill_switch",
            "error",
            "bot_",
        )
        
        # Async Queue
        self.queue = queue.Queue()
        self.running = True
        
        # Rate limiting
        self.last_message_time = 0
        # Target 1–5 msgs/min by default (12s).
        self.min_interval = float(getattr(config, "DISCORD_MIN_INTERVAL_SECONDS", 12.0))
        self.max_queue_size = int(getattr(config, "DISCORD_MAX_QUEUE_SIZE", 500))
        self.drop_noisy_when_backlog = bool(getattr(config, "DISCORD_DROP_NOISY_WHEN_BACKLOG", True))

        # Fill coalescing (prevents spam in HFT loops)
        self.fill_summary_interval = float(getattr(config, "DISCORD_FILL_SUMMARY_INTERVAL_SECONDS", 30.0))
        self._last_fill_flush = time.time()
        self._fill_summary = {
            "BUY": {"count": 0, "size": 0.0, "value": 0.0},
            "SELL": {"count": 0, "size": 0.0, "value": 0.0},
        }
        self._last_fill_reason: Optional[str] = None

        # Lightweight per-key cooldowns (enqueue-time)
        self._last_enqueue_by_key: Dict[str, float] = {}
        self.cooldowns = {
            # If something starts spamming, cap it without losing critical alerts.
            "risk_warning": float(getattr(config, "DISCORD_RISK_WARNING_COOLDOWN_SECONDS", 60.0)),
            "fill_summary": float(getattr(config, "DISCORD_FILL_SUMMARY_COOLDOWN_SECONDS", 0.0)),
        }
        
        # Color codes for embeds
        self.colors = {
            AlertLevel.INFO: 0x3498db,      # Blue
            AlertLevel.SUCCESS: 0x2ecc71,   # Green
            AlertLevel.WARNING: 0xf39c12,   # Orange
            AlertLevel.ERROR: 0xe74c3c,     # Red
            AlertLevel.CRITICAL: 0x9b59b6,  # Purple
        }
        
        if self.enabled:
            logger.info(" Discord notifications enabled (Async Mode)")
            self._start_worker()
        else:
            logger.info(" Discord notifications disabled (no webhook URL)")

    def _start_worker(self):
        """Start the background worker thread."""
        self.thread = threading.Thread(target=self._worker, daemon=True, name="DiscordWorker")
        self.thread.start()

    def _worker(self):
        """Background worker to process the notification queue."""
        while self.running:
            try:
                payload = self.queue.get()
                if payload is None:
                    break
                
                # Rate limiting
                now = time.time()
                if now - self.last_message_time < self.min_interval:
                    time.sleep(self.min_interval - (now - self.last_message_time))
                
                self._send_payload(payload)
                self.last_message_time = time.time()
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Notifier worker error: {e}")

    def _send_payload(self, payload: Dict[str, Any]):
        """Send the payload to Discord."""
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            if response.status_code not in [200, 204]:
                logger.warning(f"Discord webhook failed: {response.status_code} | {response.text}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def send(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        title: str = None,
        fields: dict = None,
        thumbnail: str = None,
        *,
        key: Optional[str] = None,
        urgent: bool = False,
    ) -> bool:
        """
        Queue a notification to be sent to Discord.
        Returns immediately (non-blocking).
        """
        if not self.enabled:
            return False
            
        try:
            now = time.time()

            # Important-only mode: allow only whitelisted keys unless urgent
            if self.important_only and (not urgent):
                if not key:
                    return False
                if not any(str(key).startswith(p) for p in self._important_key_prefixes):
                    return False

            # Enqueue-time cooldown (keeps spam under control even if caller loops fast)
            if key:
                cooldown = float(self.cooldowns.get(key, 0.0) or 0.0)
                last = self._last_enqueue_by_key.get(key, 0.0)
                if cooldown > 0 and (now - last) < cooldown:
                    return False

            # Backlog protection: drop noisy messages if we're falling behind
            if (not urgent) and self.drop_noisy_when_backlog:
                try:
                    qsize = self.queue.qsize()
                except NotImplementedError:
                    qsize = 0

                if qsize >= self.max_queue_size:
                    if level in (AlertLevel.INFO, AlertLevel.SUCCESS, AlertLevel.WARNING):
                        return False

            # Build embed
            embed = {
                "description": message,
                "color": self.colors.get(level, 0x95a5a6),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {
                    "text": f"HyperLiquid HFT • {config.COIN}",
                    "icon_url": "https://hyperliquid.xyz/favicon.ico"
                }
            }
            
            if title:
                embed["title"] = title
                
            if fields:
                embed["fields"] = [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in fields.items()
                ]
            
            if thumbnail:
                embed["thumbnail"] = {"url": thumbnail}
            
            payload = {
                "username": "HyperLiquid Bot",
                "avatar_url": "https://hyperliquid.xyz/favicon.ico",
                "embeds": [embed]
            }
            
            self.queue.put(payload)
            if key:
                self._last_enqueue_by_key[key] = now
            return True
                
        except Exception as e:
            logger.error(f"Error queuing notification: {e}")
            return False

    def trade_opened(self, coin: str, side: str, price: float, size: float, reason: str):
        """Send notification when a trade is opened."""
        # HFT loops can fill very frequently; coalesce into a periodic summary.
        if (not self.enabled) or (not self.notify_trade_opened):
            return

        now = time.time()
        side_u = (side or "").upper()
        bucket = "BUY" if side_u == "BUY" else "SELL"
        try:
            px = float(price)
            sz = float(size)
        except Exception:
            return

        s = self._fill_summary[bucket]
        s["count"] += 1
        s["size"] += sz
        s["value"] += px * sz
        self._last_fill_reason = reason

        if (now - self._last_fill_flush) < self.fill_summary_interval:
            return

        buy = self._fill_summary["BUY"]
        sell = self._fill_summary["SELL"]
        if buy["count"] == 0 and sell["count"] == 0:
            self._last_fill_flush = now
            return

        fields: Dict[str, str] = {}
        if buy["count"]:
            fields[" BUY fills"] = f"{buy['count']} | {buy['size']:.4f} | ${buy['value']:.2f}"
        if sell["count"]:
            fields[" SELL fills"] = f"{sell['count']} | {sell['size']:.4f} | ${sell['value']:.2f}"
        if self._last_fill_reason:
            fields[" Last reason"] = str(self._last_fill_reason)[:256]

        # Reset window
        self._fill_summary = {
            "BUY": {"count": 0, "size": 0.0, "value": 0.0},
            "SELL": {"count": 0, "size": 0.0, "value": 0.0},
        }
        self._last_fill_flush = now

        self.send(
            message=f"Fills summary (last ~{int(self.fill_summary_interval)}s)",
            level=AlertLevel.INFO,
            title=" Fills",
            fields=fields,
            key="fill_summary",
            urgent=False,
        )

    def trade_closed(self, coin: str, side: str, pnl: float, pnl_pct: float, reason: str, status: dict = None):
        """Send notification when a trade is closed."""
        if not self.notify_trade_closed:
            return
        is_profit = pnl > 0
        level = AlertLevel.SUCCESS if is_profit else AlertLevel.ERROR
        emoji = "" if is_profit else ""
        
        fields = {
            " PnL": f"${pnl:+.2f}",
            " ROI": f"{pnl_pct*100:+.2f}%",
            " Reason": reason,
            " Coin": coin
        }

        if status:
            # Add status fields
            fields[""] = ""
            fields[" Regime"] = status.get("regime", "N/A")
            fields[" Account"] = status.get("account", "N/A")
            fields[" Realized"] = status.get("realized", "N/A")
            fields[" Floating"] = status.get("floating", "N/A")
            fields[" Fees"] = status.get("fees", "N/A")
            fields[" Total PnL"] = status.get("total_pnl", "N/A")
            fields[" Daily PnL"] = status.get("daily_pnl", "N/A")
            fields[" Weekly PnL"] = status.get("weekly_pnl", "N/A")
            fields[" Monthly PnL"] = status.get("monthly_pnl", "N/A")

        self.send(
            message=f"Closed **{side}** position for **${pnl:+.2f}**",
            level=level,
            title=f"{emoji} Trade Closed",
            fields=fields,
            key="trade_closed",
            urgent=True,
        )

    def risk_warning(self, message: str, current_pnl: float, limit: float):
        """Send warning when approaching risk limits."""
        if not self.notify_risk:
            return
        self.send(
            message=f"**{message}**",
            level=AlertLevel.WARNING,
            title=" Risk Warning",
            fields={
                "Current PnL": f"${current_pnl:.2f}",
                "Limit": f"${limit:.2f}",
                "Utilization": f"{(current_pnl/limit)*100:.1f}%" if limit != 0 else "N/A"
            },
            key="risk_warning",
            urgent=False,
        )

    def kill_switch_activated(self, reason: str, final_pnl: float):
        """Send alert when kill switch is triggered."""
        if not self.notify_risk:
            return
        self.send(
            message=f"**{reason}**",
            level=AlertLevel.CRITICAL,
            title=" KILL SWITCH ACTIVATED",
            fields={
                "Final PnL": f"${final_pnl:.2f}",
                "Action": "All orders cancelled & trading halted"
            },
            key="kill_switch",
            urgent=True,
        )

    def regime_change(self, old_regime: str, new_regime: str, reason: str = ""):
        """Send notification on regime change."""
        # Optional: Enable if you want to track strategy shifts
        # self.send(
        #     message=f"{old_regime}  **{new_regime}**",
        #     level=AlertLevel.INFO,
        #     title=" Regime Change",
        #     fields={
        #         "Reason": reason
        #     }
        # )
        pass

    def daily_summary(self, stats: dict):
        """Send daily performance summary."""
        pnl = float(str(stats.get('daily_pnl', '0')).replace('$', ''))
        level = AlertLevel.SUCCESS if pnl >= 0 else AlertLevel.WARNING
        
        self.send(
            message="**Daily Trading Performance Report**",
            level=level,
            title=" End of Day Summary",
            fields={
                " Net PnL": stats.get('daily_pnl', '$0'),
                " Trades": str(stats.get('daily_trades', 0)),
                " Win Rate": stats.get('win_rate', 'N/A'),
                " Daily High": stats.get('daily_high', 'N/A'),
                " Daily Low": stats.get('daily_low', 'N/A')
            }
        )

    def error(self, error_msg: str, context: str = ""):
        """Send error notification."""
        if not self.notify_errors:
            return
        self.send(
            message=f"```{error_msg}```",
            level=AlertLevel.ERROR,
            title=" Bot Error",
            fields={"Context": context} if context else None,
            key="error",
            urgent=True,
        )

    def bot_started(self, coin: str, mode: str):
        """Send notification when bot starts."""
        if not self.notify_lifecycle:
            return
        self.send(
            message=f"Trading **{coin}** in **{mode}** mode",
            level=AlertLevel.INFO,
            title=" Bot Online",
            fields={
                "Network": "Testnet" if config.USE_TESTNET else "Mainnet",
                "Version": "2.0.0",
                "Time": datetime.now(timezone.utc).strftime("%H:%M UTC")
            },
            key="bot_started",
            urgent=True,
        )

    def status_update(self, pnl: float, trades: int, account_value: float, regime: str, status: dict = None):
        """Send periodic status update with PnL."""
        if not self.notify_status:
            return
        level = AlertLevel.SUCCESS if pnl >= 0 else AlertLevel.WARNING
        emoji = "" if pnl >= 0 else ""

        fields = {
            " Daily PnL": f"${pnl:+.2f}",
            " Trades": str(trades),
            " Account": f"${account_value:.2f}",
            " Regime": regime,
        }
        if status:
            # Prefer richer status snapshot if provided
            if status.get("weekly_pnl"):
                fields[" Weekly PnL"] = status.get("weekly_pnl")
            if status.get("monthly_pnl"):
                fields[" Monthly PnL"] = status.get("monthly_pnl")
            if status.get("fees"):
                fields[" Fees"] = status.get("fees")
            if status.get("floating"):
                fields[" Floating"] = status.get("floating")

        self.send(
            message="Current Session Status",
            level=level,
            title=f"{emoji} Status Update",
            fields=fields,
            key="status_update",
            urgent=False,
        )

    def bot_stopped(self, reason: str = "Manual shutdown"):
        """Send notification when bot stops."""
        if not self.notify_lifecycle:
            return
        self.send(
            message=f"**{reason}**",
            level=AlertLevel.WARNING,
            title=" Bot Stopped",
            fields={
                "Time": datetime.now(timezone.utc).strftime("%H:%M UTC")
            },
            key="bot_stopped",
            urgent=True,
        )


class ConsoleNotifier:
    """just prints to console if no discord webhook"""

    def send(self, message: str, level: AlertLevel = AlertLevel.INFO, **kwargs):
        """Log to console."""
        log_levels = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.SUCCESS: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }
        logger.log(log_levels.get(level, logging.INFO), message)
        return True

    def trade_opened(self, *args, **kwargs):
        pass
        
    def trade_closed(self, *args, **kwargs):
        pass
        
    def risk_warning(self, *args, **kwargs):
        logger.warning(f"Risk warning: {args[0] if args else ''}")
        
    def kill_switch_activated(self, reason: str, *args, **kwargs):
        logger.critical(f"KILL SWITCH: {reason}")
        
    def regime_change(self, *args, **kwargs):
        pass
        
    def daily_summary(self, *args, **kwargs):
        pass
        
    def error(self, error_msg: str, *args, **kwargs):
        logger.error(error_msg)
        
    def bot_started(self, *args, **kwargs):
        logger.info("Bot started")
        
    def status_update(self, *args, **kwargs):
        pass

    def bot_stopped(self, *args, **kwargs):
        logger.info("Bot stopped")


def get_notifier() -> DiscordNotifier:
    """
    Factory function to get the appropriate notifier.
    Returns Discord notifier if configured, otherwise console-only.
    """
    if config.DISCORD_WEBHOOK_URL:
        return DiscordNotifier()
    return ConsoleNotifier()
