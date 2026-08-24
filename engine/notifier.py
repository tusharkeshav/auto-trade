# ─────────────────────────────────────────────────────────────────
#  engine/notifier.py
#  100% Free Forever Notification Dispatcher (Telegram & Ntfy Push).
# ─────────────────────────────────────────────────────────────────

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config" / "notifications.json"


def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        "ntfy_topic": os.getenv("NTFY_TOPIC", "akhil_quant_alerts"),
        "enabled": True,
    }


def save_config(cfg: Dict[str, Any]):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


class Notifier:
    def __init__(self):
        self.cfg = load_config()
        self.bot_token = self.cfg.get("telegram_bot_token", "")
        chat_ids = self.cfg.get("telegram_chat_ids", [])
        if not chat_ids and self.cfg.get("telegram_chat_id"):
            chat_ids = [self.cfg.get("telegram_chat_id")]
        self.chat_ids = [str(cid) for cid in chat_ids if cid]
        self.ntfy_topic = self.cfg.get("ntfy_topic", "")
        self.enabled = self.cfg.get("enabled", True)

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_ids) or bool(self.ntfy_topic)

    def send_raw_text(self, text: str) -> bool:
        """Sends raw markdown text to all registered Telegram chat IDs and Ntfy (100% Free)."""
        if not self.enabled:
            return False

        success = False

        # 1. Telegram Dispatch to all registered chat IDs
        if self.bot_token and self.chat_ids:
            for cid in self.chat_ids:
                try:
                    url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                    payload = {
                        "chat_id": cid,
                        "text": text,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True,
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=8) as response:
                        if response.status == 200:
                            success = True
                except Exception as e:
                    print(f"[Notifier] Telegram error for chat_id {cid}: {e}")

        # 2. Ntfy.sh Dispatch (Free Push App)
        if self.ntfy_topic:
            try:
                url = f"https://ntfy.sh/{self.ntfy_topic}"
                req = urllib.request.Request(
                    url,
                    data=text.encode("utf-8"),
                    headers={
                        "Title": "AI Meta-Orchestrator Alert",
                        "Priority": "high",
                        "Tags": "chart_increasing,moneybag",
                    },
                )
                with urllib.request.urlopen(req, timeout=8) as response:
                    if response.status == 200:
                        success = True
            except Exception as e:
                pass

        return success

    def notify_trade_entry(
        self,
        symbol: str,
        strategy: str,
        entry_px: float,
        qty: int,
        sl_px: float,
        tp_px: float,
        risk_reward: float,
        regime: str,
    ):
        """Sends a high-priority Trade Entry alert to your phone."""
        msg = (
            f"🚀 *AI META-ORCHESTRATOR: NEW TRADE ENTRY*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Symbol*        : `{symbol}`\n"
            f"• *Strategy*      : {strategy}\n"
            f"• *Action*        : BUY CNC (Delivery)\n"
            f"• *Entry Price*   : ₹{entry_px:,.2f}\n"
            f"• *Quantity*      : {qty} shares\n"
            f"• *Invested*      : ₹{entry_px * qty:,.2f}\n"
            f"• *Stop Loss*     : ₹{sl_px:,.2f} 🛑 (1.25x ATR)\n"
            f"• *Target (TP)*   : ₹{tp_px:,.2f} 🎯 (3.50x ATR)\n"
            f"• *Payoff Ratio*  : {risk_reward:.1f}x\n"
            f"• *Market Regime* : `{regime}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👉 *Action*: Place CNC Buy & GTT Order in Zerodha/Groww"
        )
        return self.send_raw_text(msg)

    def notify_trade_exit(
        self,
        symbol: str,
        strategy: str,
        exit_px: float,
        net_pnl: float,
        net_pnl_pct: float,
        reason: str,
        taxes_inr: float,
    ):
        """Sends a Trade Exit alert to your phone."""
        emoji = "🎯" if net_pnl >= 0 else "🛑"
        status_txt = "PROFIT TARGET HIT" if net_pnl >= 0 else "STOP-LOSS HIT / EXIT"
        pnl_str = f"+₹{net_pnl:,.2f} (+{net_pnl_pct:.2f}%)" if net_pnl >= 0 else f"-₹{abs(net_pnl):,.2f} ({net_pnl_pct:.2f}%)"

        msg = (
            f"{emoji} *TRADE CLOSED: {status_txt}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Symbol*     : `{symbol}`\n"
            f"• *Strategy*   : {strategy}\n"
            f"• *Exit Price* : ₹{exit_px:,.2f}\n"
            f"• *Net P&L*    : *{pnl_str}*\n"
            f"• *Taxes Paid* : ₹{taxes_inr:,.2f} (STT/GST/DP)\n"
            f"• *Reason*     : `{reason}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return self.send_raw_text(msg)

    def notify_regime_shift(self, old_regime: str, new_regime: str, nifty_px: float, desc: str):
        """Sends a Macro Regime Shift alert to your phone."""
        msg = (
            f"🛡️ *MACRO REGIME SHIFT DETECTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Previous Regime* : `{old_regime}`\n"
            f"• *New Regime*      : *`{new_regime}`*\n"
            f"• *NIFTY 50 Price*  : ₹{nifty_px:,.2f}\n"
            f"• *Action Taken*    : {desc}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return self.send_raw_text(msg)

    def notify_daily_summary(
        self,
        nav: float,
        cash: float,
        invested: float,
        realized_pnl: float,
        open_positions_count: int,
        regime: str,
    ):
        """Sends daily 3:30 PM post-market portfolio summary to your phone."""
        now_str = datetime.now(IST).strftime("%d %b %Y, %H:%M IST")
        msg = (
            f"📊 *DAILY QUANT PORTFOLIO SUMMARY*\n"
            f"_{now_str}_\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"• *Current NAV*    : *₹{nav:,.2f}*\n"
            f"• *Available Cash* : ₹{cash:,.2f}\n"
            f"• *Invested Value* : ₹{invested:,.2f}\n"
            f"• *Realized P&L*   : ₹{realized_pnl:,.2f}\n"
            f"• *Open Trades*    : {open_positions_count} active\n"
            f"• *Active Regime*  : `{regime}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌐 [Open Web Dashboard](http://localhost:8080)"
        )
        return self.send_raw_text(msg)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Meta-Orchestrator Notification Manager")
    parser.add_argument("--test", action="store_true", help="Send a test notification to your phone")
    parser.add_argument("--set-telegram", nargs=2, metavar=("TOKEN", "CHAT_ID"), help="Set Telegram Bot Token and Chat ID")
    parser.add_argument("--set-ntfy", metavar="TOPIC", help="Set Ntfy.sh topic name for push notifications")
    args = parser.parse_args()

    cfg = load_config()

    if args.set_telegram:
        cfg["telegram_bot_token"] = args.set_telegram[0]
        cfg["telegram_chat_id"] = args.set_telegram[1]
        save_config(cfg)
        print(f"✅ Telegram credentials saved! Bot Token: {args.set_telegram[0][:8]}... | Chat ID: {args.set_telegram[1]}")
        notifier = Notifier()
        success = notifier.send_raw_text("🎉 *AI Meta-Orchestrator Notifications Active!* Your Telegram bot is now successfully connected.")
        if success:
            print("🚀 Test message sent successfully to your Telegram!")
        else:
            print("⚠️ Message failed to send. Please verify Bot Token and Chat ID.")

    elif args.set_ntfy:
        cfg["ntfy_topic"] = args.set_ntfy
        save_config(cfg)
        print(f"✅ Ntfy push topic saved: {args.set_ntfy}")
        notifier = Notifier()
        notifier.send_raw_text("🎉 AI Meta-Orchestrator Ntfy Notifications Connected!")

    elif args.test:
        notifier = Notifier()
        if not notifier.is_configured():
            print("⚠️ No notification credentials configured yet.")
            print("👉 Run: python engine/notifier.py --set-telegram <BOT_TOKEN> <CHAT_ID>")
            print("👉 Or:  python engine/notifier.py --set-ntfy <TOPIC_NAME>")
        else:
            print("Sending test trade alert...")
            notifier.notify_trade_entry("TCS.NS", "Large-Cap RS Pullback", 3850.0, 8, 3782.5, 4086.0, 3.2, "TRENDING_BULL")
            notifier.notify_daily_summary(100000.0, 69200.0, 30800.0, 0.0, 1, "TRENDING_BULL")
            print("✅ Test notifications dispatched!")
    else:
        print("Usage:")
        print("  python engine/notifier.py --set-telegram <BOT_TOKEN> <CHAT_ID>")
        print("  python engine/notifier.py --set-ntfy <TOPIC_NAME>")
        print("  python engine/notifier.py --test")


if __name__ == "__main__":
    main()
