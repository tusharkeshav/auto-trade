# ─────────────────────────────────────────────────────────────────
#  brokers/dhan_broker.py
#  Dhan sandbox/live broker using dhanhq SDK.
#
#  Set env vars before use:
#    DHAN_CLIENT_ID   — your Dhan client ID
#    DHAN_ACCESS_TOKEN — daily access token (refresh via Dhan portal)
#
#  Sandbox vs live: Dhan SDK uses the same endpoint; sandbox mode
#  must be enabled from the Dhan developer portal per-token.
# ─────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

from brokers.base_broker import BaseBroker, BrokerOrderResult, BrokerPosition


class DhanBroker(BaseBroker):
    """
    Dhan broker implementation using dhanhq Python SDK.

    Install: pip install dhanhq
    Docs: https://dhanhq.co/docs/v2/
    """

    def __init__(self, client_id: Optional[str] = None, access_token: Optional[str] = None, sandbox: bool = True):
        self._client_id    = client_id    or os.environ["DHAN_CLIENT_ID"]
        self._access_token = access_token or os.environ["DHAN_ACCESS_TOKEN"]
        self._sandbox      = sandbox
        self._dhan         = self._init_client()

    def _init_client(self):
        try:
            from dhanhq import dhanhq, DhanContext
            from dhanhq.dhan_http import DhanHTTP
            ctx = DhanContext(self._client_id, self._access_token)
            # Override base URL for sandbox mode
            if self._sandbox:
                ctx.dhan_http.base_url = "https://sandbox.dhan.co/v2"
            return dhanhq(ctx)
        except ImportError as e:
            raise ImportError("dhanhq SDK not installed. Run: pip install dhanhq") from e

    @property
    def name(self) -> str:
        return "dhan"

    def get_quote(self, symbol: str) -> float:
        # Dhan LTP feed — symbol must be a security ID string for Dhan
        # Caller is responsible for resolving NSE symbol → Dhan security ID
        resp = self._dhan.get_ltp_data(
            security_id=symbol,
            exchange_segment="NSE_EQ",
            instrument_type="EQUITY",
        )
        return float(resp["data"]["last_price"])

    def place_market_order(
        self,
        symbol:    str,
        direction: str,
        qty:       float,
        product:   str = "MIS",
    ) -> BrokerOrderResult:
        from dhanhq import dhanhq as dh
        transaction = dh.BUY if direction == "BUY" else dh.SELL
        product_map = {"MIS": dh.INTRA, "CNC": dh.CNC, "NRML": dh.MARGIN}
        resp = self._dhan.place_order(
            security_id=symbol,
            exchange_segment=dh.NSE,
            transaction_type=transaction,
            quantity=int(qty),
            order_type=dh.MARKET,
            product_type=product_map.get(product, dh.INTRA),
            price=0,
        )
        order_id = str(resp.get("data", {}).get("orderId", ""))
        success  = resp.get("status") == "success" or bool(order_id)
        logger.info(f"[dhan] {direction} {qty} {symbol} order_id={order_id} status={resp.get('status')}")
        return BrokerOrderResult(
            success=success,
            order_id=order_id,
            symbol=symbol,
            qty=qty,
            price=0.0,   # Dhan market orders return filled price in order detail
            direction=direction,
            order_type="MARKET",
            status="PLACED" if success else "REJECTED",
            message=str(resp.get("remarks", "")),
        )

    def place_limit_order(
        self,
        symbol:      str,
        direction:   str,
        qty:         float,
        limit_price: float,
        product:     str = "MIS",
    ) -> BrokerOrderResult:
        from dhanhq import dhanhq as dh
        transaction = dh.BUY if direction == "BUY" else dh.SELL
        product_map = {"MIS": dh.INTRA, "CNC": dh.CNC, "NRML": dh.MARGIN}
        resp = self._dhan.place_order(
            security_id=symbol,
            exchange_segment=dh.NSE,
            transaction_type=transaction,
            quantity=int(qty),
            order_type=dh.LIMIT,
            product_type=product_map.get(product, dh.INTRA),
            price=limit_price,
        )
        order_id = str(resp.get("data", {}).get("orderId", ""))
        success  = resp.get("status") == "success" or bool(order_id)
        return BrokerOrderResult(
            success=success,
            order_id=order_id,
            symbol=symbol,
            qty=qty,
            price=limit_price,
            direction=direction,
            order_type="LIMIT",
            status="PLACED" if success else "REJECTED",
            message=str(resp.get("remarks", "")),
        )

    def cancel_order(self, order_id: str) -> BrokerOrderResult:
        resp = self._dhan.cancel_order(order_id)
        success = resp.get("status") == "success"
        return BrokerOrderResult(
            success=success,
            order_id=order_id,
            symbol="",
            qty=0,
            price=0,
            direction="",
            order_type="",
            status="CANCELLED" if success else "REJECTED",
            message=str(resp.get("remarks", "")),
        )

    def get_positions(self) -> list[BrokerPosition]:
        resp = self._dhan.get_positions()
        positions = []
        for p in resp.get("data", []):
            positions.append(BrokerPosition(
                symbol=str(p["securityId"]),
                qty=float(p["netQty"]),
                avg_price=float(p["costPrice"]),
                current_price=float(p["lastTradedPrice"]),
                pnl=float(p["unrealizedProfit"]),
                direction="LONG" if float(p["netQty"]) >= 0 else "SHORT",
            ))
        return positions

    def get_balance(self) -> float:
        resp = self._dhan.get_fund_limits()
        if resp.get("status") == "failure":
            raise RuntimeError(f"Dhan API error: {resp.get('remarks')}")
        data = resp.get("data") or {}
        if isinstance(data, str):
            raise RuntimeError(f"Dhan returned unexpected data: {data!r}")
        return float(data.get("availabelBalance", 0.0))
