"""Turns a target trade into an order on your own Polymarket account.

Wraps Polymarket's CLOB client. In dry-run mode (the default) it imports
nothing heavy and just reports the order it *would* have placed, so you can
watch the bot end-to-end before risking funds.

Order semantics that matter for mirroring (py-clob-client market orders):
  * BUY  -> `amount` is the USDC to spend.
  * SELL -> `amount` is the number of shares to sell.
"""

from dataclasses import dataclass

from mirror.config import MirrorConfig
from mirror.data_api import DataAPI, Trade


@dataclass
class OrderResult:
    placed: bool
    detail: str


class Executor:
    def __init__(self, config: MirrorConfig, data_api: DataAPI):
        self.config = config
        self.data_api = data_api
        self._client = None  # built lazily on first live order

    # --- CLOB client (only constructed for live trading) -------------------
    def _clob(self):
        if self._client is not None:
            return self._client
        # Imported here so dry-run / data-only use never needs the dependency.
        from py_clob_client.client import ClobClient

        client = ClobClient(
            self.config.clob_host,
            key=self.config.private_key,
            chain_id=self.config.chain_id,
            signature_type=self.config.signature_type,
            funder=self.config.funder,
        )
        client.set_api_creds(client.create_or_derive_api_creds())
        self._client = client
        return client

    # --- Sizing ------------------------------------------------------------
    def _mirror_usdc(self, trade: Trade) -> float:
        """USDC to commit for a mirrored BUY, after ratio + caps."""
        amount = trade.usdc_size * self.config.copy_ratio
        return min(amount, self.config.max_usdc)

    # --- Public API --------------------------------------------------------
    def mirror(self, trade: Trade) -> OrderResult:
        if trade.side == "BUY":
            return self._mirror_buy(trade)
        if trade.side == "SELL":
            return self._mirror_sell(trade)
        return OrderResult(False, f"skip: unknown side {trade.side!r}")

    def _mirror_buy(self, trade: Trade) -> OrderResult:
        usdc = self._mirror_usdc(trade)
        if usdc < self.config.min_usdc:
            return OrderResult(False, f"skip: ${usdc:.2f} below min ${self.config.min_usdc:.2f}")

        label = f"BUY ${usdc:.2f} of '{trade.outcome}' @ ~{trade.price:.3f} — {trade.title}"
        if self.config.dry_run:
            return OrderResult(False, f"[dry-run] would {label}")
        return self._post_market(trade, side="BUY", amount=usdc, label=label)

    def _mirror_sell(self, trade: Trade) -> OrderResult:
        # Sell in proportion to the target, but never more shares than we hold.
        want_shares = trade.size * self.config.copy_ratio
        if not self.config.dry_run:
            held = self.data_api.get_position_size(self.config.funder, trade.asset)
            shares = min(want_shares, held)
        else:
            shares = want_shares  # we don't know holdings in dry-run; just report intent

        est_usdc = shares * trade.price
        if est_usdc < self.config.min_usdc or shares <= 0:
            return OrderResult(False, f"skip: nothing to sell (≈${est_usdc:.2f})")

        label = f"SELL {shares:.2f} shares of '{trade.outcome}' @ ~{trade.price:.3f} — {trade.title}"
        if self.config.dry_run:
            return OrderResult(False, f"[dry-run] would {label}")
        return self._post_market(trade, side="SELL", amount=shares, label=label)

    def _post_market(self, trade: Trade, *, side: str, amount: float, label: str) -> OrderResult:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        order_type = OrderType.FOK if self.config.order_type == "FOK" else OrderType.FAK
        clob = self._clob()
        order = MarketOrderArgs(
            token_id=trade.asset,
            amount=round(amount, 2),
            side=BUY if side == "BUY" else SELL,
            order_type=order_type,
        )
        signed = clob.create_market_order(order)
        resp = clob.post_order(signed, order_type)
        return OrderResult(True, f"{label} -> {resp}")
