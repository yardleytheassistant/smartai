"""Read-only client for Polymarket's public Data API.

We use this for two things:
  * pull the *target* wallet's trades (the activity feed), and
  * look up *our own* current positions so we can size mirrored SELLs.

Docs: https://docs.polymarket.com/api-reference  (base: data-api.polymarket.com)
"""

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class Trade:
    """One TRADE activity entry from the target wallet."""

    transaction_hash: str
    asset: str  # ERC-1155 token id == the CLOB token_id we trade on
    condition_id: str
    side: str  # "BUY" or "SELL"
    size: float  # number of outcome shares
    usdc_size: float  # USDC value of the trade
    price: float
    timestamp: int  # unix seconds
    outcome: str  # e.g. "Yes" / "No"
    title: str  # market title (for logging)

    @property
    def key(self) -> str:
        """Stable de-duplication key for a single fill."""
        return f"{self.transaction_hash}:{self.asset}:{self.side}:{self.size}:{self.price}"

    @classmethod
    def from_activity(cls, item: dict) -> "Trade":
        return cls(
            transaction_hash=str(item.get("transactionHash", "")),
            asset=str(item.get("asset", "")),
            condition_id=str(item.get("conditionId", "")),
            side=str(item.get("side", "")).upper(),
            size=float(item.get("size", 0) or 0),
            usdc_size=float(item.get("usdcSize", 0) or 0),
            price=float(item.get("price", 0) or 0),
            timestamp=int(item.get("timestamp", 0) or 0),
            outcome=str(item.get("outcome", "")),
            title=str(item.get("title", "")),
        )


class DataAPI:
    def __init__(self, base_url: str, *, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def get_trades(self, user: str, *, limit: int = 100) -> list[Trade]:
        """Return the user's recent TRADE activity, oldest first.

        We request newest-first from the API (its default) then reverse, so the
        caller processes trades in the order they happened.
        """
        resp = self.session.get(
            f"{self.base_url}/activity",
            params={
                "user": user,
                "type": "TRADE",
                "limit": limit,
                "sortBy": "TIMESTAMP",
                "sortDirection": "DESC",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        items = resp.json() or []
        trades = [Trade.from_activity(it) for it in items if it.get("type") == "TRADE"]
        trades.sort(key=lambda t: t.timestamp)
        return trades

    def get_position_size(self, user: str, asset: str) -> float:
        """Return how many shares `user` currently holds of `asset` (0 if none)."""
        resp = self.session.get(
            f"{self.base_url}/positions",
            params={"user": user, "sizeThreshold": 0},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        for pos in resp.json() or []:
            if str(pos.get("asset", "")) == asset:
                return float(pos.get("size", 0) or 0)
        return 0.0
