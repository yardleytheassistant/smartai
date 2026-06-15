"""Polymarket copy-trading (mirror) bot.

Watches a target wallet's on-chain trades via Polymarket's public Data API and
replicates each new trade on your own account through the CLOB order API.

Run it with:

    python -m mirror            # uses settings from .env

See mirror/config.py for all settings and README (Mirror bot section) for the
full walkthrough and safety notes.
"""

from mirror.config import mirror_config

__all__ = ["mirror_config"]
