"""Configuration for the Polymarket mirror bot.

Everything is environment-driven (see .env.example). The bot is intentionally
**dry-run by default**: it will log the orders it *would* place but not send
any real money until you explicitly set MIRROR_DRY_RUN=0.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# The wallet from the Polymarket profile URL the bot mirrors by default.
# https://polymarket.com/@0xf3ce251f9c4ae0f3940a9f32de5dd1a1d05b8bc6-1778725794122
# (the trailing "-<digits>" is just a profile display id; the address is the
# 0x... part.)
DEFAULT_TARGET = "0xf3ce251f9c4ae0f3940a9f32de5dd1a1d05b8bc6"


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class MirrorConfig:
    # --- Who to copy -------------------------------------------------------
    target_wallet: str = field(
        default_factory=lambda: os.getenv("MIRROR_TARGET_WALLET", DEFAULT_TARGET).lower()
    )

    # --- Polymarket endpoints ---------------------------------------------
    data_api: str = field(
        default_factory=lambda: os.getenv("POLY_DATA_API", "https://data-api.polymarket.com")
    )
    clob_host: str = field(
        default_factory=lambda: os.getenv("POLY_CLOB_HOST", "https://clob.polymarket.com")
    )
    chain_id: int = field(default_factory=lambda: int(os.getenv("POLY_CHAIN_ID", "137")))

    # --- Your account (only needed when MIRROR_DRY_RUN=0) -----------------
    private_key: str = field(default_factory=lambda: os.getenv("POLY_PRIVATE_KEY", ""))
    # The address that actually holds your USDC / positions. For the common
    # Polymarket email/Magic wallet this is your proxy ("funder") address.
    funder: str = field(default_factory=lambda: os.getenv("POLY_FUNDER", "").lower())
    # 0 = EOA (MetaMask/hardware), 1 = email/Magic proxy, 2 = browser proxy.
    signature_type: int = field(
        default_factory=lambda: int(os.getenv("POLY_SIGNATURE_TYPE", "1"))
    )

    # --- Sizing & risk guards ---------------------------------------------
    # Multiplier applied to the target's USDC size. 1.0 = copy the exact dollar
    # amount; 0.1 = copy at one tenth the size.
    copy_ratio: float = field(default_factory=lambda: float(os.getenv("MIRROR_COPY_RATIO", "1.0")))
    # Skip trades whose mirrored size would be below this many USDC (dust).
    min_usdc: float = field(default_factory=lambda: float(os.getenv("MIRROR_MIN_USDC", "1.0")))
    # Hard cap on the USDC committed to any single mirrored trade.
    max_usdc: float = field(default_factory=lambda: float(os.getenv("MIRROR_MAX_USDC", "100.0")))

    # --- Loop & execution --------------------------------------------------
    poll_seconds: float = field(
        default_factory=lambda: float(os.getenv("MIRROR_POLL_SECONDS", "15"))
    )
    # FOK = all-or-nothing immediate fill; FAK = fill what you can, cancel rest.
    order_type: str = field(
        default_factory=lambda: os.getenv("MIRROR_ORDER_TYPE", "FOK").strip().upper()
    )
    dry_run: bool = field(default_factory=lambda: _get_bool("MIRROR_DRY_RUN", True))
    # On the first run, also replay the target's existing trade history.
    # Off by default so you only start copying *new* trades.
    backfill: bool = field(default_factory=lambda: _get_bool("MIRROR_BACKFILL", False))
    state_file: str = field(
        default_factory=lambda: os.getenv("MIRROR_STATE_FILE", "./mirror_state.json")
    )

    def validate_for_live(self) -> list[str]:
        """Return a list of problems that would block live (non-dry-run) trading."""
        problems = []
        if not self.private_key:
            problems.append("POLY_PRIVATE_KEY is not set.")
        if not self.funder:
            problems.append("POLY_FUNDER (your proxy/funder address) is not set.")
        if self.order_type not in {"FOK", "FAK"}:
            problems.append(f"MIRROR_ORDER_TYPE must be FOK or FAK, got {self.order_type!r}.")
        if self.copy_ratio <= 0:
            problems.append("MIRROR_COPY_RATIO must be greater than 0.")
        return problems


mirror_config = MirrorConfig()
