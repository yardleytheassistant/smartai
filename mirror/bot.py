"""The mirror loop: poll the target's trades and replicate the new ones.

Run with `python -m mirror`.
"""

import sys
import time
from datetime import datetime, timezone

from mirror.config import MirrorConfig, mirror_config
from mirror.data_api import DataAPI, Trade
from mirror.executor import Executor
from mirror.state import SeenStore


def _log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


class MirrorBot:
    def __init__(self, config: MirrorConfig):
        self.config = config
        self.data_api = DataAPI(config.data_api)
        self.executor = Executor(config, self.data_api)
        self.seen = SeenStore(config.state_file)

    def _prime(self) -> None:
        """On a fresh state file, decide where to start so we don't replay history."""
        if not self.seen.is_empty or self.config.backfill:
            return
        trades = self.data_api.get_trades(self.config.target_wallet)
        self.seen.add_many([t.key for t in trades])
        _log(
            f"Primed: marked {len(trades)} existing trade(s) as seen "
            f"(set MIRROR_BACKFILL=1 to copy history instead)."
        )

    def _handle(self, trade: Trade) -> None:
        if self.seen.has(trade.key):
            return
        try:
            result = self.executor.mirror(trade)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive on any order error
            _log(f"ERROR mirroring {trade.side} {trade.title!r}: {exc}")
            return  # do NOT mark seen, so it can be retried next poll
        _log(result.detail if result.detail else f"{trade.side} {trade.title}")
        self.seen.add(trade.key)

    def tick(self) -> None:
        trades = self.data_api.get_trades(self.config.target_wallet)
        for trade in trades:  # oldest -> newest
            self._handle(trade)

    def run(self) -> None:
        mode = "DRY-RUN" if self.config.dry_run else "LIVE"
        _log(f"Mirror bot starting [{mode}]")
        _log(f"  target   : {self.config.target_wallet}")
        _log(f"  ratio    : {self.config.copy_ratio}x  (min ${self.config.min_usdc}, "
             f"max ${self.config.max_usdc}/trade)")
        _log(f"  poll     : every {self.config.poll_seconds}s  order_type={self.config.order_type}")

        if not self.config.dry_run:
            problems = self.config.validate_for_live()
            if problems:
                for p in problems:
                    _log(f"  CONFIG ERROR: {p}")
                _log("Refusing to run live with the above unset. Exiting.")
                sys.exit(1)

        self._prime()
        while True:
            try:
                self.tick()
            except KeyboardInterrupt:
                _log("Interrupted, shutting down.")
                return
            except Exception as exc:  # noqa: BLE001 - transient API/network errors
                _log(f"poll error: {exc}")
            time.sleep(self.config.poll_seconds)


def main() -> None:
    MirrorBot(mirror_config).run()


if __name__ == "__main__":
    main()
