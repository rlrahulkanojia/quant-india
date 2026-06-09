"""SQLite-backed store for paper trading state.

Persists portfolio, positions, orders, and shadow fills using raw sqlite3
with WAL journal mode for safe concurrent reads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now().isoformat()


class PaperStore:
    """SQLite-backed store for the paper trading engine.

    Uses WAL mode, a threading lock for write serialisation, and
    ``sqlite3.Row`` row factory so callers get dict-like access.
    Public methods return plain dicts via ``_row_to_dict``.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS paper_portfolio (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    cash_balance  REAL    NOT NULL,
                    total_realized REAL   NOT NULL DEFAULT 0.0,
                    created_at    TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_positions (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol         TEXT    NOT NULL,
                    exchange       TEXT    NOT NULL,
                    qty            INTEGER NOT NULL,
                    avg_price      REAL    NOT NULL,
                    current_price  REAL    NOT NULL DEFAULT 0.0,
                    unrealized_pnl REAL    NOT NULL DEFAULT 0.0,
                    side           TEXT    NOT NULL,
                    opened_at      TEXT    NOT NULL,
                    UNIQUE(symbol, exchange)
                );

                CREATE TABLE IF NOT EXISTS paper_orders (
                    id             TEXT    PRIMARY KEY,
                    symbol         TEXT    NOT NULL,
                    exchange       TEXT    NOT NULL,
                    side           TEXT    NOT NULL,
                    order_type     TEXT    NOT NULL,
                    qty            INTEGER NOT NULL,
                    limit_price    REAL,
                    fill_price     REAL,
                    slippage       REAL,
                    fees_total     REAL,
                    fees_breakdown TEXT,
                    status         TEXT    NOT NULL,
                    filled_qty     INTEGER NOT NULL DEFAULT 0,
                    created_at     TEXT    NOT NULL,
                    filled_at      TEXT
                );

                CREATE TABLE IF NOT EXISTS shadow_fills (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_order_id   TEXT    NOT NULL,
                    paper_fill_price REAL    NOT NULL,
                    market_ltp       REAL    NOT NULL,
                    market_bid       REAL    NOT NULL,
                    market_ask       REAL    NOT NULL,
                    divergence_pct   REAL    NOT NULL,
                    qty              INTEGER NOT NULL DEFAULT 1,
                    captured_at      TEXT    NOT NULL,
                    FOREIGN KEY(paper_order_id) REFERENCES paper_orders(id)
                );
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def init_portfolio(self, starting_cash: float) -> dict:
        """Create the initial portfolio record."""
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO paper_portfolio (cash_balance, total_realized, created_at, updated_at)
                VALUES (?, 0.0, ?, ?)
                """,
                (starting_cash, now, now),
            )
            self._conn.commit()
        return self.get_portfolio()  # type: ignore[return-value]

    def get_portfolio(self) -> dict | None:
        """Return the portfolio record, or None if uninitialised."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM paper_portfolio ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def update_cash(self, delta: float, realized_delta: float = 0.0) -> None:
        """Adjust cash balance and optionally accumulate realised P&L."""
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                UPDATE paper_portfolio
                SET cash_balance  = cash_balance + ?,
                    total_realized = total_realized + ?,
                    updated_at    = ?
                WHERE id = (SELECT MAX(id) FROM paper_portfolio)
                """,
                (delta, realized_delta, now),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def save_order(self, order: dict) -> None:
        """Persist an order dict.  ``fees_breakdown`` is stored as JSON text."""
        fees_json = json.dumps(order.get("fees_breakdown") or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO paper_orders
                    (id, symbol, exchange, side, order_type, qty, limit_price,
                     fill_price, slippage, fees_total, fees_breakdown, status,
                     filled_qty, created_at, filled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order["id"],
                    order["symbol"],
                    order["exchange"],
                    order["side"],
                    order["order_type"],
                    order["qty"],
                    order.get("limit_price"),
                    order.get("fill_price"),
                    order.get("slippage"),
                    order.get("fees_total"),
                    fees_json,
                    order["status"],
                    order.get("filled_qty", 0),
                    order["created_at"],
                    order.get("filled_at"),
                ),
            )
            self._conn.commit()

    def get_order(self, order_id: str) -> dict | None:
        """Return a single order by id, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM paper_orders WHERE id = ?", (order_id,)
            ).fetchone()
        if row is None:
            return None
        d = self._row_to_dict(row)
        d["fees_breakdown"] = json.loads(d["fees_breakdown"] or "{}")
        return d

    def list_orders(self) -> list[dict]:
        """Return all orders, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM paper_orders ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for row in rows:
            d = self._row_to_dict(row)
            d["fees_breakdown"] = json.loads(d["fees_breakdown"] or "{}")
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def save_position(self, position: dict) -> int:
        """Insert a position and return its auto-generated id."""
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO paper_positions
                    (symbol, exchange, qty, avg_price, current_price,
                     unrealized_pnl, side, opened_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position["symbol"],
                    position["exchange"],
                    position["qty"],
                    position["avg_price"],
                    position.get("current_price", 0.0),
                    position.get("unrealized_pnl", 0.0),
                    position["side"],
                    position["opened_at"],
                ),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_position_by_symbol(self, symbol: str, exchange: str) -> dict | None:
        """Look up a position by (symbol, exchange) unique pair."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM paper_positions WHERE symbol = ? AND exchange = ?",
                (symbol, exchange),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM paper_positions ORDER BY opened_at"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_position(self, position_id: int, updates: dict) -> None:
        """Update selected columns on a position row."""
        if not updates:
            return
        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [position_id]
        with self._lock:
            self._conn.execute(
                f"UPDATE paper_positions SET {set_clauses} WHERE id = ?",
                values,
            )
            self._conn.commit()

    def delete_position(self, position_id: int) -> None:
        """Remove a position by id (used when fully closed)."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM paper_positions WHERE id = ?", (position_id,)
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Shadow fills
    # ------------------------------------------------------------------

    def save_shadow_fill(self, fill: dict) -> None:
        """Persist a shadow fill record for post-session analysis."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO shadow_fills
                    (paper_order_id, paper_fill_price, market_ltp,
                     market_bid, market_ask, divergence_pct, qty, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill["paper_order_id"],
                    fill["paper_fill_price"],
                    fill["market_ltp"],
                    fill["market_bid"],
                    fill["market_ask"],
                    fill["divergence_pct"],
                    fill.get("qty", 1),
                    fill["captured_at"],
                ),
            )
            self._conn.commit()

    def get_shadow_fills(self, paper_order_id: str) -> list[dict]:
        """Return all shadow fills for a given paper order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM shadow_fills WHERE paper_order_id = ? ORDER BY captured_at",
                (paper_order_id,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_all_shadow_fills_since(self, since_iso: str) -> list[dict]:
        """Return all shadow fills captured on or after *since_iso* (ISO-8601)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM shadow_fills WHERE captured_at >= ? ORDER BY captured_at",
                (since_iso,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]
