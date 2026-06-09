"""SQLite-backed store for trade decisions, reflections, and calibrations.

Persists the full decision lifecycle: initial trade decisions from the
debate engine, periodic reflection reports, and calibration parameter
updates — using raw sqlite3 with WAL journal mode for safe concurrent reads.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now().isoformat()


class DecisionStore:
    """SQLite-backed store for trade decision memory.

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
                CREATE TABLE IF NOT EXISTS trade_decisions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT NOT NULL,
                    action          TEXT NOT NULL,
                    confidence      INTEGER NOT NULL,
                    debate_log      TEXT,
                    analysis_report TEXT,
                    signal_type     TEXT,
                    entry_price     REAL,
                    exit_price      REAL,
                    pnl_amount      REAL,
                    pnl_pct         REAL,
                    hold_duration   INTEGER,
                    exit_reason     TEXT,
                    paper_order_id  TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    closed_at       TEXT
                );

                CREATE TABLE IF NOT EXISTS reflection_reports (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_start    TEXT NOT NULL,
                    period_end      TEXT NOT NULL,
                    total_trades    INTEGER NOT NULL,
                    win_rate        REAL NOT NULL,
                    avg_pnl_pct     REAL NOT NULL,
                    sharpe_ratio    REAL NOT NULL,
                    findings        TEXT,
                    recommendations TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS calibration_updates (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    reflection_id   INTEGER REFERENCES reflection_reports(id),
                    parameter       TEXT NOT NULL,
                    old_value       REAL NOT NULL,
                    new_value       REAL NOT NULL,
                    reason          TEXT NOT NULL,
                    applied_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    reverted_at     TEXT
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

    @staticmethod
    def _deserialize_json_fields(d: dict, fields: list[str]) -> dict:
        """Parse JSON strings back into Python objects for given fields."""
        for field in fields:
            if field in d and d[field] is not None:
                d[field] = json.loads(d[field])
        return d

    # ------------------------------------------------------------------
    # Trade decisions
    # ------------------------------------------------------------------

    def save_decision(self, data: dict) -> int:
        """Persist a trade decision and return its row id."""
        debate_log = json.dumps(data.get("debate_log"), ensure_ascii=False) \
            if data.get("debate_log") is not None else None
        analysis_report = json.dumps(data.get("analysis_report"), ensure_ascii=False) \
            if data.get("analysis_report") is not None else None

        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO trade_decisions
                    (symbol, action, confidence, debate_log, analysis_report,
                     signal_type, entry_price, paper_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["symbol"],
                    data["action"],
                    data["confidence"],
                    debate_log,
                    analysis_report,
                    data.get("signal_type"),
                    data.get("entry_price"),
                    data.get("paper_order_id"),
                ),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_decision(self, decision_id: int) -> dict | None:
        """Return a single trade decision by id, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trade_decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        if row is None:
            return None
        d = self._row_to_dict(row)
        return self._deserialize_json_fields(d, ["debate_log", "analysis_report"])

    def list_decisions(
        self, since_date: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Return trade decisions, newest first.

        If *since_date* is given (ISO-8601 date or datetime), only rows
        with ``created_at >= since_date`` are returned.
        """
        if since_date is not None:
            query = (
                "SELECT * FROM trade_decisions WHERE created_at >= ? "
                "ORDER BY id DESC LIMIT ?"
            )
            params: tuple = (since_date, limit)
        else:
            query = "SELECT * FROM trade_decisions ORDER BY id DESC LIMIT ?"
            params = (limit,)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        result = []
        for row in rows:
            d = self._row_to_dict(row)
            result.append(self._deserialize_json_fields(d, ["debate_log", "analysis_report"]))
        return result

    def close_decision(
        self,
        decision_id: int,
        exit_price: float,
        pnl_amount: float,
        pnl_pct: float,
        hold_duration: int,
        exit_reason: str,
    ) -> None:
        """Fill exit fields on a trade decision when the position is closed."""
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                """
                UPDATE trade_decisions
                SET exit_price    = ?,
                    pnl_amount    = ?,
                    pnl_pct       = ?,
                    hold_duration = ?,
                    exit_reason   = ?,
                    closed_at     = ?
                WHERE id = ?
                """,
                (exit_price, pnl_amount, pnl_pct, hold_duration, exit_reason, now, decision_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Reflection reports
    # ------------------------------------------------------------------

    def save_reflection(self, data: dict) -> int:
        """Persist a reflection report and return its row id."""
        findings = json.dumps(data.get("findings"), ensure_ascii=False) \
            if data.get("findings") is not None else None
        recommendations = json.dumps(data.get("recommendations"), ensure_ascii=False) \
            if data.get("recommendations") is not None else None

        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO reflection_reports
                    (period_start, period_end, total_trades, win_rate,
                     avg_pnl_pct, sharpe_ratio, findings, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["period_start"],
                    data["period_end"],
                    data["total_trades"],
                    data["win_rate"],
                    data["avg_pnl_pct"],
                    data["sharpe_ratio"],
                    findings,
                    recommendations,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_latest_reflection(self) -> dict | None:
        """Return the most recent reflection report, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reflection_reports ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        d = self._row_to_dict(row)
        return self._deserialize_json_fields(d, ["findings", "recommendations"])

    # ------------------------------------------------------------------
    # Calibration updates
    # ------------------------------------------------------------------

    def save_calibration_update(self, data: dict) -> int:
        """Persist a calibration parameter change and return its row id."""
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO calibration_updates
                    (reflection_id, parameter, old_value, new_value, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    data.get("reflection_id"),
                    data["parameter"],
                    data["old_value"],
                    data["new_value"],
                    data["reason"],
                ),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_active_calibrations(self) -> list[dict]:
        """Return all calibration updates that have not been reverted."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM calibration_updates WHERE reverted_at IS NULL "
                "ORDER BY applied_at DESC"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def revert_calibration(self, calibration_id: int) -> None:
        """Mark a calibration update as reverted."""
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                "UPDATE calibration_updates SET reverted_at = ? WHERE id = ?",
                (now, calibration_id),
            )
            self._conn.commit()
