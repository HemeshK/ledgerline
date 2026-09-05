"""SQLite persistence. Amounts are stored as TEXT so Decimal round-trips exactly."""

import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Optional

from .models import LedgerEntry

# One transaction can produce several ledger lines (a GST-inclusive row splits
# into base plus tax), so entries are keyed by a surrogate id, not by txn_id.
SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_entries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id       TEXT NOT NULL,
    account_code TEXT,
    amount       TEXT NOT NULL,
    confidence   REAL NOT NULL,
    method       TEXT NOT NULL,
    reason       TEXT NOT NULL,
    status       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_txn ON ledger_entries(txn_id);
CREATE INDEX IF NOT EXISTS idx_ledger_status ON ledger_entries(status);

CREATE TABLE IF NOT EXISTS payee_memory (
    counterparty TEXT PRIMARY KEY,
    account_code TEXT NOT NULL,
    count        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS llm_cache (
    narration_hash TEXT PRIMARY KEY,
    response       TEXT NOT NULL
);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def save_entries(conn: sqlite3.Connection, entries: list[LedgerEntry]) -> None:
    """Replaces any existing lines for the transactions being written, so a
    re-run overwrites its previous result instead of accumulating duplicates."""
    txn_ids = {e.txn_id for e in entries}
    conn.executemany(
        "DELETE FROM ledger_entries WHERE txn_id = ?", [(t,) for t in txn_ids]
    )
    conn.executemany(
        """INSERT INTO ledger_entries
           (txn_id, account_code, amount, confidence, method, reason, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                e.txn_id,
                e.account_code,
                str(e.amount),
                e.confidence,
                e.method,
                e.reason,
                e.status,
            )
            for e in entries
        ],
    )
    conn.commit()


def load_entries(
    conn: sqlite3.Connection, status: Optional[str] = None
) -> list[LedgerEntry]:
    if status:
        rows = conn.execute(
            "SELECT * FROM ledger_entries WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM ledger_entries ORDER BY id").fetchall()
    return [
        LedgerEntry(
            txn_id=r["txn_id"],
            account_code=r["account_code"],
            amount=Decimal(r["amount"]),
            confidence=r["confidence"],
            method=r["method"],
            reason=r["reason"],
            status=r["status"],
        )
        for r in rows
    ]
