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
    status       TEXT NOT NULL,
    line_type    TEXT NOT NULL DEFAULT 'primary',
    original_account_code TEXT,
    original_method       TEXT,
    original_confidence   REAL
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
           (txn_id, account_code, amount, confidence, method, reason, status,
            line_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                e.txn_id,
                e.account_code,
                str(e.amount),
                e.confidence,
                e.method,
                e.reason,
                e.status,
                e.line_type,
            )
            for e in entries
        ],
    )
    conn.commit()


def count_payees(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM payee_memory").fetchone()[0]


def lookup_payee(
    conn: sqlite3.Connection, counterparty: str
) -> Optional[tuple[str, int]]:
    """Returns (account_code, times_confirmed) for a known counterparty."""
    row = conn.execute(
        "SELECT account_code, count FROM payee_memory WHERE counterparty = ?",
        (counterparty,),
    ).fetchone()
    return (row["account_code"], row["count"]) if row else None


def remember_payee(
    conn: sqlite3.Connection, counterparty: str, account_code: str
) -> None:
    """Records a human-verified counterparty mapping. This is the feedback
    loop: tier 1 checks payee memory before anything else, so a correction
    here changes what the next run does with that counterparty."""
    row = conn.execute(
        "SELECT account_code, count FROM payee_memory WHERE counterparty = ?",
        (counterparty,),
    ).fetchone()
    if row and row["account_code"] == account_code:
        conn.execute(
            "UPDATE payee_memory SET count = count + 1 WHERE counterparty = ?",
            (counterparty,),
        )
    else:
        # A changed mapping resets the count: the old evidence is now wrong.
        conn.execute(
            """INSERT OR REPLACE INTO payee_memory
               (counterparty, account_code, count) VALUES (?, ?, 1)""",
            (counterparty, account_code),
        )
    conn.commit()


def get_primary_entry(
    conn: sqlite3.Connection, txn_id: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM ledger_entries
           WHERE txn_id = ? AND line_type = 'primary' ORDER BY id LIMIT 1""",
        (txn_id,),
    ).fetchone()


def resolve_entry(
    conn: sqlite3.Connection,
    txn_id: str,
    account_code: str,
    status: str,
    original: Optional[tuple[Optional[str], str, float]] = None,
) -> None:
    """Applies a human decision to the primary line. `original` is recorded
    once, on the first change, so the screen can show what was predicted
    before the human intervened."""
    if original is not None:
        conn.execute(
            """UPDATE ledger_entries
               SET account_code = ?, status = ?,
                   original_account_code = ?, original_method = ?,
                   original_confidence = ?
               WHERE txn_id = ? AND line_type = 'primary'""",
            (account_code, status, *original, txn_id),
        )
    else:
        conn.execute(
            """UPDATE ledger_entries SET account_code = ?, status = ?
               WHERE txn_id = ? AND line_type = 'primary'""",
            (account_code, status, txn_id),
        )
    conn.commit()


def cache_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT response FROM llm_cache WHERE narration_hash = ?", (key,)
    ).fetchone()
    return row["response"] if row else None


def cache_put(conn: sqlite3.Connection, key: str, response: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO llm_cache (narration_hash, response) VALUES (?, ?)",
        (key, response),
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
            line_type=r["line_type"],
            original_account_code=r["original_account_code"],
            original_method=r["original_method"],
            original_confidence=r["original_confidence"],
        )
        for r in rows
    ]
