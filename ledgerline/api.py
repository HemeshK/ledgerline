"""FastAPI backend for the review screen.

Confirming or correcting a row writes the counterparty mapping into
payee_memory, which tier 1 consults before anything else. That is the loop:
a decision made here changes what the next run does.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, data_loader, storage
from .rules import normalize
from .taxes import is_gst_inclusive

STATIC_DIR = config.ROOT / "ledgerline" / "static"

app = FastAPI(title="ledgerline review")


class CorrectionRequest(BaseModel):
    account_code: str


def _load_context():
    batch = data_loader.load_batch(config.BATCH_PATH)
    accounts = data_loader.load_accounts(config.ACCOUNTS_PATH)
    return (
        {t.txn_id: t for t in batch.transactions},
        {a.code: a for a in accounts},
        accounts,
    )


def _serialise(entry, txn, accounts_by_code, tax_lines):
    account = accounts_by_code.get(entry.account_code or "")
    original = accounts_by_code.get(entry.original_account_code or "")
    return {
        "txn_id": entry.txn_id,
        "date": txn.date.isoformat() if txn else None,
        "amount": str(entry.amount),
        "direction": txn.direction if txn else None,
        "narration": txn.narration if txn else "",
        "counterparty": txn.counterparty_raw if txn else "",
        "account_code": entry.account_code,
        "account_name": account.name if account else None,
        "confidence": entry.confidence,
        "method": entry.method,
        "reason": entry.reason,
        "status": entry.status,
        "original_account_code": entry.original_account_code,
        "original_account_name": original.name if original else None,
        "original_method": entry.original_method,
        "original_confidence": entry.original_confidence,
        "gst_inclusive": bool(txn and is_gst_inclusive(txn)),
        "tax_lines": tax_lines,
    }


@app.get("/api/accounts")
def list_accounts():
    _, _, accounts = _load_context()
    return [
        {"code": a.code, "name": a.name, "family": a.family} for a in accounts
    ]


@app.get("/api/entries")
def list_entries(status: Optional[str] = None, q: Optional[str] = None):
    txns, accounts_by_code, _ = _load_context()
    conn = storage.connect(config.DB_PATH)
    storage.init_db(conn)
    entries = storage.load_entries(conn)

    tax_by_txn: dict[str, list] = {}
    for e in entries:
        if e.line_type == "tax":
            account = accounts_by_code.get(e.account_code or "")
            tax_by_txn.setdefault(e.txn_id, []).append(
                {
                    "account_code": e.account_code,
                    "account_name": account.name if account else None,
                    "amount": str(e.amount),
                }
            )

    rows = []
    for e in entries:
        if e.line_type != "primary":
            continue
        if status and status != "all" and e.status != status:
            continue
        txn = txns.get(e.txn_id)
        row = _serialise(e, txn, accounts_by_code, tax_by_txn.get(e.txn_id, []))
        if q:
            needle = q.lower()
            haystack = " ".join(
                str(v) for v in (row["narration"], row["account_code"],
                                 row["reason"], row["txn_id"]) if v
            ).lower()
            if needle not in haystack:
                continue
        rows.append(row)

    counts = {"all": 0, "flagged": 0, "posted": 0, "corrected": 0}
    for e in entries:
        if e.line_type != "primary":
            continue
        counts["all"] += 1
        if e.status in counts:
            counts[e.status] += 1

    conn.close()
    return {"entries": rows, "counts": counts}


def _apply(txn_id: str, account_code: str, status: str):
    txns, accounts_by_code, _ = _load_context()
    if account_code not in accounts_by_code:
        raise HTTPException(400, f"unknown account code {account_code}")

    conn = storage.connect(config.DB_PATH)
    storage.init_db(conn)
    row = storage.get_primary_entry(conn, txn_id)
    if row is None:
        conn.close()
        raise HTTPException(404, f"no ledger entry for {txn_id}")

    # Record what the pipeline predicted, but only the first time a human
    # touches the row, so a second edit does not overwrite the real original.
    original = None
    if row["original_account_code"] is None:
        original = (row["account_code"], row["method"], row["confidence"])

    storage.resolve_entry(conn, txn_id, account_code, status, original)

    txn = txns.get(txn_id)
    counterparty = normalize(txn.counterparty_raw) if txn else ""
    if counterparty:
        storage.remember_payee(conn, counterparty, account_code)

    updated = storage.get_primary_entry(conn, txn_id)
    payees = storage.count_payees(conn)
    conn.close()
    return {
        "txn_id": txn_id,
        "account_code": updated["account_code"],
        "status": updated["status"],
        "remembered": counterparty or None,
        "payee_memory_entries": payees,
    }


@app.post("/api/entries/{txn_id}/confirm")
def confirm(txn_id: str):
    conn = storage.connect(config.DB_PATH)
    storage.init_db(conn)
    row = storage.get_primary_entry(conn, txn_id)
    conn.close()
    if row is None:
        raise HTTPException(404, f"no ledger entry for {txn_id}")
    if not row["account_code"]:
        raise HTTPException(
            400, "this row has no proposed account, so there is nothing to "
                 "confirm - correct it to an account instead"
        )
    return _apply(txn_id, row["account_code"], "posted")


@app.post("/api/entries/{txn_id}/correct")
def correct(txn_id: str, body: CorrectionRequest):
    return _apply(txn_id, body.account_code, "corrected")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
