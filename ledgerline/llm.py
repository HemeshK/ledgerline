"""Tier 3: the LLM. Most expensive, least reliable, called last and least.

Every response is cached, so a second run over the same batch costs nothing.
The cache key includes direction as well as narration: direction is a hard
constraint in the prompt, and three narrations in the batch appear with both
directions, so keying on narration alone would replay a verdict that was
computed under the opposite constraint.
"""

import hashlib
import json
import random
import sqlite3
import time
from typing import Optional

from pydantic import BaseModel

from . import storage
from .models import Account, ClassificationResult, Transaction

MODEL_NAME = "gemini-2.5-flash-lite"
MAX_RETRIES = 5
BASE_BACKOFF = 1.0
REFUSAL_TOKEN = "NONE"


class LLMVerdict(BaseModel):
    """Structured output schema handed to the model."""

    account_code: str
    confidence: float
    reason: str


def build_chart_block(accounts: list[Account]) -> str:
    lines = ["code | name | family | hints"]
    for a in accounts:
        lines.append(
            f"{a.code} | {a.name} | {a.family} | {', '.join(a.hints)}"
        )
    return "\n".join(lines)


def build_prompt(chart_block: str, txn: Transaction) -> str:
    return f"""You are a finance controller categorising one bank transaction from an Indian SMB's statement into a chart of accounts.

CHART OF ACCOUNTS
{chart_block}

HARD CONSTRAINT - TRANSACTION DIRECTION
The direction of the transaction restricts which accounts are legal. This is a
constraint, not a preference:
- direction "credit": revenue-family accounts are allowed. Expense-family
  accounts are NOT allowed under any circumstances.
- direction "debit": expense-family accounts are allowed. Revenue-family
  accounts are NOT allowed under any circumstances.
- asset, liability and equity accounts are allowed in either direction.
An expense account on a credit, or a revenue account on a debit, is always
wrong regardless of how well the words match.

WHEN TO REFUSE
If the narration carries no usable signal - a bare account or phone number, a
generic "PAYMENT" or "TRF" with no payee and no category - return
account_code "{REFUSAL_TOKEN}" and say what is missing. Refusing is the correct
answer for such rows, and is scored as correct. A confidently wrong account is
worse than an honest refusal. Do not guess to fill the field.

OTHER NOTES
- Equipment and durable hardware (laptops, monitors, printers, furniture) are
  capital purchases and belong on the balance sheet, not in an expense account.
- Money moved between the business's own accounts is a transfer, not income.

TRANSACTION
narration: {txn.narration}
direction: {txn.direction}
amount: {txn.amount}
counterparty: {txn.counterparty_raw or "(none given)"}

Respond with JSON: account_code (a code from the chart, or "{REFUSAL_TOKEN}"),
confidence (0.0-1.0), reason (one sentence a human accountant can act on)."""


def _cache_key(txn: Transaction) -> str:
    payload = f"{txn.narration}|{txn.direction}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _is_rate_limit(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    text = str(exc).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text


class LLMTier:
    name = "llm"

    def __init__(
        self,
        accounts: list[Account],
        conn: sqlite3.Connection,
        api_key: str,
        model_name: str = MODEL_NAME,
    ):
        from google import genai

        self.conn = conn
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.chart_block = build_chart_block(accounts)
        self.valid_codes = {a.code for a in accounts}
        self.calls = 0
        self.cache_hits = 0
        self.scored = 0
        self.predictions: dict[str, tuple[Optional[str], float]] = {}

    def classify(self, txn: Transaction) -> ClassificationResult:
        self.scored += 1
        key = _cache_key(txn)

        cached = storage.cache_get(self.conn, key)
        if cached:
            self.cache_hits += 1
            return self._to_result(json.loads(cached), txn, cached=True)

        raw = self._call_with_backoff(build_prompt(self.chart_block, txn))
        storage.cache_put(self.conn, key, raw)
        self.calls += 1
        return self._to_result(json.loads(raw), txn, cached=False)

    def _call_with_backoff(self, prompt: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LLMVerdict,
            temperature=0.0,
        )
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt, config=config
                )
                return response.text
            except Exception as exc:
                if not _is_rate_limit(exc) or attempt == MAX_RETRIES - 1:
                    raise
                delay = BASE_BACKOFF * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def _to_result(
        self, payload: dict, txn: Transaction, cached: bool
    ) -> ClassificationResult:
        code = (payload.get("account_code") or "").strip()
        confidence = float(payload.get("confidence") or 0.0)
        reason = payload.get("reason") or ""
        suffix = " (cached)" if cached else ""

        if code == REFUSAL_TOKEN or not code:
            self.predictions[txn.txn_id] = (None, 0.0)
            return ClassificationResult(
                account_code=None,
                confidence=0.0,
                method="llm",
                reason=f"LLM declined to classify: {reason}{suffix}",
            )

        if code not in self.valid_codes:
            self.predictions[txn.txn_id] = (None, 0.0)
            return ClassificationResult(
                account_code=None,
                confidence=0.0,
                method="llm",
                reason=(
                    f"LLM returned '{code}', which is not in the chart of "
                    f"accounts; treated as no answer{suffix}"
                ),
            )

        self.predictions[txn.txn_id] = (code, confidence)
        return ClassificationResult(
            account_code=code,
            confidence=confidence,
            method="llm",
            reason=f"LLM: {reason}{suffix}",
        )
