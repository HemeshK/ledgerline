"""Tier 1: exact payee memory, then keyword matching against account hints.

Two things keep this tier honest. Keywords match on word boundaries, so "itc"
does not fire inside "SWITCH". And an account's family must agree with the
transaction's direction: revenue accounts only match credits, expense accounts
only match debits. That direction check is what separates 4200 Service income
from 6500 Professional fees, which share vocabulary the keywords cannot split.
"""

import re
import sqlite3
from typing import Optional

from . import storage
from .models import Account, ClassificationResult, Direction, Transaction

EXACT_PAYEE_BASE = 0.95
EXACT_PAYEE_MAX = 0.99
STRENGTH_MULTIWORD = 0.85
STRENGTH_LONG_WORD = 0.78
STRENGTH_SHORT_WORD = 0.68
LONG_WORD_CHARS = 6


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def _hint_pattern(hint: str) -> re.Pattern:
    words = normalize(hint).split()
    return re.compile(r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b")


def _hint_strength(hint: str) -> float:
    words = normalize(hint).split()
    if len(words) > 1:
        return STRENGTH_MULTIWORD
    return (
        STRENGTH_LONG_WORD
        if len(words[0]) >= LONG_WORD_CHARS
        else STRENGTH_SHORT_WORD
    )


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def direction_allows(account: Account, direction: Direction) -> bool:
    if account.family == "revenue":
        return direction == "credit"
    if account.family == "expense":
        return direction == "debit"
    return True


class RuleTier:
    name = "rule"

    def __init__(self, accounts: list[Account], conn: sqlite3.Connection):
        self.conn = conn
        self._compiled = [
            (a, [(h, _hint_pattern(h), _hint_strength(h)) for h in a.hints])
            for a in accounts
        ]

    def classify(self, txn: Transaction) -> ClassificationResult:
        payee = normalize(txn.counterparty_raw)
        if payee:
            known = storage.lookup_payee(self.conn, payee)
            if known:
                code, count = known
                confidence = min(EXACT_PAYEE_MAX, EXACT_PAYEE_BASE + 0.01 * (count - 1))
                return ClassificationResult(
                    account_code=code,
                    confidence=confidence,
                    method="rule",
                    reason=(
                        f"counterparty '{payee}' is in payee memory, mapped to "
                        f"{code} after {count} confirmed posting(s)"
                    ),
                )

        text = normalize(txn.narration)
        matched: list[tuple[Account, str, float]] = []
        wrong_direction: list[tuple[Account, str]] = []

        for account, hints in self._compiled:
            best: Optional[tuple[str, float]] = None
            for hint, pattern, strength in hints:
                if pattern.search(text) and (best is None or strength > best[1]):
                    best = (hint, strength)
            if best is None:
                continue
            if direction_allows(account, txn.direction):
                matched.append((account, best[0], best[1]))
            else:
                wrong_direction.append((account, best[0]))

        if not matched:
            return ClassificationResult(
                account_code=None,
                confidence=0.0,
                method="rule",
                reason=self._no_match_reason(txn, payee, wrong_direction),
            )

        matched.sort(key=lambda m: m[2], reverse=True)
        top_strength = matched[0][2]
        tied = [m for m in matched if m[2] == top_strength]

        if len(tied) > 1:
            names = ", ".join(f"{a.code} {a.name} ('{h}')" for a, h, _ in tied)
            return ClassificationResult(
                account_code=None,
                confidence=0.0,
                method="rule",
                reason=(
                    f"narration matches {len(tied)} accounts equally well: {names}. "
                    "A human needs to pick which one this is."
                ),
            )

        account, hint, strength = matched[0]
        reason = (
            f"narration contains '{hint}', which maps to {account.code} "
            f"{account.name}; a {txn.direction} is consistent with "
            f"{_article(account.family)} {account.family} account"
        )
        if len(matched) > 1:
            runner = matched[1][0]
            reason += f" (weaker competing match: {runner.code} {runner.name})"

        return ClassificationResult(
            account_code=account.code,
            confidence=strength,
            method="rule",
            reason=reason,
        )

    def _no_match_reason(
        self,
        txn: Transaction,
        payee: str,
        wrong_direction: list[tuple[Account, str]],
    ) -> str:
        if wrong_direction:
            account, hint = wrong_direction[0]
            opposite = "credit" if txn.direction == "debit" else "debit"
            return (
                f"narration keyword '{hint}' points to {account.code} "
                f"{account.name}, but that is {_article(account.family)} "
                f"{account.family} account and "
                f"this is a {txn.direction}, not a {opposite}. No account of the "
                f"right type matched a keyword, so the category is unclear."
            )
        if payee:
            return (
                f"counterparty '{payee}' has not been seen before and the "
                "narration contains no category keyword"
            )
        return (
            "narration carries no counterparty name and no category keyword, "
            "so there is nothing to classify on"
        )
