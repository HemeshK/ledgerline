"""The tier cascade and the confidence gate.

Every tier implements the same interface: classify(txn) -> ClassificationResult.
Tiers run cheapest-first and each only sees what the previous one could not
resolve above the confidence threshold.
"""

from typing import Protocol

from .models import ClassificationResult, LedgerEntry, Transaction


class Tier(Protocol):
    name: str

    def classify(self, txn: Transaction) -> ClassificationResult: ...


class StubTier:
    """Placeholder standing in for the real tiers. Returns a fixed account at a
    confidence below any sane threshold, so every row falls through to review.
    Replaced by the rule, classifier and LLM tiers in phases 3-5."""

    name = "classifier"

    def classify(self, txn: Transaction) -> ClassificationResult:
        return ClassificationResult(
            account_code="6800",
            confidence=0.1,
            method="classifier",
            reason=(
                "stub classifier: no classification logic implemented yet, "
                "so every row is sent to review"
            ),
        )


def run_pipeline(
    transactions: list[Transaction], tiers: list[Tier], threshold: float
) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []

    for txn in transactions:
        best: ClassificationResult | None = None

        for tier in tiers:
            result = tier.classify(txn)
            if best is None or result.confidence > best.confidence:
                best = result
            if result.confidence >= threshold:
                break

        posted = best is not None and best.confidence >= threshold
        entries.append(
            LedgerEntry(
                txn_id=txn.txn_id,
                account_code=best.account_code if best else None,
                amount=txn.amount,
                confidence=best.confidence if best else 0.0,
                method=best.method if best else "rule",
                reason=best.reason if best else "no tier produced a result",
                status="posted" if posted else "flagged",
            )
        )

    return entries
