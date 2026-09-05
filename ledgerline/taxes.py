"""GST splitting for tax-inclusive amounts.

The split cannot drift. The base is rounded to paise, and the tax component is
then taken as the remainder rather than being rounded independently:

    base = round(total / 1.18)
    tax  = total - base

Rounding both halves separately is what produces the classic one-paisa gap on
amounts where 18% does not divide cleanly. Deriving the second half by
subtraction makes base + tax == total true by construction, for every input.
"""

from decimal import Decimal, ROUND_HALF_UP

from .models import LedgerEntry, Transaction

GST_RATE = Decimal("0.18")
GST_MARKER = "GST INCL"
OUTPUT_TAX_ACCOUNT = "2300"
INPUT_CREDIT_ACCOUNT = "1450"
PAISE = Decimal("0.01")


def is_gst_inclusive(txn: Transaction) -> bool:
    return GST_MARKER in txn.narration.upper()


def split_inclusive(total: Decimal) -> tuple[Decimal, Decimal]:
    """Returns (base, tax) which always sum exactly back to total."""
    base = (total / (Decimal("1") + GST_RATE)).quantize(PAISE, rounding=ROUND_HALF_UP)
    return base, total - base


def tax_account_for(direction: str) -> str:
    """Output tax on money coming in is a liability; tax paid on money going
    out is an input credit, which is an asset."""
    return OUTPUT_TAX_ACCOUNT if direction == "credit" else INPUT_CREDIT_ACCOUNT


def apply_tax_splits(
    transactions: list[Transaction], entries: list[LedgerEntry]
) -> list[LedgerEntry]:
    """Expands posted GST-inclusive rows into a base line plus a tax line.
    Flagged rows are left alone: they are not in the ledger yet."""
    by_id = {t.txn_id: t for t in transactions}
    result: list[LedgerEntry] = []

    for entry in entries:
        txn = by_id.get(entry.txn_id)
        if (
            entry.status != "posted"
            or entry.account_code is None
            or txn is None
            or not is_gst_inclusive(txn)
        ):
            result.append(entry)
            continue

        base, tax = split_inclusive(entry.amount)
        tax_code = tax_account_for(txn.direction)

        result.append(entry.model_copy(update={"amount": base}))
        result.append(
            entry.model_copy(
                update={
                    "account_code": tax_code,
                    "amount": tax,
                    "line_type": "tax",
                    "reason": (
                        f"18% GST split out of the {entry.amount} inclusive "
                        f"amount on {entry.account_code}"
                    ),
                }
            )
        )

    return result
