"""P&L and balance sheet as group-bys over the posted ledger."""

from collections import OrderedDict
from decimal import Decimal

from .models import Account, LedgerEntry, StatementLine, Statements, Transaction

ZERO = Decimal("0.00")
PNL_FAMILIES = ("revenue", "expense")
BALANCE_FAMILIES = ("asset", "liability", "equity")


def build_statements(
    transactions: list[Transaction],
    entries: list[LedgerEntry],
    accounts: list[Account],
) -> Statements:
    by_code = {a.code: a for a in accounts}

    totals: OrderedDict[str, dict] = OrderedDict()
    for entry in entries:
        if entry.status != "posted" or entry.account_code is None:
            continue
        account = by_code.get(entry.account_code)
        if account is None:
            continue
        bucket = totals.setdefault(
            entry.account_code, {"account": account, "amount": ZERO, "lines": 0}
        )
        bucket["amount"] += entry.amount
        bucket["lines"] += 1

    def lines_for(families) -> list[StatementLine]:
        rows = [
            StatementLine(
                account_code=code,
                account_name=b["account"].name,
                family=b["account"].family,
                amount=b["amount"],
                lines=b["lines"],
            )
            for code, b in totals.items()
            if b["account"].family in families
        ]
        return sorted(rows, key=lambda r: (r.family, r.account_code))

    def total_for(rows, family) -> Decimal:
        return sum((r.amount for r in rows if r.family == family), ZERO)

    pnl = lines_for(PNL_FAMILIES)
    balance = lines_for(BALANCE_FAMILIES)

    revenue_total = total_for(pnl, "revenue")
    expense_total = total_for(pnl, "expense")

    dates = [t.date for t in transactions]

    return Statements(
        period_start=min(dates) if dates else None,
        period_end=max(dates) if dates else None,
        pnl=pnl,
        revenue_total=revenue_total,
        expense_total=expense_total,
        net=revenue_total - expense_total,
        balance_sheet=balance,
        asset_total=total_for(balance, "asset"),
        liability_total=total_for(balance, "liability"),
        equity_total=total_for(balance, "equity"),
    )
