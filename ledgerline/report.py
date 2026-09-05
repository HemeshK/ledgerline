"""Builds and renders the run report.

Scoring treats undecidable rows (ground truth null) separately from the rest:
flagging one is a correct refusal and counts as a win, posting one is an error.
Coverage is measured over decidable rows only, so the metric does not punish
the abstention behaviour the system is built around.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .models import ErrorRow, LedgerEntry, MethodStats, RunReport, Transaction


def build_report(
    transactions: list[Transaction],
    entries: list[LedgerEntry],
    truth: dict[str, Optional[str]],
    notes: Optional[list[str]] = None,
) -> RunReport:
    narrations = {t.txn_id: t.narration for t in transactions}

    total = len(transactions)
    undecidable = sum(1 for t in transactions if truth.get(t.txn_id) is None)
    decidable = total - undecidable

    attempted = 0
    correct = 0
    flagged = 0
    correct_refusals = 0
    wrongly_posted_undecidable = 0
    errors: list[ErrorRow] = []
    per_method: OrderedDict[str, dict] = OrderedDict()

    def bucket(method: str) -> dict:
        if method not in per_method:
            per_method[method] = {
                "attempted": 0,
                "correct": 0,
                "flagged": 0,
                "misposted": 0,
            }
        return per_method[method]

    for e in entries:
        actual = truth.get(e.txn_id)
        is_undecidable = actual is None
        stats = bucket(e.method)

        if e.status == "posted":
            if is_undecidable:
                # No correct account exists for this row; posting it at all is wrong.
                stats["misposted"] += 1
                wrongly_posted_undecidable += 1
                errors.append(
                    ErrorRow(
                        txn_id=e.txn_id,
                        narration=narrations.get(e.txn_id, ""),
                        predicted=e.account_code,
                        actual=None,
                        confidence=e.confidence,
                        method=e.method,
                        undecidable=True,
                    )
                )
            else:
                attempted += 1
                stats["attempted"] += 1
                if e.account_code == actual:
                    correct += 1
                    stats["correct"] += 1
                else:
                    errors.append(
                        ErrorRow(
                            txn_id=e.txn_id,
                            narration=narrations.get(e.txn_id, ""),
                            predicted=e.account_code,
                            actual=actual,
                            confidence=e.confidence,
                            method=e.method,
                        )
                    )
        else:
            flagged += 1
            stats["flagged"] += 1
            if is_undecidable:
                correct_refusals += 1

    method_stats = [
        MethodStats(
            method=m,
            attempted=s["attempted"],
            correct=s["correct"],
            flagged=s["flagged"],
            misposted=s["misposted"],
            accuracy=(s["correct"] / s["attempted"]) if s["attempted"] else None,
        )
        for m, s in per_method.items()
    ]

    return RunReport(
        total=total,
        decidable=decidable,
        undecidable=undecidable,
        attempted=attempted,
        flagged=flagged,
        correct=correct,
        accuracy_on_attempted=(correct / attempted) if attempted else None,
        coverage=(attempted / decidable) if decidable else None,
        correct_refusals=correct_refusals,
        wrongly_posted_undecidable=wrongly_posted_undecidable,
        per_method=method_stats,
        errors=errors,
        notes=notes or [],
    )


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def render(report: RunReport, console: Optional[Console] = None) -> None:
    console = console or Console()

    summary = Table(title="Run report", title_justify="left", show_edge=False)
    summary.add_column("metric")
    summary.add_column("value", justify="right")

    summary.add_row("total rows", str(report.total))
    summary.add_row("decidable", str(report.decidable))
    summary.add_row("undecidable", str(report.undecidable))
    summary.add_row("attempted", str(report.attempted))
    summary.add_row("flagged", str(report.flagged))
    accuracy = _pct(report.accuracy_on_attempted)
    if report.attempted == 0:
        accuracy += " (0 attempted)"
    summary.add_row("accuracy on attempted", accuracy)
    summary.add_row("coverage (of decidable)", _pct(report.coverage))
    summary.add_row(
        "correct refusals", f"{report.correct_refusals} of {report.undecidable}"
    )
    if report.wrongly_posted_undecidable:
        summary.add_row(
            "undecidable rows wrongly posted", str(report.wrongly_posted_undecidable)
        )
    summary.add_row("llm calls", str(report.llm_calls))

    console.print(summary)
    console.print()

    methods = Table(title="By method", title_justify="left", show_edge=False)
    methods.add_column("method")
    methods.add_column("attempted", justify="right")
    methods.add_column("correct", justify="right")
    methods.add_column("accuracy", justify="right")
    methods.add_column("flagged", justify="right")
    methods.add_column("misposted", justify="right")
    for m in report.per_method:
        methods.add_row(
            m.method,
            str(m.attempted),
            str(m.correct),
            _pct(m.accuracy),
            str(m.flagged),
            str(m.misposted),
        )
    console.print(methods)
    console.print()

    if report.errors:
        errs = Table(
            title=f"Wrong rows ({len(report.errors)})",
            title_justify="left",
            show_edge=False,
        )
        errs.add_column("txn")
        errs.add_column("narration", max_width=52, overflow="ellipsis")
        errs.add_column("predicted", justify="right")
        errs.add_column("actual", justify="right")
        errs.add_column("conf", justify="right")
        errs.add_column("via")
        for e in report.errors:
            errs.add_row(
                e.txn_id,
                e.narration,
                e.predicted or "-",
                "undecidable" if e.undecidable else (e.actual or "-"),
                f"{e.confidence:.2f}",
                e.method,
            )
        console.print(errs)
        console.print()

    for note in report.notes:
        console.print(f"note: {note}")


def write_json(report: RunReport, path: Path) -> None:
    path.write_text(report.model_dump_json(indent=2))
