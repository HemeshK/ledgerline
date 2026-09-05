"""Builds and renders the run report.

Scoring treats undecidable rows (ground truth null) separately from the rest:
flagging one is a correct refusal and counts as a win, posting one is an error.
Coverage is measured over decidable rows only, so the metric does not punish
the abstention behaviour the system is built around.
"""

from collections import Counter, OrderedDict
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .models import (
    CapexProbeRow,
    ClassifierTraining,
    ConfusionPair,
    ErrorRow,
    FlaggedRow,
    LedgerEntry,
    MethodStats,
    RunReport,
    Statements,
    Transaction,
    VendorSplitStats,
)
from .rules import normalize

FLAGGED_PREVIEW = 8
CONFUSION_PREVIEW = 10
# Below this gap, seen and held-out vendors are not meaningfully different and
# the vendor split is not testing generalisation.
MEANINGFUL_GAP = 0.05
CAPEX_ACCOUNT = "1500"


def build_report(
    transactions: list[Transaction],
    entries: list[LedgerEntry],
    truth: dict[str, Optional[str]],
    notes: Optional[list[str]] = None,
    payee_memory_entries: int = 0,
    train_vendors: Optional[list[str]] = None,
    heldout_vendors: Optional[list[str]] = None,
    classifier_training: Optional[ClassifierTraining] = None,
    classifier_predictions: Optional[dict[str, tuple[str, float]]] = None,
    llm_predictions: Optional[dict[str, tuple[Optional[str], float]]] = None,
    llm_calls: int = 0,
    llm_cache_hits: int = 0,
    llm_skipped_reason: Optional[str] = None,
    statements: Optional[Statements] = None,
) -> RunReport:
    narrations = {t.txn_id: t.narration for t in transactions}
    counterparties = {t.txn_id: normalize(t.counterparty_raw) for t in transactions}
    seen_vendors = {normalize(v) for v in (train_vendors or [])}
    heldout = {normalize(v) for v in (heldout_vendors or [])}

    total = len(transactions)
    undecidable = sum(1 for t in transactions if truth.get(t.txn_id) is None)
    decidable = total - undecidable

    attempted = 0
    correct = 0
    flagged = 0
    correct_refusals = 0
    wrongly_posted_undecidable = 0
    errors: list[ErrorRow] = []
    flagged_rows: list[FlaggedRow] = []
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
        # Derived tax lines carry no classification decision of their own, so
        # scoring them would double-count the transaction and mark the tax
        # account as a wrong answer against the primary line's ground truth.
        if e.line_type != "primary":
            continue
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
                hit = e.account_code == actual
                if hit:
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
            flagged_rows.append(
                FlaggedRow(
                    txn_id=e.txn_id,
                    narration=narrations.get(e.txn_id, ""),
                    best_guess=e.account_code,
                    confidence=e.confidence,
                    method=e.method,
                    reason=e.reason,
                )
            )
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

    def bucket_for(txn_id: str) -> str:
        vendor = counterparties.get(txn_id, "")
        if vendor in seen_vendors:
            return "seen vendors"
        if vendor in heldout:
            return "held-out vendors"
        return "neither list"

    vendor_split, confusion_pairs = _split_and_confusions(
        classifier_predictions, truth, bucket_for
    )
    llm_split, llm_confusion_pairs = _split_and_confusions(
        llm_predictions, truth, bucket_for
    )

    # How tier 3 handled the rows that have no correct answer.
    undecidable_seen = 0
    undecidable_refused = 0
    for txn_id, (predicted, _conf) in (llm_predictions or {}).items():
        if truth.get(txn_id) is None:
            undecidable_seen += 1
            if predicted is None:
                undecidable_refused += 1

    capex_probe = _capex_probe(transactions, entries, truth, narrations)

    notes = list(notes or [])
    notes.append(
        "tier accuracies above are measured on NON-COMPARABLE subsets: tier 1 "
        "sees every row, while tiers 2 and 3 see only what the tiers above "
        "them declined - which is the harder residue. A head-to-head must not "
        "present these as like-for-like; to compare tiers fairly, run each "
        "over the same rows."
    )
    notes.extend(_vendor_split_findings(vendor_split, "classifier"))
    if llm_predictions:
        notes.extend(_vendor_split_findings(llm_split, "LLM"))
        notes.extend(
            _refusal_findings(undecidable_seen, undecidable_refused)
        )

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
        flagged_rows=flagged_rows,
        classifier_training=classifier_training,
        statements=statements,
        gst_split_lines=sum(1 for e in entries if e.line_type == "tax"),
        classifier_scored=len(classifier_predictions or {}),
        classifier_posted=next(
            (m.attempted for m in method_stats if m.method == "classifier"), 0
        ),
        classifier_vendor_split=vendor_split,
        classifier_confusions=confusion_pairs,
        llm_scored=len(llm_predictions or {}),
        llm_posted=next(
            (m.attempted for m in method_stats if m.method == "llm"), 0
        ),
        llm_calls=llm_calls,
        llm_cache_hits=llm_cache_hits,
        llm_call_rate=(
            llm_calls / len(llm_predictions) if llm_predictions else None
        ),
        llm_skipped_reason=llm_skipped_reason,
        llm_vendor_split=llm_split,
        llm_confusions=llm_confusion_pairs,
        llm_undecidable_seen=undecidable_seen,
        llm_undecidable_refused=undecidable_refused,
        capex_probe=capex_probe,
        payee_memory_entries=payee_memory_entries,
        payee_memory_warm=payee_memory_entries > 0,
        notes=notes,
    )


def _split_and_confusions(
    predictions: Optional[dict],
    truth: dict[str, Optional[str]],
    bucket_for,
) -> tuple[list[VendorSplitStats], list[ConfusionPair]]:
    """Accuracy by vendor familiarity over every row a tier scored, plus its
    top confusions. Refusals are not accuracy events on decidable rows, so they
    are excluded from both rather than counted as errors."""
    split: OrderedDict[str, dict] = OrderedDict(
        (b, {"scored": 0, "correct": 0})
        for b in ("seen vendors", "held-out vendors", "neither list")
    )
    confusions: Counter = Counter()

    for txn_id, (predicted, _conf) in (predictions or {}).items():
        actual = truth.get(txn_id)
        if actual is None or predicted is None:
            continue
        bucket = split[bucket_for(txn_id)]
        bucket["scored"] += 1
        if predicted == actual:
            bucket["correct"] += 1
        else:
            confusions[(actual, predicted)] += 1

    stats = [
        VendorSplitStats(
            bucket=name,
            scored=s["scored"],
            correct=s["correct"],
            accuracy=(s["correct"] / s["scored"]) if s["scored"] else None,
        )
        for name, s in split.items()
    ]
    pairs = [
        ConfusionPair(actual=a, predicted=p, count=n)
        for (a, p), n in confusions.most_common(CONFUSION_PREVIEW)
    ]
    return stats, pairs


def _capex_probe(
    transactions: list[Transaction],
    entries: list[LedgerEntry],
    truth: dict[str, Optional[str]],
    narrations: dict[str, str],
) -> list[CapexProbeRow]:
    posted = {
        e.txn_id: e
        for e in entries
        if e.status == "posted" and e.line_type == "primary"
    }
    rows = []
    for txn in transactions:
        if truth.get(txn.txn_id) != CAPEX_ACCOUNT:
            continue
        entry = posted.get(txn.txn_id)
        rows.append(
            CapexProbeRow(
                txn_id=txn.txn_id,
                narration=narrations.get(txn.txn_id, ""),
                decided_by=entry.method if entry else None,
                predicted=entry.account_code if entry else None,
                correct=bool(entry and entry.account_code == CAPEX_ACCOUNT),
            )
        )
    return rows


def _refusal_findings(seen: int, refused: int) -> list[str]:
    if seen == 0:
        return []
    assigned = seen - refused
    if assigned == 0:
        return [
            f"the LLM refused all {seen} no-signal rows it saw, which is the "
            "correct answer for rows with no usable signal"
        ]
    return [
        f"FINDING: the LLM assigned an account to {assigned} of {seen} "
        "no-signal rows instead of refusing. Those rows carry no usable "
        "signal, so a confident answer there is a fabrication, not a win."
    ]


def _vendor_split_findings(
    split: list[VendorSplitStats], tier: str = "classifier"
) -> list[str]:
    """States plainly whether held-out vendors were actually harder. If they
    were not, the two vendor sets are too similar and the split is not
    measuring generalisation - that is a finding, not something to smooth over."""
    stats = {s.bucket: s for s in split}
    seen = stats.get("seen vendors")
    held = stats.get("held-out vendors")

    if not seen or not held or seen.scored == 0 or held.scored == 0:
        return [
            f"{tier} vendor split not measurable: it scored "
            f"{seen.scored if seen else 0} seen-vendor and "
            f"{held.scored if held else 0} held-out rows"
        ]

    gap = seen.accuracy - held.accuracy
    if gap >= MEANINGFUL_GAP:
        return [
            f"{tier} held-out accuracy ({held.accuracy:.1%}) is "
            f"{gap:.1%} below its seen-vendor accuracy ({seen.accuracy:.1%}), "
            "which is the expected direction: unseen payee names are harder"
        ]
    return [
        f"FINDING: {tier} held-out accuracy ({held.accuracy:.1%}) is NOT "
        f"clearly below seen-vendor accuracy ({seen.accuracy:.1%}); gap is "
        f"{gap:.1%}. For the classifier that would mean the vendor lists are "
        "too similar; for the LLM it may simply mean payee familiarity is not "
        "what it relies on."
    ]


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
    if report.llm_skipped_reason:
        summary.add_row("llm", f"skipped - {report.llm_skipped_reason}")
    else:
        summary.add_row(
            "llm calls",
            f"{report.llm_calls} live, {report.llm_cache_hits} cached"
            + (
                f" ({report.llm_call_rate:.1%} call rate)"
                if report.llm_call_rate is not None
                else ""
            ),
        )
    memory_state = "warm" if report.payee_memory_warm else "empty"
    summary.add_row(
        "payee memory at start",
        f"{memory_state} ({report.payee_memory_entries} entries)",
    )

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

    training = report.classifier_training
    if training:
        console.print("[bold]Classifier training set[/bold]")
        console.print(f"  labels from: {training.label_source}")
        console.print(
            f"  {training.rows} rows, {training.classes} classes, "
            f"{training.vendors} vendors; smallest class has "
            f"{training.min_examples} example(s), "
            f"{training.single_example_classes} class(es) have exactly one"
        )
        per_class = Table(show_edge=False, box=None, pad_edge=False)
        per_class.add_column("  account", justify="left")
        per_class.add_column("examples", justify="right")
        for c in training.per_class:
            per_class.add_row(f"  {c.account_code}", str(c.examples))
        console.print(per_class)
        console.print()

    if report.classifier_vendor_split:
        vs = Table(
            title=(
                "Classifier accuracy by vendor familiarity "
                f"(over all {report.classifier_scored} rows it scored; "
                f"{report.classifier_posted} cleared the gate and posted)"
            ),
            title_justify="left",
            show_edge=False,
        )
        vs.add_column("bucket")
        vs.add_column("scored", justify="right")
        vs.add_column("correct", justify="right")
        vs.add_column("accuracy", justify="right")
        for s in report.classifier_vendor_split:
            vs.add_row(s.bucket, str(s.scored), str(s.correct), _pct(s.accuracy))
        console.print(vs)
        console.print()

    if report.classifier_confusions:
        cm = Table(
            title="Top classifier confusions (actual -> predicted)",
            title_justify="left",
            show_edge=False,
        )
        cm.add_column("actual")
        cm.add_column("predicted")
        cm.add_column("count", justify="right")
        for c in report.classifier_confusions:
            cm.add_row(c.actual, c.predicted, str(c.count))
        console.print(cm)
        console.print()

    if report.llm_vendor_split and report.llm_scored:
        ls = Table(
            title=(
                f"LLM accuracy by vendor familiarity (scored "
                f"{report.llm_scored} rows, {report.llm_posted} posted; "
                f"refused {report.llm_undecidable_refused} of "
                f"{report.llm_undecidable_seen} no-signal rows)"
            ),
            title_justify="left",
            show_edge=False,
        )
        ls.add_column("bucket")
        ls.add_column("scored", justify="right")
        ls.add_column("correct", justify="right")
        ls.add_column("accuracy", justify="right")
        for s in report.llm_vendor_split:
            ls.add_row(s.bucket, str(s.scored), str(s.correct), _pct(s.accuracy))
        console.print(ls)
        console.print()

    if report.llm_confusions:
        lcm = Table(
            title="Top LLM confusions (actual -> predicted)",
            title_justify="left",
            show_edge=False,
        )
        lcm.add_column("actual")
        lcm.add_column("predicted")
        lcm.add_column("count", justify="right")
        for c in report.llm_confusions:
            lcm.add_row(c.actual, c.predicted, str(c.count))
        console.print(lcm)
        console.print()

    if report.capex_probe:
        right = sum(1 for r in report.capex_probe if r.correct)
        cap = Table(
            title=(
                f"Capex vs opex probe - rows that belong in {CAPEX_ACCOUNT} "
                f"Equipment ({right} of {len(report.capex_probe)} correct)"
            ),
            title_justify="left",
            show_edge=False,
        )
        cap.add_column("txn")
        cap.add_column("narration", max_width=46, overflow="fold")
        cap.add_column("decided by")
        cap.add_column("posted", justify="right")
        cap.add_column("ok", justify="center")
        for r in report.capex_probe:
            cap.add_row(
                r.txn_id,
                r.narration,
                r.decided_by or "unresolved",
                r.predicted or "-",
                "yes" if r.correct else "no",
            )
        console.print(cap)
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

    if report.statements:
        _render_statements(report.statements, report.gst_split_lines, console)

    if report.flagged_rows:
        shown = report.flagged_rows[:FLAGGED_PREVIEW]
        queue = Table(
            title=(
                f"Exception queue ({len(report.flagged_rows)} rows, "
                f"showing {len(shown)})"
            ),
            title_justify="left",
            show_edge=False,
        )
        queue.add_column("txn")
        queue.add_column("narration", max_width=40, overflow="fold")
        queue.add_column("why it was not posted", max_width=64, overflow="fold")
        for f in shown:
            queue.add_row(f.txn_id, f.narration, f.reason)
        console.print(queue)
        console.print()

    for note in report.notes:
        console.print(f"note: {note}")


def _money(value) -> str:
    return f"{value:,.2f}"


def _render_statements(
    statements: Statements, gst_split_lines: int, console: Console
) -> None:
    period = ""
    if statements.period_start and statements.period_end:
        period = f" {statements.period_start} to {statements.period_end}"

    pnl = Table(
        title=f"Profit & loss{period}", title_justify="left", show_edge=False
    )
    pnl.add_column("account")
    pnl.add_column("name", max_width=32)
    pnl.add_column("lines", justify="right")
    pnl.add_column("amount", justify="right")
    for family in ("revenue", "expense"):
        for row in statements.pnl:
            if row.family != family:
                continue
            pnl.add_row(
                row.account_code, row.account_name, str(row.lines), _money(row.amount)
            )
        total = (
            statements.revenue_total
            if family == "revenue"
            else statements.expense_total
        )
        pnl.add_row("", f"total {family}", "", _money(total), style="bold")
    pnl.add_row("", "net", "", _money(statements.net), style="bold")
    console.print(pnl)
    console.print()

    bs = Table(title="Balance sheet", title_justify="left", show_edge=False)
    bs.add_column("account")
    bs.add_column("name", max_width=32)
    bs.add_column("lines", justify="right")
    bs.add_column("amount", justify="right")
    for family in ("asset", "liability", "equity"):
        for row in statements.balance_sheet:
            if row.family != family:
                continue
            bs.add_row(
                row.account_code, row.account_name, str(row.lines), _money(row.amount)
            )
        total = {
            "asset": statements.asset_total,
            "liability": statements.liability_total,
            "equity": statements.equity_total,
        }[family]
        bs.add_row("", f"total {family}", "", _money(total), style="bold")
    console.print(bs)
    console.print()
    console.print(
        f"{gst_split_lines} GST line(s) split out of tax-inclusive amounts. "
        "These statements summarise the classified side of each transaction "
        "only; the contra bank line is not recorded, so this does not balance."
    )
    console.print()


def write_json(report: RunReport, path: Path) -> None:
    path.write_text(report.model_dump_json(indent=2))
