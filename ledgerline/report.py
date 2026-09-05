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
    ClassifierTraining,
    ConfusionPair,
    ErrorRow,
    FlaggedRow,
    LedgerEntry,
    MethodStats,
    RunReport,
    Transaction,
    VendorSplitStats,
)
from .rules import normalize

FLAGGED_PREVIEW = 8
CONFUSION_PREVIEW = 10
# Below this gap, seen and held-out vendors are not meaningfully different and
# the vendor split is not testing generalisation.
MEANINGFUL_GAP = 0.05


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
    split: OrderedDict[str, dict] = OrderedDict(
        (b, {"scored": 0, "correct": 0})
        for b in ("seen vendors", "held-out vendors", "neither list")
    )
    confusions: Counter = Counter()

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

    # Scored over every prediction the classifier made, gate or no gate.
    for txn_id, (predicted, _conf) in (classifier_predictions or {}).items():
        actual = truth.get(txn_id)
        if actual is None:
            continue
        vendor = counterparties.get(txn_id, "")
        if vendor in seen_vendors:
            bucket_name = "seen vendors"
        elif vendor in heldout:
            bucket_name = "held-out vendors"
        else:
            bucket_name = "neither list"
        split[bucket_name]["scored"] += 1
        if predicted == actual:
            split[bucket_name]["correct"] += 1
        else:
            confusions[(actual, predicted)] += 1

    vendor_split = [
        VendorSplitStats(
            bucket=name,
            scored=s["scored"],
            correct=s["correct"],
            accuracy=(s["correct"] / s["scored"]) if s["scored"] else None,
        )
        for name, s in split.items()
    ]

    confusion_pairs = [
        ConfusionPair(actual=a, predicted=p, count=n)
        for (a, p), n in confusions.most_common(CONFUSION_PREVIEW)
    ]

    notes = list(notes or [])
    notes.extend(_vendor_split_findings(vendor_split))

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
        classifier_scored=len(classifier_predictions or {}),
        classifier_posted=next(
            (m.attempted for m in method_stats if m.method == "classifier"), 0
        ),
        classifier_vendor_split=vendor_split,
        classifier_confusions=confusion_pairs,
        payee_memory_entries=payee_memory_entries,
        payee_memory_warm=payee_memory_entries > 0,
        notes=notes,
    )


def _vendor_split_findings(split: list[VendorSplitStats]) -> list[str]:
    """States plainly whether held-out vendors were actually harder. If they
    were not, the two vendor sets are too similar and the split is not
    measuring generalisation - that is a finding, not something to smooth over."""
    stats = {s.bucket: s for s in split}
    seen = stats.get("seen vendors")
    held = stats.get("held-out vendors")

    if not seen or not held or seen.scored == 0 or held.scored == 0:
        return [
            "classifier vendor split not measurable: the classifier scored "
            f"{seen.scored if seen else 0} seen-vendor and "
            f"{held.scored if held else 0} held-out rows"
        ]

    gap = seen.accuracy - held.accuracy
    if gap >= MEANINGFUL_GAP:
        return [
            f"held-out vendor accuracy ({held.accuracy:.1%}) is "
            f"{gap:.1%} below seen-vendor accuracy ({seen.accuracy:.1%}), "
            "which is the expected direction: unseen payee names are harder"
        ]
    return [
        f"FINDING: held-out accuracy ({held.accuracy:.1%}) is NOT clearly "
        f"below seen-vendor accuracy ({seen.accuracy:.1%}); gap is {gap:.1%}. "
        "The two vendor lists are probably too similar, so this split is not "
        "testing generalisation as intended."
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
    summary.add_row("llm calls", str(report.llm_calls))
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


def write_json(report: RunReport, path: Path) -> None:
    path.write_text(report.model_dump_json(indent=2))
