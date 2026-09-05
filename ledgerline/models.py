"""Core domain models. All money is Decimal, never float."""

import datetime as dt
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field

Direction = Literal["credit", "debit"]
Method = Literal["rule", "classifier", "llm"]
Status = Literal["posted", "flagged", "corrected"]
Family = Literal["revenue", "expense", "asset", "liability", "equity"]
Statement = Literal["pnl", "balance_sheet"]


class Transaction(BaseModel):
    txn_id: str
    date: dt.date
    amount: Decimal
    direction: Direction
    narration: str
    counterparty_raw: str = ""


class Account(BaseModel):
    code: str
    name: str
    family: Family
    statement: Statement
    hints: list[str] = Field(default_factory=list)


class Batch(BaseModel):
    train_vendors: list[str] = Field(default_factory=list)
    heldout_vendors: list[str] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)


class ClassificationResult(BaseModel):
    """What a single tier decided. account_code is None when the tier abstains."""

    account_code: Optional[str] = None
    confidence: float = 0.0
    method: Method
    reason: str


LineType = Literal["primary", "tax"]


class LedgerEntry(BaseModel):
    """A single ledger line. One transaction yields one primary line, plus a
    derived tax line when the amount is GST-inclusive. Only the primary line
    carries a classification decision, so only it is scored."""

    txn_id: str
    account_code: Optional[str] = None
    amount: Decimal
    confidence: float
    method: Method
    reason: str
    status: Status
    line_type: LineType = "primary"


class MethodStats(BaseModel):
    """Per-tier counts, using the same definitions as the headline metrics:
    `attempted` is decidable rows posted, so accuracy is comparable to
    accuracy_on_attempted. Undecidable rows posted are counted as `misposted`."""

    method: Method
    attempted: int
    correct: int
    flagged: int
    misposted: int = 0
    accuracy: Optional[float] = None


class ErrorRow(BaseModel):
    txn_id: str
    narration: str
    predicted: Optional[str] = None
    actual: Optional[str] = None
    confidence: float
    method: Method
    undecidable: bool = False


class ClassCount(BaseModel):
    account_code: str
    examples: int


class ClassifierTraining(BaseModel):
    """What tier 2 actually learned from. Made explicit because with this many
    accounts and a small batch, several classes carry only a handful of
    examples and an accuracy number alone would hide that."""

    label_source: str
    rows: int
    classes: int
    vendors: int
    per_class: list[ClassCount] = Field(default_factory=list)
    min_examples: int = 0
    single_example_classes: int = 0


class VendorSplitStats(BaseModel):
    """Accuracy split by whether the counterparty was in the training vendor
    list. Held-out vendors are names the classifier has genuinely never seen.

    Measured over every row the classifier scored, not only rows that cleared
    the confidence gate, so the model's own quality stays visible even when the
    gate correctly refuses to post any of its predictions."""

    bucket: str
    scored: int
    correct: int
    accuracy: Optional[float] = None


class ConfusionPair(BaseModel):
    actual: str
    predicted: str
    count: int


class StatementLine(BaseModel):
    account_code: str
    account_name: str
    family: Family
    amount: Decimal
    lines: int


class Statements(BaseModel):
    """P&L as a group-by over revenue and expense families for the period, and
    balance-sheet families summed cumulatively.

    This is a categorisation summary, not a balanced double-entry statement:
    only the category side of each transaction is recorded, never the contra
    bank line, so assets do not equal liabilities plus equity."""

    period_start: Optional[dt.date] = None
    period_end: Optional[dt.date] = None
    pnl: list[StatementLine] = Field(default_factory=list)
    revenue_total: Decimal = Decimal("0.00")
    expense_total: Decimal = Decimal("0.00")
    net: Decimal = Decimal("0.00")
    balance_sheet: list[StatementLine] = Field(default_factory=list)
    asset_total: Decimal = Decimal("0.00")
    liability_total: Decimal = Decimal("0.00")
    equity_total: Decimal = Decimal("0.00")


class CapexProbeRow(BaseModel):
    """Capex-vs-opex rows (truth 1500 Equipment), and which tier ended up
    deciding each one. A tier only sees rows the tiers above it left open, so
    this also shows where a confident early answer blocks a later tier."""

    txn_id: str
    narration: str
    decided_by: Optional[Method] = None
    predicted: Optional[str] = None
    correct: bool = False


class FlaggedRow(BaseModel):
    """A row sent to the exception queue, with the reason a human acts on."""

    txn_id: str
    narration: str
    best_guess: Optional[str] = None
    confidence: float
    method: Method
    reason: str


class RunReport(BaseModel):
    """Metrics for one pass over a batch.

    Rows whose ground truth is null are undecidable and are scored apart from
    the rest: flagging one is a correct refusal, posting one is an error. They
    are excluded from `decidable`, so coverage never penalises abstention.
    """

    total: int
    decidable: int
    undecidable: int
    attempted: int
    flagged: int
    correct: int
    accuracy_on_attempted: Optional[float] = None
    coverage: Optional[float] = None
    correct_refusals: int = 0
    wrongly_posted_undecidable: int = 0
    per_method: list[MethodStats] = Field(default_factory=list)
    errors: list[ErrorRow] = Field(default_factory=list)
    flagged_rows: list[FlaggedRow] = Field(default_factory=list)
    classifier_training: Optional[ClassifierTraining] = None
    classifier_scored: int = 0
    classifier_posted: int = 0
    classifier_vendor_split: list[VendorSplitStats] = Field(default_factory=list)
    classifier_confusions: list[ConfusionPair] = Field(default_factory=list)
    llm_scored: int = 0
    llm_posted: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_call_rate: Optional[float] = None
    llm_skipped_reason: Optional[str] = None
    llm_vendor_split: list[VendorSplitStats] = Field(default_factory=list)
    llm_confusions: list[ConfusionPair] = Field(default_factory=list)
    llm_undecidable_seen: int = 0
    llm_undecidable_refused: int = 0
    capex_probe: list[CapexProbeRow] = Field(default_factory=list)
    statements: Optional[Statements] = None
    gst_split_lines: int = 0
    # Payee memory grows as humans correct rows, so runs stop being identical
    # once it is warm. Recording its size makes that visible in the report.
    payee_memory_entries: int = 0
    payee_memory_warm: bool = False
    notes: list[str] = Field(default_factory=list)
