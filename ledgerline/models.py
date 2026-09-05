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


class LedgerEntry(BaseModel):
    txn_id: str
    account_code: Optional[str] = None
    amount: Decimal
    confidence: float
    method: Method
    reason: str
    status: Status


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
    llm_calls: int = 0
    # Payee memory grows as humans correct rows, so runs stop being identical
    # once it is warm. Recording its size makes that visible in the report.
    payee_memory_entries: int = 0
    payee_memory_warm: bool = False
    notes: list[str] = Field(default_factory=list)
