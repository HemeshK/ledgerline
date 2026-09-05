"""Reads the chart of accounts, the batch, and the ground-truth key."""

import json
from pathlib import Path
from typing import Optional

from .models import Account, Batch


def load_accounts(path: Path) -> list[Account]:
    return [Account(**a) for a in json.loads(path.read_text())]


def load_batch(path: Path) -> Batch:
    """batch.json nests the rows under "transactions", alongside the two
    vendor lists that make the later classifier evaluation vendor-disjoint."""
    return Batch(**json.loads(path.read_text()))


def load_truth(path: Path) -> dict[str, Optional[str]]:
    """txn_id -> account_code. A null value means the row is undecidable:
    there is no correct account, and the correct behaviour is to refuse."""
    return json.loads(path.read_text())
