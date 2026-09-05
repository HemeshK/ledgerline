"""Tier 2: character n-gram TF-IDF into logistic regression.

The training labels are deliberately NOT the ground-truth key. They are the
accounts tier 1 resolved on its own, restricted to train_vendors rows - the
ledger history the system legitimately produced. Training on truth.json would
hand the classifier answers the pipeline never earned and make the held-out
evaluation circular.
"""

from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .models import (
    ClassCount,
    ClassifierTraining,
    ClassificationResult,
    LedgerEntry,
    Transaction,
)
from .rules import normalize

LABEL_SOURCE = (
    "accounts tier 1 posted for train_vendors rows (the ledger the rules "
    "produced), not the ground-truth key"
)


def build_training_set(
    transactions: list[Transaction],
    tier1_entries: list[LedgerEntry],
    train_vendors: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Returns (texts, labels, vendors) drawn only from rows tier 1 posted
    whose counterparty is a train vendor."""
    train_set = {normalize(v) for v in train_vendors}
    by_id = {t.txn_id: t for t in transactions}

    texts: list[str] = []
    labels: list[str] = []
    vendors: list[str] = []

    for entry in tier1_entries:
        if entry.status != "posted" or entry.account_code is None:
            continue
        txn = by_id.get(entry.txn_id)
        if txn is None:
            continue
        vendor = normalize(txn.counterparty_raw)
        if vendor not in train_set:
            continue
        texts.append(txn.narration)
        labels.append(entry.account_code)
        vendors.append(vendor)

    return texts, labels, vendors


def describe_training(
    labels: list[str], vendors: list[str]
) -> ClassifierTraining:
    counts = Counter(labels)
    per_class = [
        ClassCount(account_code=code, examples=n)
        for code, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return ClassifierTraining(
        label_source=LABEL_SOURCE,
        rows=len(labels),
        classes=len(counts),
        vendors=len(set(vendors)),
        per_class=per_class,
        min_examples=min(counts.values()) if counts else 0,
        single_example_classes=sum(1 for n in counts.values() if n == 1),
    )


def train(texts: list[str], labels: list[str]):
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
        LogisticRegression(max_iter=1000),
    )
    model.fit(texts, labels)
    return model


def save_model(model, path: Path) -> None:
    joblib.dump(model, path)


class ClassifierTier:
    name = "classifier"

    def __init__(self, model, training_rows: int):
        self.model = model
        self.training_rows = training_rows
        self.classes = list(model.classes_)
        # Every prediction, including ones the gate rejects. The model's own
        # quality is measured over everything it scored; what it contributed to
        # the ledger is measured over what got past the gate. Those are
        # different questions and conflating them hides a weak classifier.
        self.predictions: dict[str, tuple[str, float]] = {}

    def classify(self, txn: Transaction) -> ClassificationResult:
        proba = self.model.predict_proba([txn.narration])[0]
        ranked = sorted(zip(self.classes, proba), key=lambda cp: cp[1], reverse=True)
        code, confidence = ranked[0]
        self.predictions[txn.txn_id] = (code, float(confidence))

        reason = (
            f"character n-gram model trained on {self.training_rows} "
            f"rules-resolved rows puts this closest to {code} "
            f"at {confidence:.0%} probability"
        )
        if len(ranked) > 1:
            runner_code, runner_p = ranked[1]
            reason += f" (runner-up {runner_code} at {runner_p:.0%})"

        return ClassificationResult(
            account_code=code,
            confidence=float(confidence),
            method="classifier",
            reason=reason,
        )
