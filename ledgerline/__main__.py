import typer
from rich.console import Console

from . import classifier, config, data_loader, report as report_mod, storage
from .pipeline import run_pipeline
from .rules import RuleTier

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """ledgerline - an AI finance controller that refuses to guess."""


@app.command()
def run() -> None:
    """Classify the batch, post what is confident enough, flag the rest."""
    console = Console()

    batch = data_loader.load_batch(config.BATCH_PATH)
    truth = data_loader.load_truth(config.TRUTH_PATH)
    accounts = data_loader.load_accounts(config.ACCOUNTS_PATH)

    conn = storage.connect(config.DB_PATH)
    storage.init_db(conn)

    # Read before the run: corrections seed this, so it changes between runs.
    payee_entries = storage.count_payees(conn)

    rule_tier = RuleTier(accounts, conn)

    # Tier 1 runs alone first so its confident postings can serve as tier 2's
    # training labels. Training on truth.json instead would hand the classifier
    # answers the pipeline never earned.
    tier1_entries = run_pipeline(
        batch.transactions, [rule_tier], config.CONFIDENCE_THRESHOLD
    )
    texts, labels, vendors = classifier.build_training_set(
        batch.transactions, tier1_entries, batch.train_vendors
    )
    model = classifier.train(texts, labels)
    classifier.save_model(model, config.MODEL_PATH)
    training = classifier.describe_training(labels, vendors)

    classifier_tier = classifier.ClassifierTier(model, len(labels))
    tiers = [rule_tier, classifier_tier]
    entries = run_pipeline(batch.transactions, tiers, config.CONFIDENCE_THRESHOLD)
    storage.save_entries(conn, entries)

    result = report_mod.build_report(
        batch.transactions,
        entries,
        truth,
        notes=[
            "tiers 1-2 are built; tier 3 (LLM) lands in phase 5, so what "
            "neither rules nor the classifier can resolve stays in the queue",
        ],
        payee_memory_entries=payee_entries,
        train_vendors=batch.train_vendors,
        heldout_vendors=batch.heldout_vendors,
        classifier_training=training,
        classifier_predictions=classifier_tier.predictions,
    )
    report_mod.render(result, console)
    report_mod.write_json(result, config.REPORT_PATH)
    console.print(f"wrote {config.REPORT_PATH.name}")

    conn.close()


if __name__ == "__main__":
    app()
