import os

import typer
from dotenv import load_dotenv
from rich.console import Console

from . import classifier, config, data_loader, llm, report as report_mod, storage
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
    load_dotenv(config.ROOT / ".env")

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

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    llm_tier = None
    llm_skipped_reason = None
    if api_key:
        llm_tier = llm.LLMTier(accounts, conn, api_key)
        tiers.append(llm_tier)
    else:
        llm_skipped_reason = (
            "GEMINI_API_KEY is not set, so tier 3 did not run and the rows it "
            "would have seen stay in the exception queue"
        )

    entries = run_pipeline(batch.transactions, tiers, config.CONFIDENCE_THRESHOLD)
    storage.save_entries(conn, entries)

    result = report_mod.build_report(
        batch.transactions,
        entries,
        truth,
        payee_memory_entries=payee_entries,
        train_vendors=batch.train_vendors,
        heldout_vendors=batch.heldout_vendors,
        classifier_training=training,
        classifier_predictions=classifier_tier.predictions,
        llm_predictions=llm_tier.predictions if llm_tier else None,
        llm_calls=llm_tier.calls if llm_tier else 0,
        llm_cache_hits=llm_tier.cache_hits if llm_tier else 0,
        llm_skipped_reason=llm_skipped_reason,
    )
    report_mod.render(result, console)
    report_mod.write_json(result, config.REPORT_PATH)
    console.print(f"wrote {config.REPORT_PATH.name}")

    conn.close()


if __name__ == "__main__":
    app()
