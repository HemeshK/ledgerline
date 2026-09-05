import typer
from rich.console import Console

from . import config, data_loader, report as report_mod, storage
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

    tiers = [RuleTier(accounts, conn)]
    entries = run_pipeline(batch.transactions, tiers, config.CONFIDENCE_THRESHOLD)
    storage.save_entries(conn, entries)

    result = report_mod.build_report(
        batch.transactions,
        entries,
        truth,
        notes=[
            "only tier 1 (rules) is built; tiers 2-3 land in later phases, so "
            "everything rules cannot resolve stays in the exception queue",
        ],
        payee_memory_entries=payee_entries,
    )
    report_mod.render(result, console)
    report_mod.write_json(result, config.REPORT_PATH)
    console.print(f"wrote {config.REPORT_PATH.name}")

    conn.close()


if __name__ == "__main__":
    app()
