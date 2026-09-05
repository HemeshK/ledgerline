import typer
from rich.console import Console

from . import config, data_loader, report as report_mod, storage
from .pipeline import StubTier, run_pipeline

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
    data_loader.load_accounts(config.ACCOUNTS_PATH)

    conn = storage.connect(config.DB_PATH)
    storage.init_db(conn)

    tiers = [StubTier()]
    entries = run_pipeline(batch.transactions, tiers, config.CONFIDENCE_THRESHOLD)
    storage.save_entries(conn, entries)

    result = report_mod.build_report(
        batch.transactions,
        entries,
        truth,
        notes=[
            "tiers 1-3 are not built yet; a stub classifier returns a fixed "
            "account at 0.1 confidence, so every row is flagged by design",
        ],
    )
    report_mod.render(result, console)
    report_mod.write_json(result, config.REPORT_PATH)
    console.print(f"wrote {config.REPORT_PATH.name}")

    conn.close()


if __name__ == "__main__":
    app()
