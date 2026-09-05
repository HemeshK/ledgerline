from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

ACCOUNTS_PATH = DATA_DIR / "chart_of_accounts.json"
BATCH_PATH = DATA_DIR / "batch.json"
TRUTH_PATH = DATA_DIR / "truth.json"

DB_PATH = ROOT / "ledgerline.db"
REPORT_PATH = ROOT / "report.json"

CONFIDENCE_THRESHOLD = 0.60
