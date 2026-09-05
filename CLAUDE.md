# ledgerline — build instructions

An AI finance controller that categorises messy bank transaction narrations into a chart of accounts, splits GST, and refuses to guess when it isn't sure.

## How to work

Build in the phases below, **in order**. Stop after each phase and report what you did. Do not start the next phase until told to continue.

After every phase the project must still run. Never leave it broken.

**End every phase with a commit and a push.** Message format: `phase N: short description`. This means the repo is presentable at every checkpoint, and if time runs out, whatever is pushed is a coherent submission rather than a half-written phase.

## Non-negotiables

- Python 3.11+. All money as `Decimal`, never float.
- Every tier implements one interface: `classify(txn) -> ClassificationResult`.
- The LLM is the most expensive and least reliable tier. It is called last and as rarely as possible.
- `python -m ledgerline run` must work from a clean clone **with no API key set** — the LLM tier skips gracefully and the report says so.
- No feature outside these phases. If something seems missing, note it in the README roadmap instead of building it.
- **Never commit a secret.** The API key lives in `.env`, which is gitignored. `.env.example` is committed with the key name and an empty value.

---

## Phase 0 — Repo setup

**Build:**

Initialise the repo and connect it to GitHub. Ask me for the remote URL if it isn't already configured — do not guess a username or create a repo under a made-up account.

```
ledgerline/
├── ledgerline/          package code
│   └── __init__.py
├── data/                generated batch, truth key, chart of accounts
├── tests/
├── CLAUDE.md
├── README.md            already written — do not overwrite
├── requirements.txt
├── .env.example
└── .gitignore
```

`.gitignore` must cover: `.env`, `*.db`, `__pycache__/`, `*.joblib`, `.venv/`, `.pytest_cache/`, `report.json`.

`data/` **is** committed — the batch and ground-truth key are part of the submission, since a reviewer needs to reproduce the numbers.

`.env.example` contains `GEMINI_API_KEY=`.

`requirements.txt`: pydantic, scikit-learn, joblib, google-genai, fastapi, uvicorn, rich, typer, python-dotenv, matplotlib, pytest, faker.

**Check:** clean `git status`, remote set, first commit pushed, README renders correctly on GitHub.

---

## Phase 1 — Synthetic data

**Build:**

`data/chart_of_accounts.json` — 30 accounts for an Indian SMB. Fields: `code`, `name`, `family` (revenue | expense | asset | liability | equity), `statement` (pnl | balance_sheet), `hints` (keywords).

Must include: 4100 Sales revenue, 4200 Service income, 5100 Salaries & wages, 5200 Rent expense, 5600 Utilities & telecom, 5900 Office supplies, 6200 Marketing expense, 6500 Professional fees, 6800 Miscellaneous expense, 1100 Bank account, 1200 Accounts receivable, 1450 GST input credit, 1500 Equipment (PPE), 2100 Accounts payable, 2300 GST payable, 2400 TDS payable. Add the rest to reach 30.

`generate.py` — writes `data/batch.json` (200 transactions) and `data/truth.json` (txn_id → correct account_code).

Transaction fields: `txn_id`, `date`, `amount` (Decimal), `direction` (credit | debit), `narration`, `counterparty_raw`.

Narration formats to mix, all realistic Indian bank strings:
```
NEFT/HDFC/SWIGGY DESIGNS PVT LTD/INV2291/DATED 05-AUG-2024
UPI/PAYTM/9876543210/RAVI KUMAR/CONSULTING
RTGS/AXIS/GST PAYMENT AUG24/GSTIN27AAACS1234B
IMPS/ICICI/RENT AUG24/PRESTIGE ESTATES
CR/NEFT/HDFC/ZOMATO INDIA/SETTLEMENT
```

The batch must contain, deliberately:
- ~120 straightforward rows
- ~25 GST-inclusive revenue rows (amount includes 18%, base must be split out)
- ~15 capex-vs-opex traps (a laptop or equipment purchase that belongs in 1500 Equipment, not 5900 Office supplies)
- ~10 inter-account transfers (hit neither P&L nor income)
- ~15 genuinely ambiguous rows — payee name collisions (a design agency whose name contains a food-delivery brand), individuals paid for services with no category signal, invoice references not matching anything
- ~15 professional-fee payments to individuals where TDS applies

`data/batch.json` must also record two disjoint vendor lists: `train_vendors` and `heldout_vendors`. **No vendor appears in both.** This is what makes the later classifier evaluation honest.

**Check:** `python generate.py` writes both files. Print 10 sample rows with their ground-truth account so they can be eyeballed. Confirm no vendor overlap between the two lists.

---

## Phase 2 — Models, storage, skeleton

**Build:**

Pydantic models: `Transaction`, `Account`, `ClassificationResult` (`account_code`, `confidence`, `method`, `reason`), `LedgerEntry` (`txn_id`, `account_code`, `amount`, `confidence`, `method`, `reason`, `status`), `RunReport`.

`status` is `posted` | `flagged` | `corrected`. `method` is `rule` | `classifier` | `llm`.

SQLite (`ledgerline.db`), three tables: `ledger_entries`, `payee_memory` (counterparty → account_code, with a count), `llm_cache` (narration hash → response).

Pipeline skeleton with a stub classifier returning a fixed code at confidence 0.1, so everything is flagged.

Report: total, attempted, flagged, accuracy on attempted, coverage, per-method breakdown, and a table of wrong rows with predicted vs actual. Printed with `rich`, also written to `report.json`.

**Check:** `python -m ledgerline run` completes and prints a report showing ~0% accuracy and 0% coverage. That's the correct result for a stub — the plumbing works.

---

## Phase 3 — Tier 1 rules

**Build:** exact payee lookup against `payee_memory`, then keyword matching against account `hints`. Exact payee match returns 0.95+; keyword match returns 0.65–0.85 by strength.

Confidence gate at 0.60. Above: post. Below: exception queue with a written reason a human can act on — "counterparty not seen before and narration has no category keyword", not "low confidence".

**Check:** report now shows a real accuracy and a coverage well below 100%. Flagged rows have specific reasons.

---

## Phase 4 — Tier 2 classifier

**Build:** `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))` → `LogisticRegression`. Train only on `train_vendors` rows. Confidence is `max(predict_proba)`. Persist with joblib.

Called only on rows tier 1 left unresolved.

Report must break out accuracy on `train_vendors` rows versus `heldout_vendors` rows separately.

**Check:** the held-out accuracy should be clearly lower than the seen-vendor accuracy. If it isn't, the generator made the two vendor sets too similar — say so rather than hiding it.

---

## Phase 5 — Tier 3 LLM

**Build:** `google-genai`, model `gemini-2.5-flash-lite`. Structured JSON output: `account_code`, `confidence`, `reason`. Prompt includes the chart of accounts with hints.

Called only on what tiers 1 and 2 couldn't resolve above threshold.

Cache every response in `llm_cache` keyed on the narration, so re-runs cost nothing. Retry with exponential backoff on HTTP 429. If `GEMINI_API_KEY` is unset, skip the tier, leave those rows flagged, and note it in the report.

Track and report LLM call count and call rate.

**Check:** runs with and without an API key. Second run with a key makes zero new calls (cache hit).

---

## Phase 6 — Tax split and P&L

**Build:** GST-inclusive revenue rows emit two ledger entries — base to 4100, the 18% component to 2300. Same for input credit on purchases (1450).

P&L as a group-by: sum revenue and expense families for the period. Balance sheet: sum asset, liability, equity cumulatively. Print both.

**Check:** one GST-inclusive transaction produces two ledger lines summing exactly to the original amount. No rounding drift.

---

## Phase 7 — Review UI

FastAPI serving JSON, single-page frontend. One screen: a review queue for the exception list and the full ledger.

**Endpoints:** `GET /api/entries` (filterable by status), `POST /api/entries/{txn_id}/confirm`, `POST /api/entries/{txn_id}/correct` (body: new account_code). A correction writes back to `payee_memory` so the next run improves — this closes the loop and is the point of the screen.

### Design direction

This is a working tool for an accountant, not a marketing page. It should feel like a well-made piece of financial software: dense, quiet, quick to scan, trustworthy. Data is the hero. No hero section, no cards, no gradients, no icons for decoration.

**Palette** — cool paper, not cream:
```
--paper:   #FBFBF9   page
--surface: #FFFFFF   table
--ink:     #14181F   primary text
--muted:   #6B7280   secondary text
--rule:    #E4E5E1   borders, 1px
--posted:  #0F766E   resolved
--flag:    #B45309   needs attention
--flagbg:  #FDF6EC   flagged row tint
```

**Type** — IBM Plex Sans for chrome and labels, IBM Plex Mono for all data: narrations, amounts, account codes. The mono is functional, not stylistic — these are machine strings and column alignment matters. Amounts right-aligned with tabular figures. Base size 13px in the table; this is a dense tool, not an article.

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│ Transaction Review    batch · 200 rows · 12 flagged   │
├──────────────────────────────────────────────────────┤
│ [All] [Flagged] [Posted] [Corrected]      search...   │
├──────────────────────────────────────────────────────┤
│ DATE   AMOUNT  NARRATION   ACCOUNT  CONF  VIA  STATUS │
│ ──────────────────────────────────────────────────────│
│ rows, 36px tall, hairline rules, no zebra striping     │
│ flagged rows tinted, everything else on white          │
│ ▼ expanded row: full narration, full reason,           │
│   tax split lines, and for corrections:                │
│   "predicted X (classifier, 73%) → corrected to Y"     │
└──────────────────────────────────────────────────────┘
```

**Rules:**
- Row click expands in place. No modals.
- Confidence: the number, plus a thin 40px bar. No donuts, no rings.
- `VIA` column shows Rule / Classifier / LLM as small plain-text tags, not coloured pills.
- Actions appear on row hover and in the expanded state — not persistently on every row, which is what caused the button collisions in the earlier mockup.
- Correcting an account uses a searchable dropdown of account codes.
- Empty flagged queue reads "Nothing needs review." — an accomplishment, not an error.

**Avoid:** all-caps eyebrow labels, meta strings joined with middle dots, `→` inside button text, drop shadows under everything, rounded corners above 4px, any animation that isn't responding to a click.

**Check:** the screen loads real data from the API, filters work, expanding a row shows the reason and tax split, and correcting a row changes the result of the next `run`.

---

## Phase 8 — Evidence

**Build:** head-to-head table (all three tiers over the same batch, same ground truth). Two matplotlib charts: tier comparison, and LLM call rate falling across the batch. A pytest that runs the batch and fails if accuracy drops below a threshold, wired to GitHub Actions.

**Check:** `pytest` passes; deliberately breaking a rule makes it fail.
