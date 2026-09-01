# ledgerline

An AI finance controller that categorises messy bank and payment-gateway transactions into a chart of accounts, splits tax out of revenue, and — critically — refuses to guess when it isn't sure.

**Status: in active development.** Core pipeline and evaluation harness are being built. See [Roadmap](#roadmap) for what's done and what isn't.

---

## The problem

Every business bank statement is a list of strings like this:

```
NEFT/HDFC/SWIGGY DESIGNS PVT LTD/INV2291/DATED 05-AUG-2024
```

Somebody has to decide which account that hits. Today that somebody is a human, working through hundreds of rows a month, and the work is slow, repetitive, and easy to get subtly wrong.

The obvious fix is to throw an LLM at it. That produces a categoriser that is confidently wrong some fraction of the time, which in accounting is worse than useless — a bad entry propagates silently into the P&L and gets discovered a quarter later.

`ledgerline` is built on a different premise: **the valuable output isn't a label, it's a trustworthy label plus an honest list of what couldn't be labelled.**

## Approach

A three-tier cascade, cheapest first. Each tier only sees what the previous one couldn't resolve.

| Tier | Method | Handles |
|---|---|---|
| 1 | Deterministic rules + payee memory | Known counterparties, exact patterns |
| 2 | TF-IDF character n-grams → logistic regression | Vendors seen before in a different format |
| 3 | LLM | Genuinely novel or ambiguous rows |

Two things follow from this design:

**The LLM is the most expensive and least reliable tier, so the system is built to call it as little as possible.** LLM call rate is tracked across each batch and falls as payee memory grows.

**Every tier emits a calibrated confidence.** Below a threshold, the row is not posted — it goes to an exception queue with a written reason a human can act on. Coverage and accuracy are reported as two separate numbers, because a system that flags everything has perfect accuracy and is worthless.

## Architecture

```
                 ┌──────────────┐
   transactions →│  Tier 1      │→ resolved ─┐
                 │  rules       │            │
                 └──────┬───────┘            │
                        │ residue            │
                 ┌──────▼───────┐            │
                 │  Tier 2      │→ resolved ─┤
                 │  classifier  │            │
                 └──────┬───────┘            │
                        │ residue            │
                 ┌──────▼───────┐            │
                 │  Tier 3      │→ resolved ─┤
                 │  LLM         │            │
                 └──────────────┘            │
                                             ▼
                                    ┌─────────────────┐
                                    │ confidence gate │
                                    └────┬───────┬────┘
                                         │       │
                                   posted│       │flagged
                                         ▼       ▼
                                    ┌────────┐ ┌───────────┐
                                    │ ledger │ │ exception │
                                    └───┬────┘ │   queue   │
                                        │      └─────┬─────┘
                                        │            │ human resolves
                                        │            └──────────────┐
                                        ▼                           │
                              ┌───────────────────┐                 │
                              │ P&L, balance sheet│                 │
                              └───────────────────┘                 │
                                                                    ▼
                                                          writes back to
                                                          Tier 1 payee memory
```

Human corrections feed back into tier-1 memory, so resolving an exception makes the next run cheaper and more accurate.

## Evaluation

The project is evaluated against a synthetic batch of transactions with a ground-truth key. Reported metrics:

- **Accuracy on attempted** — of the rows it chose to post, how many were right
- **Coverage** — what fraction it was willing to attempt at all
- **Per-tier breakdown** — accuracy and volume for rules, classifier, and LLM separately
- **Error list** — every row it got wrong, with the predicted and correct account

The train/test split is **vendor-disjoint**, not random. A random split leaks the same counterparties into both sides and inflates the classifier's score, since the task is largely payee recognition. Held-out vendors are ones the classifier has genuinely never seen.

A CI check runs the full batch on every push and fails if accuracy regresses.

## Domain notes

Some things the system has to get right that aren't obvious from a pure ML framing:

- **Capex vs opex.** A laptop purchase is not an expense — it's an asset swap, and it belongs on the balance sheet, not the P&L.
- **GST.** An invoice amount is usually tax-inclusive. Output tax on sales is a liability; input credit on purchases is an asset. Failing to split it overstates revenue.
- **Transfers aren't income.** Money moving between a company's own accounts hits neither statement.
- **One transaction, multiple lines.** A single inflow can post to two or three accounts at once.

## Stack

Python · scikit-learn · Anthropic API · SQLite · pydantic · FastAPI · pytest

## Roadmap

- [ ] Chart of accounts and core data models
- [ ] Synthetic batch generator with ground-truth key
- [ ] Pipeline skeleton and evaluation harness
- [ ] Tier 1: rules and payee memory
- [ ] Tier 2: classifier with vendor-disjoint evaluation
- [ ] Tier 3: LLM and confidence gate
- [ ] Tax splitting and P&L generation
- [ ] Review UI and human-in-the-loop correction
- [ ] Head-to-head tier comparison and CI regression gate

**Deliberately out of scope for v1**, and noted here so the omissions read as choices rather than gaps: a depreciation and accruals scheduler, settlement fan-out (commission / reserve / payout), and a ratio and cash-flow forecasting layer. Each of these consumes the ledger this project produces, and none of them is meaningful until the categorisation underneath is measurably correct.

## Design notes

The classifier sits behind a single `classify(transaction) -> (account_code, confidence, reason)` interface. Swapping the synthetic-trained baseline for one trained on real ledger history is one class implementation and a config change.

---

*Built for the Razorpay AI Buildathon, Track 04 — AI Finance Controller.*
