"""Generates data/batch.json (200 transactions) and data/truth.json (ground truth)."""

import json
import random
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

SEED = 42
GST_RATE = Decimal("0.18")

BANKS = ["HDFC", "ICICI", "AXIS", "SBI", "KOTAK", "YES"]

# Vendor lists are disjoint by construction: train_vendors and heldout_vendors
# never share a name. Each vendor is tagged with the account_code its
# transactions should resolve to, plus enough shape info to render a narration.

TRAIN_VENDORS = [
    {"name": "SWIGGY DESIGNS PVT LTD", "kind": "biz", "account": "4200"},
    {"name": "ZOMATO INDIA", "kind": "biz", "account": "4100"},
    {"name": "BRIGHTPATH CONSULTING LLP", "kind": "biz", "account": "4200"},
    {"name": "NIMBUS RETAIL PVT LTD", "kind": "biz", "account": "4100"},
    {"name": "PRESTIGE ESTATES", "kind": "biz", "account": "5200"},
    {"name": "AIRTEL BUSINESS", "kind": "biz", "account": "5600"},
    {"name": "JIO FIBER", "kind": "biz", "account": "5600"},
    {"name": "AMAZON WEB SERVICES", "kind": "biz", "account": "5700"},
    {"name": "FRESHWORKS SAAS", "kind": "biz", "account": "5700"},
    {"name": "OFFICE MART SUPPLIES", "kind": "biz", "account": "5900"},
    {"name": "STATIONERY HUB", "kind": "biz", "account": "5900"},
    {"name": "BLUE DART EXPRESS", "kind": "biz", "account": "6100"},
    {"name": "DIGITAL REACH MARKETING", "kind": "biz", "account": "6200"},
    {"name": "GOOGLE ADS INDIA", "kind": "biz", "account": "6200"},
    {"name": "SHARMA & ASSOCIATES CA", "kind": "biz", "account": "6500"},
    {"name": "APEX LEGAL SERVICES", "kind": "biz", "account": "6500"},
    {"name": "ICICI LOMBARD INSURANCE", "kind": "biz", "account": "5500"},
    {"name": "QUICKFIX REPAIRS", "kind": "biz", "account": "5400"},
    {"name": "TATA POWER", "kind": "biz", "account": "5600"},
    {"name": "DELL TECHNOLOGIES INDIA", "kind": "biz", "account": "1500"},
    {"name": "RAVI KUMAR", "kind": "individual", "account": "4200"},
    {"name": "ANITA DESAI", "kind": "individual", "account": "6500"},
    {"name": "SURESH MENON", "kind": "individual", "account": "6500"},
    {"name": "PRIYA NAIR", "kind": "individual", "account": "6500"},
    {"name": "MERIDIAN CAPITAL FD", "kind": "biz", "account": "4300"},
    {"name": "HDFC BANK LOAN CELL", "kind": "biz", "account": "6600"},
    {"name": "SELF TRANSFER SAVINGS", "kind": "biz", "account": "1100"},
    {"name": "SKILLBRIDGE ACADEMY", "kind": "biz", "account": "6400"},
    {"name": "CARE FOUNDATION TRUST", "kind": "biz", "account": "6700"},
    {"name": "GST DEPARTMENT", "kind": "biz", "account": "2300"},
]

HELDOUT_VENDORS = [
    {"name": "URBAN CRUST FOODS PVT LTD", "kind": "biz", "account": "4100"},
    {"name": "SPARKLINE CONSULTING", "kind": "biz", "account": "4200"},
    {"name": "GREENLEAF TRADERS", "kind": "biz", "account": "4100"},
    {"name": "SKYLINE ESTATES", "kind": "biz", "account": "5200"},
    {"name": "VODAFONE IDEA BUSINESS", "kind": "biz", "account": "5600"},
    {"name": "BSNL BROADBAND", "kind": "biz", "account": "5600"},
    {"name": "MICROSOFT AZURE", "kind": "biz", "account": "5700"},
    {"name": "ZOHO CORPORATION", "kind": "biz", "account": "5700"},
    {"name": "PAPERTRAIL SUPPLIES", "kind": "biz", "account": "5900"},
    {"name": "DELHIVERY LOGISTICS", "kind": "biz", "account": "6100"},
    {"name": "ADPULSE MEDIA", "kind": "biz", "account": "6200"},
    {"name": "META ADS INDIA", "kind": "biz", "account": "6200"},
    {"name": "VERMA & CO CHARTERED ACCOUNTANTS", "kind": "biz", "account": "6500"},
    {"name": "NORTHSTAR LEGAL", "kind": "biz", "account": "6500"},
    {"name": "BAJAJ ALLIANZ INSURANCE", "kind": "biz", "account": "5500"},
    {"name": "RAPIDFIX SERVICES", "kind": "biz", "account": "5400"},
    {"name": "ADANI ELECTRICITY", "kind": "biz", "account": "5600"},
    {"name": "LENOVO INDIA", "kind": "biz", "account": "1500"},
    {"name": "DEEPAK IYER", "kind": "individual", "account": "4200"},
    {"name": "KAVITA RAO", "kind": "individual", "account": "6500"},
    {"name": "MANOJ PILLAI", "kind": "individual", "account": "6500"},
    {"name": "SUNIL D'SOUZA", "kind": "individual", "account": "6500"},
    {"name": "TRUSTLINE CAPITAL FD", "kind": "biz", "account": "4300"},
    {"name": "AXIS BANK LOAN CELL", "kind": "biz", "account": "6600"},
    {"name": "SELF TRANSFER CURRENT", "kind": "biz", "account": "1100"},
    {"name": "LEARNWELL INSTITUTE", "kind": "biz", "account": "6400"},
    {"name": "HOPE FOUNDATION TRUST", "kind": "biz", "account": "6700"},
]

# Ambiguous payees: names deliberately collide with an unrelated category,
# or otherwise carry no clean keyword signal. Not in either vendor list.
AMBIGUOUS_VENDORS = [
    {"name": "SWIGGY INTERIOR DESIGNS", "account": "4200", "note": "food-brand collision, actually a design agency"},
    {"name": "ZOMATO CONSULTANTS LLP", "account": "4200", "note": "food-brand collision, actually consulting"},
    {"name": "RAJESH GUPTA", "account": "6500", "note": "individual paid for services, no category keyword"},
    {"name": "MEENA IYER", "account": "4200", "note": "individual paid for services, no category keyword"},
    {"name": "VIKRAM SINGH", "account": "6500", "note": "individual paid for services, no category keyword"},
    {"name": "STARDOM ENTERTAINMENT", "account": "6200", "note": "generic name, could be many things"},
    {"name": "APEX", "account": "4200", "note": "single-word name, invoice ref doesn't match anything on file"},
    {"name": "BLUEWAVE", "account": "4100", "note": "single-word name, no category signal"},
    {"name": "OM ENTERPRISES", "account": "5900", "note": "generic trading-name, no keyword"},
    {"name": "SAI TRADERS", "account": "4100", "note": "generic trading-name, no keyword"},
    {"name": "KRISHNA ASSOCIATES", "account": "6500", "note": "generic name shared by many business types"},
    {"name": "NEW HORIZON PVT LTD", "account": "4100", "note": "generic name, invoice ref INV-9999 matches nothing"},
    {"name": "FIRSTCHOICE VENTURES", "account": "6200", "note": "generic name, no keyword"},
    {"name": "UNITY SOLUTIONS", "account": "4200", "note": "generic name, no keyword"},
    {"name": "GLOBALINK TRADERS", "account": "4100", "note": "generic name, invoice ref mismatched"},
]

# No-signal rows: undecidable, not merely ambiguous. Bare account numbers,
# generic UPI handles, no name, no narration content at all.
NO_SIGNAL_ROWS = [
    {"template": "NEFT/{bank}/9876543210", "account": None},
    {"template": "UPI/PAYTM/PAYMENT", "account": None},
    {"template": "IMPS/{bank}/TRF", "account": None},
    {"template": "NEFT/{bank}/8123456789", "account": None},
    {"template": "UPI/PHONEPE/PAYMENT", "account": None},
    {"template": "IMPS/{bank}/7009988776", "account": None},
    {"template": "NEFT/{bank}/TRANSFER", "account": None},
    {"template": "UPI/GPAY/9988776655", "account": None},
]

INDIVIDUAL_UPI_HANDLES = ["9876543210", "9123456780", "9001122334", "8765432109", "7890012345"]


def random_date(start: date, end: date, rng: random.Random) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def fmt_date(d: date) -> str:
    return d.strftime("%d-%b-%Y").upper()


def round2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def make_narration_neft(bank, name, ref, extra=None, prefix="NEFT"):
    parts = [prefix, bank, name, ref]
    if extra:
        parts.append(extra)
    return "/".join(parts)


def make_narration_upi(handle, name, note=None):
    parts = ["UPI", "PAYTM", handle, name]
    if note:
        parts.append(note)
    return "/".join(parts)


def gen_straightforward(txn_id, rng, vendors):
    v = rng.choice(vendors)
    bank = rng.choice(BANKS)
    d = random_date(date(2024, 8, 1), date(2024, 8, 31), rng)
    account = v["account"]
    is_revenue = account.startswith("4")
    direction = "credit" if is_revenue else "debit"
    amount = round2(Decimal(rng.randrange(2000, 250000)) / 100)

    if v.get("kind") == "individual":
        handle = rng.choice(INDIVIDUAL_UPI_HANDLES)
        note = "CONSULTING" if account in ("4200", "6500") else None
        narration = make_narration_upi(handle, v["name"], note)
    else:
        ref = f"INV{rng.randint(1000, 9999)}/DATED {fmt_date(d)}"
        rail = rng.choice(["NEFT", "RTGS", "IMPS"])
        narration = make_narration_neft(bank, v["name"], ref, prefix=rail)
        if direction == "credit" and rng.random() < 0.3:
            narration = "CR/" + narration

    return {
        "txn_id": txn_id,
        "date": d.isoformat(),
        "amount": str(amount),
        "direction": direction,
        "narration": narration,
        "counterparty_raw": v["name"],
    }, account


def gen_gst_inclusive_revenue(txn_id, rng, vendors):
    revenue_vendors = [v for v in vendors if v["account"] in ("4100", "4200")]
    v = rng.choice(revenue_vendors)
    bank = rng.choice(BANKS)
    d = random_date(date(2024, 8, 1), date(2024, 8, 31), rng)
    base = round2(Decimal(rng.randrange(5000, 150000)) / 100)
    total = round2(base * (Decimal("1") + GST_RATE))
    ref = f"INV{rng.randint(1000, 9999)}/DATED {fmt_date(d)}/GST INCL"
    narration = make_narration_neft(bank, v["name"], ref, prefix="NEFT")
    return {
        "txn_id": txn_id,
        "date": d.isoformat(),
        "amount": str(total),
        "direction": "credit",
        "narration": narration,
        "counterparty_raw": v["name"],
    }, v["account"], "gst_inclusive_revenue"


def gen_capex_trap(txn_id, rng, vendors):
    equip_vendors = [v for v in vendors if v["account"] == "1500"]
    v = rng.choice(equip_vendors)
    bank = rng.choice(BANKS)
    d = random_date(date(2024, 8, 1), date(2024, 8, 31), rng)
    amount = round2(Decimal(rng.randrange(3000000, 12000000)) / 100)
    item = rng.choice(["LAPTOP", "DESKTOP", "OFFICE CHAIR SET", "PRINTER UNIT", "MONITOR"])
    ref = f"PO{rng.randint(100, 999)}/{item}"
    narration = make_narration_neft(bank, v["name"], ref, prefix="NEFT")
    return {
        "txn_id": txn_id,
        "date": d.isoformat(),
        "amount": str(amount),
        "direction": "debit",
        "narration": narration,
        "counterparty_raw": v["name"],
    }, "1500", "capex_trap"


def gen_transfer(txn_id, rng, vendors):
    v = rng.choice([x for x in vendors if x["account"] == "1100"])
    bank = rng.choice(BANKS)
    d = random_date(date(2024, 8, 1), date(2024, 8, 31), rng)
    amount = round2(Decimal(rng.randrange(1000000, 50000000)) / 100)
    direction = rng.choice(["credit", "debit"])
    narration = make_narration_neft(bank, v["name"], "SELF/OWN ACCOUNT", prefix="IMPS")
    return {
        "txn_id": txn_id,
        "date": d.isoformat(),
        "amount": str(amount),
        "direction": direction,
        "narration": narration,
        "counterparty_raw": v["name"],
    }, "1100", "transfer"


def gen_ambiguous(txn_id, rng, idx):
    a = AMBIGUOUS_VENDORS[idx % len(AMBIGUOUS_VENDORS)]
    bank = rng.choice(BANKS)
    d = random_date(date(2024, 8, 1), date(2024, 8, 31), rng)
    account = a["account"]
    direction = "credit" if account.startswith("4") else "debit"
    amount = round2(Decimal(rng.randrange(2000, 200000)) / 100)
    ref = f"INV-{rng.randint(9000, 9999)}"
    if rng.random() < 0.5:
        narration = make_narration_upi(rng.choice(INDIVIDUAL_UPI_HANDLES), a["name"])
    else:
        rail = rng.choice(["NEFT", "IMPS"])
        narration = make_narration_neft(bank, a["name"], ref, prefix=rail)
    return {
        "txn_id": txn_id,
        "date": d.isoformat(),
        "amount": str(amount),
        "direction": direction,
        "narration": narration,
        "counterparty_raw": a["name"],
    }, account, "ambiguous", a["note"]


def gen_tds_professional(txn_id, rng, vendors):
    individuals = [v for v in vendors if v.get("kind") == "individual"]
    v = rng.choice(individuals)
    bank = rng.choice(BANKS)
    d = random_date(date(2024, 8, 1), date(2024, 8, 31), rng)
    gross = round2(Decimal(rng.randrange(1000000, 8000000)) / 100)
    handle = rng.choice(INDIVIDUAL_UPI_HANDLES)
    narration = make_narration_upi(handle, v["name"], "PROFESSIONAL FEES TDS DEDUCTED")
    return {
        "txn_id": txn_id,
        "date": d.isoformat(),
        "amount": str(gross),
        "direction": "debit",
        "narration": narration,
        "counterparty_raw": v["name"],
    }, "6500", "tds_professional"


def gen_no_signal(txn_id, rng, idx):
    row = NO_SIGNAL_ROWS[idx % len(NO_SIGNAL_ROWS)]
    bank = rng.choice(BANKS)
    d = random_date(date(2024, 8, 1), date(2024, 8, 31), rng)
    amount = round2(Decimal(rng.randrange(500, 50000)) / 100)
    direction = rng.choice(["credit", "debit"])
    narration = row["template"].format(bank=bank)
    return {
        "txn_id": txn_id,
        "date": d.isoformat(),
        "amount": str(amount),
        "direction": direction,
        "narration": narration,
        "counterparty_raw": "",
    }, None, "no_signal"


def gen_bank_charge_gst(txn_id, rng):
    bank = rng.choice(BANKS)
    d = random_date(date(2024, 8, 1), date(2024, 8, 31), rng)
    base = round2(Decimal(rng.randrange(100, 2000)) / 100)
    total = round2(base * (Decimal("1") + GST_RATE))
    narration = f"DEBIT/{bank}/BANK CHARGES GST INCL/SERVICE CHARGE"
    return {
        "txn_id": txn_id,
        "date": d.isoformat(),
        "amount": str(total),
        "direction": "debit",
        "narration": narration,
        "counterparty_raw": f"{bank} BANK",
    }, "5800", "bank_charge_gst"


def main():
    rng = random.Random(SEED)
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    batch = []
    truth = {}
    meta = {}

    txn_counter = 1

    def next_id():
        nonlocal txn_counter
        tid = f"T{txn_counter:04d}"
        txn_counter += 1
        return tid

    all_vendors = TRAIN_VENDORS + HELDOUT_VENDORS

    # Straightforward rows, split across train/heldout vendors. Trimmed from
    # the nominal ~120 to 105 so the no-signal and bank-charge-GST additions
    # below still fit the 200-row batch total.
    for _ in range(53):
        tid = next_id()
        row, acct = gen_straightforward(tid, rng, TRAIN_VENDORS)
        batch.append(row)
        truth[tid] = acct
    for _ in range(52):
        tid = next_id()
        row, acct = gen_straightforward(tid, rng, HELDOUT_VENDORS)
        batch.append(row)
        truth[tid] = acct

    # ~25 GST-inclusive revenue rows
    for _ in range(25):
        tid = next_id()
        row, acct, tag = gen_gst_inclusive_revenue(tid, rng, all_vendors)
        batch.append(row)
        truth[tid] = acct
        meta[tid] = tag

    # ~15 capex-vs-opex traps
    for _ in range(15):
        tid = next_id()
        row, acct, tag = gen_capex_trap(tid, rng, all_vendors)
        batch.append(row)
        truth[tid] = acct
        meta[tid] = tag

    # ~10 inter-account transfers
    for _ in range(10):
        tid = next_id()
        row, acct, tag = gen_transfer(tid, rng, all_vendors)
        batch.append(row)
        truth[tid] = acct
        meta[tid] = tag

    # ~15 genuinely ambiguous rows
    for i in range(15):
        tid = next_id()
        row, acct, tag, note = gen_ambiguous(tid, rng, i)
        batch.append(row)
        truth[tid] = acct
        meta[tid] = f"{tag}: {note}"

    # ~15 professional-fee payments to individuals where TDS applies
    for _ in range(15):
        tid = next_id()
        row, acct, tag = gen_tds_professional(tid, rng, all_vendors)
        batch.append(row)
        truth[tid] = acct
        meta[tid] = tag

    # ~8 no-signal rows (undecidable, correct answer is "flag it")
    for i in range(8):
        tid = next_id()
        row, acct, tag = gen_no_signal(tid, rng, i)
        batch.append(row)
        truth[tid] = acct
        meta[tid] = tag

    # bank-charge rows carrying GST, to exercise input credit on the expense side
    for _ in range(7):
        tid = next_id()
        row, acct, tag = gen_bank_charge_gst(tid, rng)
        batch.append(row)
        truth[tid] = acct
        meta[tid] = tag

    rng.shuffle(batch)

    train_vendor_names = sorted({v["name"] for v in TRAIN_VENDORS})
    heldout_vendor_names = sorted({v["name"] for v in HELDOUT_VENDORS})
    overlap = set(train_vendor_names) & set(heldout_vendor_names)
    if overlap:
        raise SystemExit(f"train/heldout vendor overlap detected: {overlap}")

    batch_doc = {
        "train_vendors": train_vendor_names,
        "heldout_vendors": heldout_vendor_names,
        "transactions": batch,
    }

    (out_dir / "batch.json").write_text(json.dumps(batch_doc, indent=2))
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    print(f"Wrote {len(batch)} transactions to data/batch.json")
    print(f"Wrote {len(truth)} ground-truth entries to data/truth.json")
    print(f"train_vendors: {len(train_vendor_names)}, heldout_vendors: {len(heldout_vendor_names)}, overlap: {len(overlap)}")

    print("\nSample rows:\n")
    sample_ids = []
    ambiguous_ids = [tid for tid, tag in meta.items() if tag.startswith("ambiguous")][:3]
    no_signal_ids = [tid for tid, tag in meta.items() if tag == "no_signal"][:2]
    sample_ids.extend(ambiguous_ids)
    sample_ids.extend(no_signal_ids)
    remaining = [r["txn_id"] for r in batch if r["txn_id"] not in sample_ids]
    rng.shuffle(remaining)
    sample_ids.extend(remaining[: max(0, 10 - len(sample_ids))])

    by_id = {r["txn_id"]: r for r in batch}
    for tid in sample_ids[:10]:
        r = by_id[tid]
        acct = truth[tid]
        note = meta.get(tid, "")
        print(f"{tid} | {r['date']} | {r['direction']:>6} | {r['amount']:>10} | {r['narration']}")
        print(f"       -> account: {acct}   {('(' + note + ')') if note else ''}")


if __name__ == "__main__":
    main()
