"""
Download training datasets for TokenCraft-AI.

Run from project root:
    python scripts/download_datasets.py
"""

from __future__ import annotations

import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

URLS = {
    "the-verdict.txt": (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
        "main/ch02/01_main-chapter-code/the-verdict.txt"
    ),
    "instruction-data.json": (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/"
        "main/ch07/01_main-chapter-code/instruction-data.json"
    ),
    "sms_spam_collection.zip": (
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    ),
}


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
        return ctx
    except Exception:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading: {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, context=_ssl_context()) as response:
        dest.write_bytes(response.read())


def prepare_spam_csv(tsv_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"  skip (exists): {out_path.name}")
        return
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["Label", "Text"])
    num_spam = (df["Label"] == "spam").sum()
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced = balanced.sample(frac=1, random_state=123).reset_index(drop=True)
    balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})
    split = int(0.9 * len(balanced))
    train_df = balanced[:split]
    train_df.to_csv(out_path, index=False)
    balanced[split:].to_csv(DATA_DIR / "validation.csv", index=False)
    print(f"  wrote {out_path.name} ({len(train_df)} rows), validation.csv")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR}\n")

    print("Text & instruction datasets:")
    for name, url in (
        ("the-verdict.txt", URLS["the-verdict.txt"]),
        ("instruction-data.json", URLS["instruction-data.json"]),
    ):
        download_file(url, DATA_DIR / name)

    print("\nSMS spam collection:")
    zip_path = DATA_DIR / "sms_spam_collection.zip"
    download_file(URLS["sms_spam_collection.zip"], zip_path)

    tsv_path = DATA_DIR / "sms_spam_collection" / "SMSSpamCollection.tsv"
    if not tsv_path.exists() and zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR / "sms_spam_collection")
        raw = DATA_DIR / "sms_spam_collection" / "SMSSpamCollection"
        if raw.exists() and not tsv_path.exists():
            raw.rename(tsv_path)
        print(f"  extracted: {tsv_path.name}")

    if tsv_path.exists():
        prepare_spam_csv(tsv_path, DATA_DIR / "train.csv")

    verdict = DATA_DIR / "the-verdict.txt"
    if verdict.exists():
        text = verdict.read_text(encoding="utf-8")
        print(f"\n{verdict.name}: {len(text)} chars")

    inst = DATA_DIR / "instruction-data.json"
    if inst.exists():
        with inst.open(encoding="utf-8") as f:
            n = len(json.load(f))
        print(f"{inst.name}: {n} entries")

    print("\nDone.")


if __name__ == "__main__":
    main()
