#!/usr/bin/env python3
"""Regenerate resources/languages/<lang>/function_words.txt from UD CoNLL-U treebanks.

Usage:
    python scripts/regenerate_function_words.py --lang tr \
        --treebank path/to/UD_Turkish-BOUN \
        --out src/bitig/resources/languages/tr/function_words.txt

Fetches all tokens tagged with closed-class UPOS (DET PRON ADP CCONJ SCONJ AUX PART), counts
lowercased frequencies, writes the top N. A header comment records source + generation date.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from collections import Counter
from pathlib import Path

_CLOSED_UPOS = {"DET", "PRON", "ADP", "CCONJ", "SCONJ", "AUX", "PART"}


def parse_conllu(path: Path) -> list[tuple[str, str]]:
    """Yield (form, upos) pairs from a CoNLL-U file. Skips comments + multi-word tokens."""
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            idx = fields[0]
            if "-" in idx or "." in idx:
                continue  # multi-word token range or empty node
            form, upos = fields[1], fields[3]
            out.append((form, upos))
    return out


def count_closed_class(treebank_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for f in treebank_dir.glob("*.conllu"):
        for form, upos in parse_conllu(f):
            if upos in _CLOSED_UPOS:
                counts[form.lower()] += 1
    return counts


def write_list(
    out_path: Path,
    counts: Counter[str],
    n: int,
    lang: str,
    source: str,
) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    header = [
        f"# Function-word list for {lang} — top {n} closed-class tokens by frequency",
        f"# UPOS filter: {sorted(_CLOSED_UPOS)}",
        f"# Source treebank(s): {source}",
        f"# Generated: {dt.date.today().isoformat()} by scripts/regenerate_function_words.py @ {commit}",
    ]
    top = [w for w, _ in counts.most_common(n)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join([*header, "", *top]) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--treebank", required=True, type=Path, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    merged: Counter[str] = Counter()
    names = []
    for tb in args.treebank:
        merged.update(count_closed_class(tb))
        names.append(tb.name)

    write_list(args.out, merged, args.n, args.lang, source=" + ".join(names))
    print(f"Wrote {args.n} tokens to {args.out}")


if __name__ == "__main__":
    main()
