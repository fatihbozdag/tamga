"""Fetch Ömer Seyfettin's short stories from tr.wikisource.org.

Stories are public domain in Turkey (Seyfettin died in 1920; the 70-year
post-mortem clock expired in 1991). Wikisource transcriptions are released
under CC BY-SA 4.0; this script attributes back via a manifest.json next to
the corpus.

Output: examples/turkish_seyfettin/corpus/<slugified-title>.txt
        examples/turkish_seyfettin/manifest.json (titles + URLs)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://tr.wikisource.org/w/api.php"
UA = "bitig-stylometry-research/0.1 (https://github.com/fatihbozdag/bitig)"
CATEGORY = "Kategori:Ömer Seyfettin hikayeleri"


def _api(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode({**params, 'format': 'json'})}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _list_titles(category: str, limit: int) -> list[str]:
    titles: list[str] = []
    cont: dict = {}
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": min(50, limit - len(titles)),
            "cmtype": "page",
            **cont,
        }
        data = _api(params)
        titles.extend(m["title"] for m in data.get("query", {}).get("categorymembers", []))
        cont = data.get("continue", {})
        if not cont:
            break
    return titles[:limit]


_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^/]*/>", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NAMESPACED_LINK_RE = re.compile(
    r"\[\[(?:Kategori|Category|Resim|Image|Dosya|File):[^\]]+\]\]", re.IGNORECASE
)
_LINK_RE = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]")
_EXT_LINK_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]|\[https?://\S+\]")
_HEADER_RE = re.compile(r"^=+\s*(.*?)\s*=+$", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
_NEWLINES_RE = re.compile(r"\n{3,}")
# Lines that survived the link strip and turn out to be unbracketed namespace
# declarations (e.g. ``Kategori:Ömer Seyfettin hikayeleri`` at end of file).
_BARE_NAMESPACE_LINE_RE = re.compile(
    r"^\s*(?:Kategori|Category|Resim|Image|Dosya|File):[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)


def _strip_wiki(text: str) -> str:
    """Drop wiki markup. Iterates a few times to handle nested templates."""
    prev = None
    while prev != text:
        prev = text
        text = _TEMPLATE_RE.sub("", text)
    text = _REF_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _NAMESPACED_LINK_RE.sub("", text)
    text = _LINK_RE.sub(lambda m: m.group(2) or m.group(1) or m.group(3), text)
    text = _EXT_LINK_RE.sub(lambda m: m.group(1) or "", text)
    text = _BARE_NAMESPACE_LINE_RE.sub("", text)
    text = _HEADER_RE.sub(r"\1", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _slugify(title: str) -> str:
    """Make an ASCII filename safe for tooling. Keeps Turkish letters? No -- ASCII."""
    norm = unicodedata.normalize("NFKD", title)
    ascii_only = "".join(c for c in norm if not unicodedata.combining(c))
    out = re.sub(r"[^A-Za-z0-9]+", "_", ascii_only).strip("_").lower()
    return out or "untitled"


def fetch(out_dir: Path, n: int, sleep_s: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Listing up to {n} stories from category {CATEGORY!r}...")
    titles = _list_titles(CATEGORY, n)
    print(f"  found {len(titles)} stories")
    manifest: list[dict] = []
    for i, title in enumerate(titles, 1):
        slug = _slugify(title)
        target = out_dir / f"{slug}.txt"
        url = f"https://tr.wikisource.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        if target.exists():
            print(f"  [{i:>2}/{len(titles)}] {title:<40} (cached)")
            manifest.append({"title": title, "slug": slug, "url": url})
            continue
        try:
            data = _api({"action": "parse", "page": title, "prop": "wikitext"})
        except Exception as exc:
            print(f"  [{i:>2}/{len(titles)}] {title:<40} FAIL: {exc}")
            continue
        wt = data.get("parse", {}).get("wikitext", {}).get("*", "")
        body = _strip_wiki(wt)
        n_tok = len(body.split())
        if n_tok < 200:
            print(f"  [{i:>2}/{len(titles)}] {title:<40} too short ({n_tok} tokens), skipping")
            continue
        target.write_text(body, encoding="utf-8")
        print(f"  [{i:>2}/{len(titles)}] {title:<40} -> {slug}.txt ({n_tok} tokens)")
        manifest.append({"title": title, "slug": slug, "url": url, "tokens": n_tok})
        time.sleep(sleep_s)

    (out_dir.parent / "manifest.json").write_text(
        json.dumps(
            {
                "author": "Ömer Seyfettin",
                "source": "tr.wikisource.org",
                "license": "CC BY-SA 4.0 (transcription); underlying texts public domain",
                "stories": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote manifest -> {out_dir.parent / 'manifest.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "corpus",
        help="Corpus output directory.",
    )
    parser.add_argument("--n", type=int, default=30, help="How many stories to fetch.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Seconds to sleep between API calls (be polite to Wikisource).",
    )
    args = parser.parse_args()
    fetch(args.out, args.n, args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
