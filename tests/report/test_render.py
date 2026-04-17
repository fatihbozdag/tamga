"""Tests for report rendering."""

import json
from datetime import datetime
from pathlib import Path

from tamga.report import build_report


def _make_result_dir(tmp_path: Path) -> Path:
    d = tmp_path / "delta"
    d.mkdir()
    (d / "result.json").write_text(
        json.dumps(
            {
                "method_name": "burrows_delta",
                "params": {"method": "burrows", "mfw": 500},
                "values": {},
                "provenance": {
                    "tamga_version": "0.1.0.dev0",
                    "python_version": "3.11",
                    "spacy_model": "en",
                    "spacy_version": "3.7",
                    "corpus_hash": "deadbeef",
                    "feature_hash": None,
                    "seed": 42,
                    "timestamp": datetime.now().isoformat(),
                    "resolved_config": {"name": "demo"},
                },
            }
        )
    )
    return d


def test_build_report_html(tmp_path: Path) -> None:
    result_dir = _make_result_dir(tmp_path)
    out = tmp_path / "report.html"
    build_report(result_dir, output=out, format="html", title="Demo report")
    text = out.read_text()
    assert "<html" in text.lower()
    assert "Demo report" in text
    assert "burrows_delta" in text


def test_build_report_md(tmp_path: Path) -> None:
    result_dir = _make_result_dir(tmp_path)
    out = tmp_path / "report.md"
    build_report(result_dir, output=out, format="md", title="Demo")
    text = out.read_text()
    assert text.startswith("# Demo")
    assert "burrows_delta" in text


def test_build_report_multi_method(tmp_path: Path) -> None:
    parent = tmp_path / "run"
    parent.mkdir()
    for method in ["delta", "zeta"]:
        sub = parent / method
        sub.mkdir()
        (sub / "result.json").write_text(
            json.dumps(
                {
                    "method_name": method,
                    "params": {},
                    "values": {},
                    "provenance": {
                        "tamga_version": "0.1.0.dev0",
                        "python_version": "3.11",
                        "spacy_model": "en",
                        "spacy_version": "3.7",
                        "corpus_hash": "x",
                        "feature_hash": None,
                        "seed": 42,
                        "timestamp": datetime.now().isoformat(),
                        "resolved_config": {},
                    },
                }
            )
        )
    out = tmp_path / "report.html"
    build_report(parent, output=out, format="html", title="Multi")
    text = out.read_text()
    assert "delta" in text and "zeta" in text
