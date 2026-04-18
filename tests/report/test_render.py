"""Tests for report rendering."""

import json
from datetime import datetime
from pathlib import Path

from tamga.report import build_forensic_report, build_report


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


def _make_forensic_result_dir(tmp_path: Path, *, with_custody: bool = True) -> Path:
    d = tmp_path / "gi"
    d.mkdir()
    provenance: dict = {
        "tamga_version": "0.1.0.dev0",
        "python_version": "3.11",
        "spacy_model": "en",
        "spacy_version": "3.7",
        "corpus_hash": "deadbeef",
        "feature_hash": None,
        "seed": 42,
        "timestamp": datetime.now().isoformat(),
        "resolved_config": {},
    }
    if with_custody:
        provenance.update(
            {
                "questioned_description": "threat letter acquired 2026-04-15",
                "known_description": "15 emails from suspect",
                "hypothesis_pair": "H1: suspect wrote it; H0: someone else wrote it",
                "acquisition_notes": "drive image from warrant A-2026-0412",
                "custody_notes": "no modifications after acquisition",
                "source_hashes": {"q1": "abc123", "k1": "def456"},
            }
        )
    (d / "result.json").write_text(
        json.dumps(
            {
                "method_name": "general_impostors",
                "params": {"n_iterations": 100, "seed": 42},
                "values": {"score": 0.87},
                "provenance": provenance,
            }
        )
    )
    return d


def test_forensic_report_renders_custody_block_when_fields_populated(tmp_path: Path) -> None:
    result_dir = _make_forensic_result_dir(tmp_path, with_custody=True)
    out = tmp_path / "forensic.html"
    build_forensic_report(result_dir, output=out, title="Forensic report")
    text = out.read_text()
    # All populated fields must appear in the rendered output.
    assert "threat letter acquired 2026-04-15" in text
    assert "15 emails from suspect" in text
    assert "H1: suspect wrote it" in text
    assert "warrant A-2026-0412" in text
    assert "no modifications after acquisition" in text
    assert "abc123" in text
    assert "Chain of custody" in text
    assert "Evidentiary disclaimer" in text


def test_forensic_report_omits_custody_block_when_fields_empty(tmp_path: Path) -> None:
    """Report renders cleanly when the provenance has no forensic fields populated."""
    result_dir = _make_forensic_result_dir(tmp_path, with_custody=False)
    out = tmp_path / "forensic.html"
    build_forensic_report(result_dir, output=out, title="Forensic stub")
    text = out.read_text()
    # Disclaimer always present; chain-of-custody section skipped since no fields exist.
    assert "Evidentiary disclaimer" in text
    assert "Chain of custody" not in text
    assert "Hypotheses under test" not in text


def test_forensic_report_includes_lr_summary_when_provided(tmp_path: Path) -> None:
    result_dir = _make_forensic_result_dir(tmp_path, with_custody=True)
    out = tmp_path / "forensic.html"
    build_forensic_report(
        result_dir,
        output=out,
        title="LR report",
        lr_summaries={"general_impostors": {"log_lr": "1.34", "lr": "21.9"}},
    )
    text = out.read_text()
    assert "1.34" in text
    assert "21.9" in text
    assert "Log" in text
