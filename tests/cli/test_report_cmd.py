import json
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()


def test_report_html_runs(tmp_path: Path) -> None:
    d = tmp_path / "delta"
    d.mkdir()
    (d / "result.json").write_text(
        json.dumps(
            {
                "method_name": "delta",
                "params": {},
                "values": {},
                "provenance": {
                    "bitig_version": "0.1",
                    "python_version": "3.11",
                    "spacy_model": "en",
                    "spacy_version": "3.7",
                    "corpus_hash": "x",
                    "feature_hash": None,
                    "seed": 1,
                    "timestamp": datetime.now().isoformat(),
                    "resolved_config": {},
                },
            }
        )
    )
    out = tmp_path / "r.html"
    result = runner.invoke(app, ["report", str(d), "--output", str(out), "--format", "html"])
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
