"""End-to-end integration: init → drop fixtures → ingest → info."""

from pathlib import Path
from shutil import copytree

import pytest
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures" / "mini_corpus"


pytestmark = [pytest.mark.integration, pytest.mark.spacy]


def test_end_to_end_init_ingest_info(tmp_path: Path) -> None:
    project = tmp_path / "demo-study"

    # 1. init
    r_init = runner.invoke(app, ["init", "demo-study", "--target", str(project)])
    assert r_init.exit_code == 0, r_init.stdout
    assert (project / "study.yaml").is_file()

    # 2. drop fixture corpus files
    copytree(FIXTURES, project / "corpus", dirs_exist_ok=True)

    # 3. ingest
    r_ingest = runner.invoke(
        app,
        [
            "ingest",
            str(project / "corpus"),
            "--metadata",
            str(project / "corpus" / "metadata.tsv"),
            "--cache-dir",
            str(project / ".tamga" / "cache"),
            "--spacy-model",
            "en_core_web_sm",
        ],
    )
    assert r_ingest.exit_code == 0, r_ingest.stdout

    # 4. cache reports 4 entries
    r_size = runner.invoke(app, ["cache", "size", "--cache-dir", str(project / ".tamga" / "cache")])
    assert r_size.exit_code == 0
    assert "4 entries" in r_size.stdout

    # 5. info runs
    r_info = runner.invoke(app, ["info"])
    assert r_info.exit_code == 0
