"""Tests for `bitig cache`."""

from pathlib import Path

from typer.testing import CliRunner

from bitig.cli import app
from bitig.preprocess.cache import DocBinCache

runner = CliRunner()


def _seed(cache_dir: Path, n: int) -> None:
    (cache_dir / "docbin").mkdir(parents=True, exist_ok=True)
    cache = DocBinCache(cache_dir / "docbin")
    for i in range(n):
        cache.put(f"k{i}", b"x" * (i + 1) * 10)


def test_cache_size_reports_bytes(tmp_path: Path) -> None:
    _seed(tmp_path, 3)
    result = runner.invoke(app, ["cache", "size", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "bytes" in result.stdout.lower()


def test_cache_list_lists_keys(tmp_path: Path) -> None:
    _seed(tmp_path, 2)
    result = runner.invoke(app, ["cache", "list", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "k0" in result.stdout
    assert "k1" in result.stdout


def test_cache_clear_empties_cache(tmp_path: Path) -> None:
    _seed(tmp_path, 2)
    result = runner.invoke(app, ["cache", "clear", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    cache = DocBinCache(tmp_path / "docbin")
    assert cache.keys() == []
