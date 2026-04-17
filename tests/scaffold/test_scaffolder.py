"""Tests for project scaffolding."""

from pathlib import Path

import pytest

from tamga.scaffold import scaffold_project


def test_scaffold_creates_expected_layout(tmp_path: Path) -> None:
    target = tmp_path / "my-study"
    created = scaffold_project(name="my-study", target=target)
    assert created == target
    assert (target / "study.yaml").is_file()
    assert (target / "README.md").is_file()
    assert (target / ".gitignore").is_file()
    assert (target / "corpus").is_dir()
    assert (target / ".tamga" / "cache").is_dir()


def test_scaffold_refuses_to_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "file").write_text("dont destroy me")
    with pytest.raises(FileExistsError):
        scaffold_project(name="existing", target=target)


def test_scaffold_force_overrides_existing_directory(tmp_path: Path) -> None:
    target = tmp_path / "force"
    target.mkdir()
    (target / "preexisting.txt").write_text("hi")
    scaffold_project(name="force", target=target, force=True)
    assert (target / "study.yaml").is_file()
    assert (target / "preexisting.txt").is_file()  # force only fills gaps; never deletes user files


def test_scaffold_study_yaml_contains_project_name(tmp_path: Path) -> None:
    target = tmp_path / "named-project"
    scaffold_project(name="named-project", target=target)
    body = (target / "study.yaml").read_text()
    assert "name: named-project" in body
