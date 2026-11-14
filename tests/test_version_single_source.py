"""The package version has one source: skskills.__version__."""

from __future__ import annotations

import json
import re
from pathlib import Path

import skskills

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_reads_the_version_from_the_package() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^dynamic = \["version"\]', text, re.M)
    assert not re.search(r'^version = "', text, re.M)
    assert 'version = {attr = "skskills.__version__"}' in text


def test_npm_package_matches_the_python_version() -> None:
    package = json.loads((ROOT / "package.json").read_text())
    assert package["version"] == skskills.__version__


def test_version_is_plain_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", skskills.__version__)



def test_root_skill_manifest_matches_the_python_version() -> None:
    text = (ROOT / "skill.yaml").read_text()
    found = re.search(r'^version: "?([^"\n]+)"?$', text, re.M)
    assert found and found.group(1) == skskills.__version__
