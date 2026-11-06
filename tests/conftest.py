import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixture"


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    return FIXTURE


@pytest.fixture(scope="session")
def manifest() -> dict:
    return json.loads((FIXTURE / "manifest.json").read_text())
