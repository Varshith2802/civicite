from pathlib import Path

import pytest

from civicite.agent.core import CiviCite
from civicite.app import build_index

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def index():
    return build_index(ROOT / "data" / "demo", out_dir=None)


@pytest.fixture
def app(index, tmp_path):
    return CiviCite(index, trace_path=tmp_path / "traces.jsonl")
