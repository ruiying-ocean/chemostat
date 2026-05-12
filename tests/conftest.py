import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "model_config.toml"

sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def config_path():
    return CONFIG_PATH


@pytest.fixture
def model(config_path):
    """A loaded Chemostat at a fixed, non-trivial reference state."""
    from chemostat import Chemostat
    m = Chemostat(20.0, 2.5)
    m.load_ecoconfig(str(config_path))
    return m
