from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)
