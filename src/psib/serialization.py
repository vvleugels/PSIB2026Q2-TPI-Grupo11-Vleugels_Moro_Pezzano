"""Escritura atómica de diagnósticos JSON con valores NumPy."""
import json
from pathlib import Path
import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=_json_default) + '\n', encoding='utf-8')
    temporary.replace(path)
