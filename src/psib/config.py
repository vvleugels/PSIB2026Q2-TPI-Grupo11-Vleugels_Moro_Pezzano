"""Project configuration loading and path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, preferring a sibling ``config.local.yaml``.

    Relative data/output paths are resolved against the selected config file.
    The returned mapping includes ``_root`` and ``_output_dir`` absolute paths.
    """
    config_path = Path(path).expanduser().resolve() if path else Path(__file__).resolve().parents[2] / "config.yaml"
    local_path = config_path.with_name("config.local.yaml")
    selected_path = local_path if local_path.is_file() else config_path
    with selected_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {selected_path}")
    root = selected_path.parent
    for key, default in (("data_dir", "data"), ("output_dir", "results")):
        value = Path(config.get(key, default)).expanduser()
        config[key] = str((value if value.is_absolute() else root / value).resolve())
    output = Path(config["output_dir"])
    (output / "figures").mkdir(parents=True, exist_ok=True)
    config["_root"] = root
    config["_output_dir"] = output
    return config
