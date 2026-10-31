"""Launch 2D selection and baseline segmentation with a timestamped run ID."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import yaml

from psib.config import load_config
from psib.data_io import list_cases
from scripts.run_segmentation import run_experiment
from scripts.select_slices_2d import select_2d


def run_tpi(config_path: str | Path = ROOT / "config.yaml") -> Path:
    """Use configured data, falling back to the local selected database."""
    cfg = load_config(config_path)
    configured = Path(cfg["data_dir"])
    fallback = ROOT / "data" / "Selected_database"
    if not list_cases(configured):
        if not list_cases(fallback):
            raise FileNotFoundError(
                f"No paired CT/mask cases found in configured data_dir ({configured}) "
                f"or fallback ({fallback}). Restore the data or update data_dir."
            )
        print(f"No paired cases found in {configured}; using fallback {fallback}.", flush=True)
        cfg["data_dir"] = str(fallback)
    print(f"Using data directory: {cfg['data_dir']}", flush=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = Path(cfg["_output_dir"]) / "segmentation" / run_id
    if destination.exists():
        raise FileExistsError(f"Run already exists: {destination}. Try again in the next second.")
    # Share one resolved config between both stages without editing the user's
    # YAML. Absolute data/output paths survive relocation to this temporary dir.
    effective = {key: value for key, value in cfg.items() if not key.startswith("_")}
    with tempfile.TemporaryDirectory(prefix="tpi-config-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(effective), encoding="utf-8")
        print("Selecting 2D slices...", flush=True)
        select_2d(path)
        print(f"Running baseline segmentation ({run_id})...", flush=True)
        result = run_experiment(path, run_id)
    print(f"TPI run completed. Results are in {result}.", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    try:
        run_tpi(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
