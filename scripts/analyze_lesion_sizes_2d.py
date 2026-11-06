"""Per-lesion size analysis (medium 10–20 mm, large >=20 mm) for a completed 2D run."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psib.config import load_config
from psib.run_summary import compute_lesion_sizes_2d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Carpeta dentro de results/segmentation")
    parser.add_argument("--configuration-id", default="cfg_000")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
            raise ValueError("run-id solo admite letras ASCII, números, guiones y guiones bajos.")
        run_dir = Path(load_config(args.config.resolve())["_output_dir"]) / "segmentation" / args.run_id
        output = compute_lesion_sizes_2d(run_dir, args.configuration_id,
                                         progress=lambda message: print(message, flush=True))
        print(f"Resultados guardados en {output}", flush=True)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Error en el análisis por tamaño: {exc}\n")


if __name__ == "__main__":
    main()
