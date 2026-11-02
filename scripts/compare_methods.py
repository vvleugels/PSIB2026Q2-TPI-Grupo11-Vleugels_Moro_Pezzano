"""Paired per-case Dice comparison with exact Wilcoxon and Holm correction."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from psib.config import load_config


def resolve_choice(run_dir: Path, choice: str) -> str:
    configurations = pd.read_csv(run_dir / "configurations.csv")
    ids = set(configurations.configuration_id.astype(str))
    if choice in ids:
        return choice
    if not choice.startswith("best:") or not choice[5:]:
        raise ValueError(f"{choice!r} debe ser una cfg_xxx o best:<method>.")
    method = choice[5:]
    best_path = run_dir / "best_configurations.csv"
    candidates = pd.read_csv(best_path) if best_path.is_file() else None
    if candidates is None or not {"method", "configuration_id"}.issubset(candidates.columns):
        summary = pd.read_csv(run_dir / "summary.csv").merge(configurations, on="configuration_id")
        candidates = summary
    candidates = candidates.loc[candidates.method.astype(str).str.lower() == method.lower()]
    if candidates.empty:
        raise ValueError(f"No hay una configuración ganadora para method={method!r}.")
    return str(candidates.sort_values(["dice_mean", "iou_mean", "configuration_id"],
                                      ascending=[False, False, True]).iloc[0].configuration_id)


def compare_pair(frame: pd.DataFrame, configuration_a: str, configuration_b: str) -> dict[str, object]:
    selected = frame.loc[frame.configuration_id.astype(str).isin((configuration_a, configuration_b))]
    pivot = selected.pivot(index="case_id", columns="configuration_id", values="dice").dropna(
        subset=[configuration_a, configuration_b])
    a, b = pivot[configuration_a].to_numpy(float), pivot[configuration_b].to_numpy(float)
    differences = a - b
    nonzero = differences[differences != 0]
    if len(nonzero):
        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="exact")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    else:
        statistic, p_value = 0.0, 1.0
    median_difference = float(np.median(differences)) if len(differences) else np.nan
    direction = "A>B" if median_difference > 0 else ("A<B" if median_difference < 0 else "equal")
    return {"configuration_a": configuration_a, "configuration_b": configuration_b,
            "n_cases": len(differences), "n_nonzero_differences": len(nonzero),
            "wilcoxon_statistic": statistic, "p_value": p_value,
            "median_difference_a_minus_b": median_difference, "direction": direction,
            "warning": "n<6; low power" if len(differences) < 6 else ""}


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    valid = sorted(((i, value) for i, value in enumerate(p_values) if np.isfinite(value)), key=lambda item: item[1])
    running = 0.0
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (len(valid) - rank) * value))
        adjusted[index] = running
    return adjusted


def compare_methods(run_dir: Path, comparisons: list[tuple[str, str]]) -> Path:
    frame = pd.read_csv(run_dir / "metrics_per_case.csv")
    if not {"configuration_id", "case_id", "dice"}.issubset(frame.columns):
        raise ValueError("metrics_per_case.csv debe contener configuration_id, case_id y dice.")
    resolved = [(resolve_choice(run_dir, a), resolve_choice(run_dir, b)) for a, b in comparisons]
    rows = [compare_pair(frame, a, b) for a, b in resolved]
    for row, adjusted in zip(rows, holm_adjust([r["p_value"] for r in rows])):
        row["p_holm"] = adjusted
        row["inference_unit"] = "case"
        if row["n_cases"] == 0:
            raise ValueError(f"No hay casos emparejados para {row['configuration_a']} vs. {row['configuration_b']}.")
    output = run_dir / "method_comparisons" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "method_comparison.csv", index=False)
    print(f"Paired method comparisons saved to {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compare", action="append", required=True, metavar="A,B",
                        help="Pair to compare; repeat to include more tests. Use cfg_xxx or best:<method>.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    try:
        comparisons = []
        for value in args.compare:
            parts = [item.strip() for item in value.split(",")]
            if len(parts) != 2 or not all(parts):
                raise ValueError("Cada --compare debe tener el formato A,B.")
            comparisons.append((parts[0], parts[1]))
        cfg = load_config(args.config)
        compare_methods(Path(cfg["_output_dir"]) / "segmentation" / args.run_id, comparisons)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error en comparación de métodos: {exc}\n")


if __name__ == "__main__":
    main()
