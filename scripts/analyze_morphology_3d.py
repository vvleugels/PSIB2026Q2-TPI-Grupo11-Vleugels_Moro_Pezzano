"""Quantify and visualize 3D tumor morphology in the selected reference masks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from psib.config import load_config
from psib.data_io import load_segmentation
from psib.morphology3d import tumor_components_3d


def export_surface(path: Path, case_id: int, segmentation: np.ndarray, spacing, components) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    tumor = segmentation == 2
    sx, sy, sz = spacing
    if not components:
        ax.text2D(0.5, 0.5, "La máscara no contiene tumor", transform=ax.transAxes, ha="center")
        ax.set_title(f"Caso {case_id:03d} · sin voxeles tumorales")
    else:
        labels, bounds, objects = components
        colors = plt.get_cmap("tab20")
        for component_id, obj in enumerate(objects, start=1):
            if obj is None:
                continue
            # Padding closes the surface at the mask boundary.
            local = labels[obj] == component_id
            padded = np.pad(local, 1)
            vertices, faces, _, _ = marching_cubes(padded.astype(np.uint8), level=0.5,
                                                    spacing=(sx, sy, sz))
            offsets = ((bounds[0][0] + obj[0].start - 1) * sx,
                       (bounds[1][0] + obj[1].start - 1) * sy,
                       (bounds[2][0] + obj[2].start - 1) * sz)
            vertices += np.asarray(offsets)
            mesh = Poly3DCollection(vertices[faces], alpha=0.85,
                                    facecolor=colors((component_id - 1) % 20),
                                    edgecolor="none")
            ax.add_collection3d(mesh)
        starts = np.asarray([axis[0] for axis in bounds]) * np.asarray(spacing)
        stops = np.asarray([axis[1] for axis in bounds]) * np.asarray(spacing)
        ax.set_xlim(starts[0], stops[0])
        ax.set_ylim(starts[1], stops[1])
        ax.set_zlim(starts[2], stops[2])
        ax.set_box_aspect(np.maximum(stops - starts, 1))
        ax.set_title(f"Caso {case_id:03d} · {len(components[2])} componentes 3D")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def analyze(config_path: Path) -> Path:
    cfg = load_config(config_path)
    output = Path(cfg["_output_dir"]) / "morphology_3d"
    output.mkdir(parents=True, exist_ok=True)
    selection_path = Path(cfg["_output_dir"]) / "selected_cases_slices.csv"
    if not selection_path.is_file():
        raise FileNotFoundError("Falta selected_cases_slices.csv. Ejecutá primero el flujo 2D para seleccionar los casos.")
    selected = pd.read_csv(selection_path)
    if "case_id" not in selected or selected.empty:
        raise ValueError("La selección está vacía o no tiene la columna case_id.")
    case_ids = sorted(pd.to_numeric(selected["case_id"], errors="raise").astype(int).unique())

    summaries, lesions = [], []
    for case_id in case_ids:
        print(f"Analizando morfología 3D del caso {case_id:03d}…", flush=True)
        segmentation = load_segmentation(case_id, cfg["data_dir"])
        spacing = _spacing_for_case(case_id, cfg["data_dir"])
        components, render_data = tumor_components_3d(segmentation, spacing)
        voxel_count = int(np.count_nonzero(segmentation == 2))
        total_volume = voxel_count * float(np.prod(spacing)) / 1000.0
        largest = components[0] if components else None
        summaries.append({
            "case_id": case_id,
            "tumor_voxels": voxel_count,
            "tumor_volume_cm3": total_volume,
            "connected_components_6": len(components),
            "largest_component_cm3": largest["volume_cm3"] if largest else 0.0,
            "largest_extent_x_mm": largest["extent_x_mm"] if largest else 0.0,
            "largest_extent_y_mm": largest["extent_y_mm"] if largest else 0.0,
            "largest_extent_z_mm": largest["extent_z_mm"] if largest else 0.0,
            "spacing_x_mm": spacing[0], "spacing_y_mm": spacing[1], "spacing_z_mm": spacing[2],
        })
        for item in components:
            lesions.append({"case_id": case_id, **item})
        export_surface(output / f"case_{case_id:03d}_surface_3d.png", case_id,
                       segmentation, spacing, render_data)
        del segmentation, render_data

    pd.DataFrame(summaries).to_csv(output / "morphology_3d_per_case.csv", index=False)
    pd.DataFrame(lesions, columns=["case_id", "component_id", "voxel_count", "volume_cm3",
                                   "extent_x_mm", "extent_y_mm", "extent_z_mm",
                                   "bbox_start_x", "bbox_start_y", "bbox_start_z"]
                 ).to_csv(output / "morphology_3d_per_lesion.csv", index=False)
    print(f"Resultados guardados en {output}", flush=True)
    return output


def _spacing_for_case(case_id: int, data_dir: Path) -> tuple[float, float, float]:
    # Read the NIfTI header only; morphology needs the mask, not CT intensities.
    import nibabel as nib
    from psib.data_io import _pairs
    volume_path, _ = _pairs(data_dir)[int(case_id)]
    image = nib.as_closest_canonical(nib.load(str(volume_path)))
    return tuple(float(value) for value in nib.affines.voxel_sizes(image.affine))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    try:
        analyze(args.config.resolve())
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error en el análisis morfológico 3D: {exc}\n")


if __name__ == "__main__":
    main()
