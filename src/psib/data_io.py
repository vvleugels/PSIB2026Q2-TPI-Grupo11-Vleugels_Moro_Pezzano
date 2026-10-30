"""NIfTI discovery and loading for paired liver CT volumes and masks."""
from __future__ import annotations

from pathlib import Path
import re
from typing import TypeAlias

import nibabel as nib
import numpy as np

Array3D: TypeAlias = np.ndarray


def _pairs(data_dir: str | Path) -> dict[int, tuple[Path, Path]]:
    """Find matched images and masks in LiTS or MSD Task03 folder layouts."""
    root = Path(data_dir).expanduser()
    candidates: list[tuple[Path, Path, str]] = []
    for image_dir, mask_dir in ((root / "imagesTr", root / "labelsTr"), (root, root)):
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue
        for image in image_dir.iterdir():
            if not image.is_file() or image.name.startswith("._"):
                continue
            name = image.name
            match = re.fullmatch(r"(?:volume-|liver_)(\d+)\.nii(?:\.gz)?", name)
            if not match:
                continue
            case_id = int(match.group(1))
            if name.startswith("volume-"):
                mask_names = (f"segmentation-{case_id}.nii.gz", f"segmentation-{case_id}.nii")
            else:
                mask_names = (name,)
            mask = next((mask_dir / n for n in mask_names if (mask_dir / n).is_file()), None)
            if mask:
                candidates.append((image, mask, str(case_id)))
    pairs: dict[int, tuple[Path, Path]] = {}
    for image, mask, raw_id in candidates:
        case_id = int(raw_id)
        if case_id in pairs:
            raise ValueError(f"Duplicate CT/mask pair for case {case_id} under {root}")
        pairs[case_id] = (image, mask)
    return pairs


def list_cases(data_dir: str | Path) -> list[int]:
    """Return sorted case IDs with both a CT volume and a segmentation."""
    return sorted(_pairs(data_dir))


def _canonical_image(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    canonical = nib.as_closest_canonical(image)
    # get_fdata applies NIfTI slope/intercept through the ArrayProxy exactly once.
    data = np.asarray(canonical.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume at {path}; got shape {data.shape}")
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(canonical.affine))
    return data, canonical.affine, spacing


def load_case(
    case_id: int, data_dir: str | Path | None = None
) -> tuple[Array3D, Array3D, tuple[float, float, float]]:
    """Load one matched case as canonical RAS ``(X, Y, Z)`` arrays.

    Slices are extracted as ``array[:, :, k]``. Both image and mask are reoriented
    to closest canonical RAS before loading, and their canonical affine, shape,
    and spacing are checked. Spacing follows array axes in millimetres. NiBabel's
    ``get_fdata`` applies stored slope/intercept scaling to CT values.
    """
    if data_dir is None:
        from .config import load_config
        data_dir = load_config()["data_dir"]
    pairs = _pairs(data_dir)
    if int(case_id) not in pairs:
        raise FileNotFoundError(f"No paired CT volume and segmentation for case {case_id} in {data_dir}")
    volume_path, mask_path = pairs[int(case_id)]
    volume, volume_affine, spacing = _canonical_image(volume_path)
    mask, mask_affine, mask_spacing = _canonical_image(mask_path)
    if volume.shape != mask.shape:
        raise ValueError(f"Case {case_id} volume/mask shape mismatch: {volume.shape} vs {mask.shape}")
    if not np.allclose(volume_affine, mask_affine, atol=1e-4):
        raise ValueError(f"Case {case_id} volume/mask canonical affine mismatch")
    if not np.allclose(spacing, mask_spacing, atol=1e-5):
        raise ValueError(f"Case {case_id} volume/mask spacing mismatch")
    rounded = np.rint(mask)
    if not np.allclose(mask, rounded, atol=1e-4):
        raise ValueError(f"Case {case_id} segmentation contains non-integer labels")
    seg = rounded.astype(np.uint8)
    labels = set(np.unique(seg).tolist())
    if not labels.issubset({0, 1, 2}):
        raise ValueError(f"Case {case_id} has unsupported labels {sorted(labels)}; expected 0, 1, 2")
    return volume, seg, spacing


def load_segmentation(case_id: int, data_dir: str | Path | None = None) -> np.ndarray:
    """Load only one canonical mask when a workflow does not need the CT array."""
    if data_dir is None:
        from .config import load_config
        data_dir = load_config()["data_dir"]
    pairs = _pairs(data_dir)
    if int(case_id) not in pairs:
        raise FileNotFoundError(f"No paired CT volume and segmentation for case {case_id} in {data_dir}")
    volume_path, mask_path = pairs[int(case_id)]
    volume_header = nib.as_closest_canonical(nib.load(str(volume_path)))
    mask_header = nib.as_closest_canonical(nib.load(str(mask_path)))
    if volume_header.shape != mask_header.shape or not np.allclose(volume_header.affine, mask_header.affine, atol=1e-4):
        raise ValueError(f"Case {case_id} volume/mask shape or canonical affine mismatch")
    mask = np.asanyarray(mask_header.dataobj)
    rounded = np.rint(mask)
    if not np.allclose(mask, rounded, atol=1e-4):
        raise ValueError(f"Case {case_id} segmentation contains non-integer labels")
    seg = rounded.astype(np.uint8)
    labels = set(np.unique(seg).tolist())
    if not labels.issubset({0, 1, 2}):
        raise ValueError(f"Case {case_id} has unsupported labels {sorted(labels)}; expected 0, 1, 2")
    return seg


def get_slice(volume: np.ndarray, k: int) -> np.ndarray:
    """Return axial slice ``k`` from a 3D ``(X, Y, Z)`` volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    if not 0 <= k < volume.shape[2]:
        raise IndexError(f"Slice index {k} outside [0, {volume.shape[2] - 1}]")
    return volume[:, :, k]


def _slice_geometry(image):
    """Canonical geometry from header only, without materializing a volume."""
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI header; got shape {image.shape}")
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")))
    order = np.argsort(transform[:, 0].astype(int))
    shape = tuple(image.shape[int(axis)] for axis in order)
    affine = image.affine @ nib.orientations.inv_ornt_aff(transform, image.shape)
    return shape, affine, transform


def _read_slice(image, transform, k):
    """Read one proxy plane and orient it to canonical X,Y using 2D operations."""
    axial_axis = int(np.flatnonzero(transform[:, 0] == 2)[0])
    source_k = k if transform[axial_axis, 1] > 0 else image.shape[axial_axis] - 1 - k
    selection = [slice(None)] * 3
    selection[axial_axis] = source_k
    plane = np.asarray(image.dataobj[tuple(selection)], dtype=np.float32)
    remaining = [axis for axis in range(3) if axis != axial_axis]
    for plane_axis, source_axis in enumerate(remaining):
        if transform[source_axis, 1] < 0:
            plane = np.flip(plane, axis=plane_axis)
    return plane.transpose(np.argsort(transform[remaining, 0])).copy()


def iter_case_slices(case_id: int, slice_indices=None, data_dir=None, *, mask_only=False):
    """Yield ``(k, ct_2d, seg_2d, spacing)`` without loading full 3D arrays.

    NIfTI headers are checked for matching canonical shape/affine. Each proxy
    read extracts one source plane, then flips/transposes that 2D plane to X,Y.
    Scaling is applied by the proxy. ``slice_indices=None`` scans axial slices
    in order; ``mask_only=True`` returns None for CT and reads only masks.
    Gzipped files may require repeated decompression, trading time for memory.
    """
    if data_dir is None:
        from .config import load_config
        data_dir = load_config()["data_dir"]
    pairs = _pairs(data_dir)
    if int(case_id) not in pairs:
        raise FileNotFoundError(f"No paired CT volume and segmentation for case {case_id} in {data_dir}")
    volume_path, mask_path = pairs[int(case_id)]
    volume = nib.load(str(volume_path), mmap=True, keep_file_open=True)
    mask = nib.load(str(mask_path), mmap=True, keep_file_open=True)
    shape, affine, volume_transform = _slice_geometry(volume)
    mask_shape, mask_affine, mask_transform = _slice_geometry(mask)
    if shape != mask_shape or not np.allclose(affine, mask_affine, atol=1e-4):
        raise ValueError(f"Case {case_id} volume/mask canonical shape or affine mismatch")
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(affine))
    indices = range(shape[2]) if slice_indices is None else slice_indices
    for k in indices:
        if not isinstance(k, (int, np.integer)) or not 0 <= k < shape[2]:
            raise IndexError(f"Case {case_id} slice index {k} outside [0, {shape[2] - 1}]")
        raw = _read_slice(mask, mask_transform, k)
        rounded = np.rint(raw)
        if (not np.isfinite(raw).all() or not np.allclose(raw, rounded, atol=1e-4)
                or not np.isin(rounded, (0, 1, 2)).all()):
            raise ValueError(f"Case {case_id}, slice {k}: expected integer labels 0, 1, 2")
        seg = rounded.astype(np.uint8)
        ct = None if mask_only else _read_slice(volume, volume_transform, k)
        yield int(k), ct, seg, spacing
