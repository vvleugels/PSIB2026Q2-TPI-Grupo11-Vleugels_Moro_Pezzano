# PSIB 2026 Q2 — Liver CT practical project

Person A's classical image-processing work for liver tumor data: dataset loading,
exploration, reference-mask-based case/slice selection, and preprocessing. Baseline 2D tumor segmentation, Dice/IoU metrics and a reproducible experiment
runner are also available. The feature API includes 2D geometric measurements
and masked GLCM texture descriptors. 3D lesion quantification, compression and
reconstruction are not implemented. See [documentation.md](documentation.md)
for the complete current workflow.

## Install

Python 3.12 is the target version. From this repository root:

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-notebooks.txt  # optional notebooks
```

### Windows (CMD)

Install 64-bit Python 3.12, then run these commands from the project folder:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip install -r requirements-notebooks.txt
```

The notebook dependencies are optional for the scripts and library. Do not copy
a virtual environment between operating systems.

## Data and configuration

The project includes a compact `data/Selected_database/` with 10 labeled CT/mask
pairs selected from the 131-case training split. It is the portable default and
is enough to run the program. For the full source dataset, place Task03 Liver
under `data/Task03_Liver/`. Supported layouts include
`imagesTr/liver_N.nii.gz` paired with `labelsTr/liver_N.nii.gz`, and LiTS-style
`volume-N.nii[.gz]` plus `segmentation-N.nii[.gz]` pairs. `imagesTs/` has no masks
in the supplied dataset and is excluded. Data stays local and is ignored by Git.
Case and slice selection uses the reference masks, so this subset is suitable
for running examples, not for claiming an unbiased prospective sample.

The subset README and `dataset.json` retain the source license and citation
details. The Medical Segmentation Decathlon describes the data as CC-BY-SA 4.0
and asks users to cite its dataset paper.

Edit portable defaults in `config.yaml`. `selection.candidate_pool_size` (15 by
default) controls the seeded pool sampled from the available labeled cases;
`selection.n_cases` (10) controls the final cases selected from that pool;
`selection.slices_per_case` is configurable from 3 to 5. `selection.seed` makes
both stages reproducible. If `config.local.yaml` exists beside it, that file
takes precedence and can hold machine-specific paths. Relative paths are
resolved from the selected config file. Generated tables and figures go to
`results/`, which is ignored by Git.

The original workflow below loads complete volumes and counts 3D components.
It is not used on the current resource-limited PC; use the 2D segmentation
workflow at the end of this README instead.

Original workflow:

```bash
python scripts/explore_dataset.py
python scripts/select_cases.py
python scripts/preprocess_selected.py
```

The first script writes `results/candidate_pool.csv`, a pool-only
`results/dataset_inventory.csv`, and three context figures. The pool CSV makes
clear that the inventory is a seeded subsample of the 131 labeled LiTS cases;
the 70 unlabeled test cases are excluded. The second script selects 10 cases
from that pool and writes `results/selected_cases_slices.csv`. It stops with an
explicit error if the pool cannot cover all three tumor-volume terciles or has
no multi-lesion cases. The third saves HU-window, denoised, and liver-cropped
figures for the selected slices. The notebooks
`00_dataset_exploration.ipynb` and `01_preprocessing.ipynb` are unexecuted
teaching/report skeletons. Selection uses reference masks and is intended for
report examples, not as an unbiased cohort.

## Array convention

`data_io.load_case` returns canonical RAS-oriented arrays in `(X, Y, Z)` order;
the axial slice at index `k` is `array[:, :, k]`. Spacing is `(sx, sy, sz)` in
millimetres along those same axes. NiBabel canonicalizes both CT and mask; shape,
canonical affine, spacing, and integer labels are checked before returning.
`get_fdata()` applies NIfTI slope/intercept scaling to CT data.

## Feature extraction (2D)

`psib.features.extract_geometric_features(mask, spacing)` returns one row per
8-connected lesion in a binary axial mask. It reports pixel and physical area,
eccentricity, solidity, maximum Feret diameter in millimetres, and centroid in
millimetres. Pass `(sx, sy)` from the case spacing; the axial slice spacing is
not used for these 2D measurements.

`psib.features.extract_texture_features(image, roi_mask)` computes contrast,
homogeneity, energy, and correlation from a symmetric gray-level co-occurrence
matrix. The image must be normalized to `[0, 1]`, such as the `denoised` output
of preprocessing. The default uses 32 gray levels, distance 1, and four in-plane
directions. A pixel pair contributes only when both pixels belong to `roi_mask`.
Call it separately for tumor (`seg == 2`) and healthy liver (`seg == 1`) when
comparing reference regions. A region with no valid neighboring pixel pairs
returns NaN texture values and a pair count of zero.

These measurements describe individual 2D slices. They do not yet calculate a
3D lesion volume or track a lesion across slices. In the current experiment,
reference masks also guide slice and region selection, so feature comparisons
are descriptive and not an independent evaluation.

## Python files and ownership

| File | Description | Owner |
| --- | --- | --- |
| `src/psib/__init__.py` | Package metadata | Shared scaffolding |
| `src/psib/config.py` | YAML config precedence, path resolution, result folders | Shared scaffolding |
| `src/psib/data_io.py` | Paired NIfTI discovery, orientation, spacing, volume loading | Person A |
| `src/psib/dataset.py` | Inventory, tumor-burden case sampling, slice selection | Person A |
| `src/psib/preprocessing.py` | HU window, denoising, enhancement, liver crop | Person A |
| `src/psib/visualization.py` | Superposición de contornos de predicción y referencia | Person C |
| `src/psib/features.py` | Geometría 2D por lesión y textura GLCM en una ROI | Person C |
| `scripts/explore_dataset.py` | Inventory CSV and context figures | Person A |
| `scripts/select_cases.py` | Selected case/slice CSV | Person A |
| `scripts/preprocess_selected.py` | Selected-slice preprocessing figures | Person A |
| `tests/test_data_io.py` | Synthetic NIfTI pairing, orientation, spacing, loading checks | Person A |
| `tests/test_dataset.py` | Candidate sampling, tumor volume, lesions, slice-gap checks | Person A |
| `tests/test_preprocessing.py` | Windowing, crop coordinates, filter shape/dtype checks | Person A |
| `tests/__init__.py` | Test discovery and source-package path setup | Person A |

## Shared interfaces

- `psib.data_io.load_case(case_id, data_dir=None)` returns `(ct_hu, seg, spacing)`.
- `psib.dataset.select_slices(seg, n_slices, minimum_gap=5)` returns slice indices.
- `psib.preprocessing.preprocess_slice(slice_hu, liver_mask, cfg)` returns
  `windowed`, `denoised`, `enhanced`, `cropped`, and `crop_box` values.

Persons B and C can rely on these signatures. The feature API is:

- `psib.features.extract_geometric_features(tumor_mask, spacing)` returns a
  table with one row per 8-connected lesion in an axial mask.
- `psib.features.extract_texture_features(image, roi_mask, ...)` returns GLCM
  properties aggregated over valid distances and directions.

Both functions operate on one 2D slice; they do not change the shared loading,
selection, or preprocessing interfaces.

## Tests

```bash
python -m unittest
```

Tests use synthetic data and do not require LiTS files.


## Segmentation experiments

Run from the project root using only 2D slice reads:

```bash
python scripts/select_slices_2d.py
python scripts/run_segmentation.py --run-id base
python scripts/run_segmentation.py --run-id comparacion --sweep
```

Selection scans masks plane by plane, without 3D components or volume-burden
stratification. All 10 local cases are included with the default configuration.
The base run uses the `segmentation` YAML block. The sweep compares 252
fixed threshold/morphology configurations using the same preprocessing. CT
and labels are read as individual planes without loading full volumes. Optional
`--config` selects a config path, retaining local-config precedence. Run IDs must
be new and contain only ASCII letters, digits, hyphens or underscores.

Outputs live under `results/segmentation/<run-id>/`: effective configuration,
selection copy, per-slice/per-case metrics, configuration summary, status and
prediction masks/figures. Sweep mode exports masks/figures for the best setting
of each method. Case metrics pool pixel counts over selected slices; summaries
weight cases equally. These are exploratory results with known liver regions,
not an independent full-volume evaluation.

See [documentation.md](documentation.md) for output schemas and failure handling.
