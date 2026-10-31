# PSIB 2026 Q2 — Liver CT practical project

Person A's classical image-processing work for liver tumor data: dataset loading,
exploration, reference-mask-based case/slice selection, and preprocessing. Baseline 2D tumor segmentation, Dice/IoU metrics and a reproducible experiment
runner are also available. Feature extraction, quantification, compression and
reconstruction modules are not implemented. See [documentation.md](documentation.md)
for the complete current workflow.

## Install

Python 3.12 is the target version. From this repository root:

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows (CMD)

Install 64-bit Python 3.12, then run these commands from the project folder:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Do not copy a virtual environment between operating systems.

On Windows, double-click `Run_TPI.bat` to set up or reuse `.venv` and run the
baseline workflow. Every launch installs any missing or outdated requirements
and runs `pip check` before proceeding. It uses the configured `data_dir`
(including `config.local.yaml` precedence), then falls back to
`data/Selected_database/` if no paired CT/mask cases are found. If neither
location has paired cases, it stops with an error. The fallback does not edit
your configuration files.

Run IDs use local time in `YYYYMMDD-HHMMSS` format, for example
`20261031-180409`. Results go to `<output_dir>/segmentation/<run-id>/`.
If that run already exists, wait a second and launch again.

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
figures for the selected slices. Selection uses reference masks and is intended
for report examples, not as an unbiased cohort.

Compare all three denoising filters on the first selected slice with:

```bash
python scripts/compare_filters.py
# Optional configuration file (a sibling config.local.yaml takes precedence):
python scripts/compare_filters.py --config /path/to/config.yaml
```

Run either selection script first to create `selected_cases_slices.csv`.
The comparison reads only the selected 2D slice and saves a four-panel figure
(windowed input, median, Gaussian, bilateral) under the configured output folder
as `figures/filter_comparison_case_<id>_slice_<index>.png` (zero-padded IDs).
It uses the configured HU window and each filter's parameters, with a common
`[0, 1]` display scale. Review boundary visibility and noise reduction together;
this comparison does not measure segmentation accuracy. Preprocessing figures
also show the liver crop, which uses the reference mask. Crop boxes are
`(x0, y0, x1, y1)`; add their origin to crop-local coordinates when mapping back
to the full slice.

## Array convention

`data_io.load_case` returns canonical RAS-oriented arrays in `(X, Y, Z)` order;
the axial slice at index `k` is `array[:, :, k]`. Spacing is `(sx, sy, sz)` in
millimetres along those same axes. NiBabel canonicalizes both CT and mask; shape,
canonical affine, spacing, and integer labels are checked before returning.
`get_fdata()` applies NIfTI slope/intercept scaling to CT data.

## Python files and ownership

| File | Description | Owner |
| --- | --- | --- |
| `src/psib/__init__.py` | Package metadata | Shared scaffolding |
| `src/psib/config.py` | YAML config precedence, path resolution, result folders | Shared scaffolding |
| `src/psib/data_io.py` | Paired NIfTI discovery, orientation, spacing, volume loading | Person A |
| `src/psib/dataset.py` | Inventory, tumor-burden case sampling, slice selection | Person A |
| `src/psib/preprocessing.py` | HU window, denoising, enhancement, liver crop | Person A |
| `scripts/explore_dataset.py` | Inventory CSV and context figures | Person A |
| `scripts/select_cases.py` | Selected case/slice CSV | Person A |
| `scripts/preprocess_selected.py` | Selected-slice preprocessing figures | Person A |
| `scripts/compare_filters.py` | First-selected-slice denoising comparison | Person A |
| `tests/test_data_io.py` | Synthetic NIfTI pairing, orientation, spacing, loading checks | Person A |
| `tests/test_dataset.py` | Candidate sampling, tumor volume, lesions, slice-gap checks | Person A |
| `tests/test_preprocessing.py` | Windowing, crop coordinates, filter shape/dtype checks | Person A |
| `tests/__init__.py` | Test discovery and source-package path setup | Person A |

## Shared interfaces

- `psib.data_io.load_case(case_id, data_dir=None)` returns `(ct_hu, seg, spacing)`.
- `psib.dataset.select_slices(seg, n_slices, minimum_gap=5)` returns slice indices.
- `psib.preprocessing.preprocess_slice(slice_hu, liver_mask, cfg)` returns
  `windowed`, `denoised`, `enhanced`, `cropped`, and `crop_box` values.

Persons B and C can rely on these signatures. Add segmentation, metrics, feature
extraction, and quantification in their own modules without changing these
interfaces.

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
