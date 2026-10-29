# PSIB 2026 Q2 — Liver CT practical project

Person A's classical image-processing work for liver tumor data: dataset loading,
exploration, reference-mask-based case/slice selection, and preprocessing. No
segmentation algorithm, evaluation metrics, feature extraction, quantification,
compression, or reconstruction module is implemented here.

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

Place the labeled LiTS/Medical Segmentation Decathlon Task03 Liver data under
`data/`. The supported layouts include `data/Task03_Liver/imagesTr/liver_N.nii.gz`
with paired masks in `labelsTr/`, and LiTS-style `volume-N.nii[.gz]` plus
`segmentation-N.nii[.gz]` pairs. `imagesTs/` has no masks in the supplied dataset
and is not included in the reference-mask inventory. Keep data local; it is
ignored by Git. The case and slice selection intentionally uses the reference
segmentation masks, so it is suitable for choosing report examples, not for
claiming an unbiased prospective sample.

Edit portable defaults in `config.yaml`. `selection.candidate_pool_size` (15 by
default) controls the seeded pool sampled from the 131 labeled training cases;
`selection.n_cases` (10) controls the final cases selected from that pool;
`selection.slices_per_case` is configurable from 3 to 5. `selection.seed` makes
both stages reproducible. If `config.local.yaml` exists beside it, that file
takes precedence and can hold machine-specific paths. Relative paths are
resolved from the selected config file. Generated tables and figures go to
`results/`, which is ignored by Git.

Run the workflow from the project root:

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
