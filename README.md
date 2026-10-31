# PSIB 2026 Q2 — Liver CT practical project

Python tools for exploring labeled liver CT data, selecting axial slices,
comparing denoising filters, and running classical 2D tumor segmentation with
Dice/IoU evaluation. Reusable code lives in `src/psib/`; runnable workflows live
in `scripts/`. All workflows use Python scripts. Feature extraction includes
2D geometry and masked GLCM texture descriptors. 3D lesion quantification,
compression, and reconstruction are not implemented.

See [documentation.md](documentation.md) for the detailed Spanish guide,
API contracts, output schemas, and experimental limitations.

## Install

Python 3.12 is the target version. Run commands from the repository root.
Data must be provided locally; it is not downloaded or tracked by Git.

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

### Windows (CMD)

Install 64-bit Python 3.12 and its `py` launcher. For automatic setup and a
baseline run, double-click `Run_TPI.bat`. It creates `.venv` if needed, runs
`pip install -r requirements.txt` on every launch, and checks dependency
compatibility with `pip check`. A failed check stops the workflow. Packages
that already satisfy the requirements are retained; this does not upgrade
everything to the latest release. Native Windows execution has not been tested
in the Linux development environment.

For manual setup:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip check
```

Do not copy a virtual environment between operating systems.

## Quick start: 2D baseline

With the environment activated:

```bash
python scripts/run_tpi.py
```

This is the same Python workflow used by the Windows launcher. It:

1. Loads the configuration and looks for paired CT/mask cases in `data_dir`.
2. Falls back to the repository's `data/Selected_database/` if no pairs are found.
   If neither location has pairs, it stops with an error.
3. Regenerates `<output_dir>/selected_cases_slices.csv` using 2D mask reads.
4. Runs baseline segmentation with the configured preprocessing and
   `segmentation` settings.

Run IDs use local time in `YYYYMMDD-HHMMSS` format, such as `20261031-180409`.
Results go to `<output_dir>/segmentation/<run-id>/`. Existing runs are never
overwritten; if the timestamp already exists, wait a second and launch again.
The data fallback applies to this workflow only and does not edit your YAML.
Selection still must satisfy the configured case count and slice settings.

For a different configuration:

```bash
python scripts/run_tpi.py --config path/to/config.yaml
```

Calling the Python workflow directly assumes dependencies are already installed;
the automatic dependency checks are performed by `Run_TPI.bat`.

## Data and configuration

The default data directory is `data/Selected_database/`, intended for the
10-case subset of the 131 labeled Task03 Liver training cases. Restore or copy
this subset locally, or configure another dataset. Supported pairs are
`imagesTr/liver_N.nii.gz` with `labelsTr/liver_N.nii.gz`, or
`volume-N.nii[.gz]` with `segmentation-N.nii[.gz]`. Unlabeled `imagesTs/` cases
are excluded. Preserve the dataset's README, license, and citation information.

Edit `config.yaml` for portable defaults. A sibling `config.local.yaml`
**replaces the entire base configuration**, rather than merging with it. Copy
the complete base file before customizing local paths. Relative data/output
paths resolve from the selected config file's directory. Default output is
`results/`, ignored by Git.

The 2D workflow uses `selection.n_cases` (10), `slices_per_case` (3–5; default
5), `minimum_slice_gap` (2), and `seed` (2026). It samples cases without volume
stratification and ranks slices by tumor area, liver area, then index. It can
select fewer slices when the gap prevents the requested count. The
`candidate_pool_size` setting belongs to the optional volumetric workflow.

All selection uses reference masks. Segmentation also uses the known liver
region; tumor references are used for selection and evaluation, not as inputs
to the tumor segmentation algorithm. These results are exploratory examples,
not an independent full-volume evaluation.

## Manual selection and experiments

Use separate commands to retain control over selection and run IDs:

```bash
python scripts/select_slices_2d.py
python scripts/run_segmentation.py --run-id base
python scripts/run_segmentation.py --run-id comparacion --sweep
```

These commands read CT/masks as individual planes without loading full volumes.
`run_segmentation.py` requires an existing selection CSV and does not regenerate
it. Run IDs must be new and contain only ASCII letters, digits, hyphens, or
underscores. The sweep compares 252 fixed threshold/morphology configurations
with the same preprocessing.

Both commands accept `--config`. Use the same config for selection and
segmentation. Standalone commands use the configured `data_dir` directly;
if the automatic launcher needed its fallback, correct your config before
running standalone commands against that selection.

Each experiment saves its effective configuration, selection copy, metrics per
slice and case, summary tables, status in `run.json`, and prediction masks and
figures. Case metrics pool pixel counts over selected slices; summaries weight
cases equally. Sweep runs export masks/figures for the winning configuration
of each method. Failed runs may leave partial output; choose a new ID to retry.

## Compare denoising filters

After generating a selection CSV:

```bash
python scripts/compare_filters.py
```

Optional `--config path/to/config.yaml` uses the same configuration rules.
The script reads only the first selected slice and saves windowed input,
median, Gaussian, and bilateral views to
`<output_dir>/figures/filter_comparison_case_<id>_slice_<index>.png`, with
zero-padded identifiers. It applies the configured HU window and each filter's
parameters, regardless of which denoising method is active for segmentation.
All panels use a common `[0, 1]` display scale; no GUI opens. Missing or empty
selection CSVs produce an error. Compare noise reduction and boundary visibility;
this figure does not measure segmentation accuracy.

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

To build comparison tables for the selected slices, run:

```bash
python scripts/extract_features.py
```

The command reads `<output_dir>/selected_cases_slices.csv` and writes
`texture_per_region.csv`, `texture_summary.csv`, and `tumor_geometry.csv` under
`<output_dir>/features/`. The summary reports per-slice means and population standard
deviations; it does not perform statistical hypothesis tests.

The command accepts `--config path/to/config.yaml` and uses the configured
data directory directly, without launcher fallback. Segmentation and features
share `prepare_denoised_slice` for HU windowing and denoising.

## Optional volumetric exploration

For a machine with enough memory to load full volumes and count 3D components:

```bash
python scripts/explore_dataset.py
python scripts/select_cases.py
python scripts/preprocess_selected.py
```

Run in order. Exploration writes `candidate_pool.csv`, `dataset_inventory.csv`,
and context figures. Selection balances cases across tumor-volume terciles,
requires at least one multi-lesion case, and writes the same selection CSV
used by the 2D workflow. It fails if the candidate pool cannot meet its criteria.
Preprocessing saves HU-window, denoised, and liver-crop figures for selected
slices. These scripts use the repository configuration and do not accept
`--config`. Regenerating selection replaces the shared selection CSV.

Crop boxes use `(x0, y0, x1, y1)` with exclusive upper bounds; add the origin
to crop-local coordinates when mapping back to the full slice. Crops use the
reference liver mask. Baseline segmentation uses full-slice windowed and
denoised images, without enhancement or cropping.

## Code layout and interfaces

| Location | Purpose |
| --- | --- |
| `src/psib/config.py` | Config precedence, absolute paths, output directories |
| `src/psib/data_io.py` | Paired NIfTI discovery, volume and bounded-memory slice loading |
| `src/psib/dataset.py` | Inventory and volume-based case/slice selection |
| `src/psib/preprocessing.py` | HU windowing, denoising, enhancement, liver cropping |
| `src/psib/segmentation.py` | Global, Otsu, and statistical thresholds with morphology |
| `src/psib/metrics.py` | Dice and IoU |
| `src/psib/features.py` | 2D lesion geometry and masked GLCM texture |
| `src/psib/visualization.py` | Prediction/reference contour overlays |
| `scripts/run_tpi.py` | Automatic selection and timestamped baseline run |
| `scripts/select_slices_2d.py` | Seeded case selection and slice ranking using 2D reads |
| `scripts/run_segmentation.py` | Baseline experiments and parameter sweeps |
| `scripts/extract_features.py` | Per-region texture, summary, and tumor geometry tables |
| `scripts/compare_filters.py` | First-selected-slice denoising comparison |
| `scripts/explore_dataset.py` | Volume inventory and context figures |
| `scripts/select_cases.py` | Volume-stratified case and slice selection |
| `scripts/preprocess_selected.py` | Selected-slice preprocessing figures |
| `Run_TPI.bat` | Windows environment setup/checks and Python workflow launch |
| `tests/` | Synthetic `unittest` coverage |

Arrays use canonical RAS `(X, Y, Z)` order; axial slices are `array[:, :, k]`
with zero-based indices, and spacing is `(sx, sy, sz)` in millimetres. Figures
show `image.T` with `origin="lower"`.

- `load_case(case_id, data_dir=None)` returns `(ct_hu, seg, spacing)`.
- `iter_case_slices(...)` yields `(k, ct_2d, seg_2d, spacing)` using plane reads.
- `preprocess_slice(...)` returns `windowed`, `denoised`, `enhanced`, `cropped`,
  and `crop_box`.
- `prepare_denoised_slice(slice_hu, cfg)` returns the normalized full-slice
  image shared by segmentation and feature extraction.
- `segment_slice(image, liver_mask, settings)` returns a full-slice boolean mask.
- `plot_overlay(...)` and `save_overlay(...)` draw prediction contours in cyan
  and reference contours in orange-red. These optional helpers are not called
  by the experiment runner; its exported PNGs show three separate panels.

Scripts arrange their import paths automatically. For direct API use, add
`src/` to the Python path; the project is not currently installable with
`pip install .`.

## Tests

```bash
python -m unittest
python -m pip check
```

Tests use small synthetic datasets and do not require local CT files. They
cover loading, selection, preprocessing, segmentation, metrics, filter
comparison, 2D geometry/texture, and launcher data fallback/timestamp behavior. The synthetic
launcher test exercises the Python workflow; it does not execute Windows CMD.
