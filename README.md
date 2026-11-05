# PSIB 2026 Q2 — Liver CT practical project

<p><img src="assets/icon/psib_icon_256.png" width="96" alt="PSIB application icon"></p>

Python tools for exploring labeled liver CT data, selecting axial slices,
comparing denoising filters, and running classical 2D tumor segmentation with
Dice/IoU evaluation. Reusable code lives in `src/psib/`; runnable workflows live
in `scripts/`. All workflows use Python scripts. Feature extraction includes
2D geometry and masked GLCM texture descriptors. The optional volumetric
inventory summarizes 3D tumor volume and connected-component count from the
reference masks. The primary segmentation workflow remains 2D, with an optional
full-volume estimate created by stacking those same axial predictions.

See [documentation.md](documentation.md) for the detailed Spanish guide,
API contracts, output schemas, and experimental limitations.

| Capability | Scope | Main output |
| --- | --- | --- |
| Stratified slice selection | Reference masks, plane reads, physical mm spacing | `selected_cases_slices.csv` |
| Dataset context | Seeded pool of labeled cases, one volume in memory | `dataset_context_table.csv` |
| Classical segmentation | Liver-only preprocessing and 2D segmentation within a known liver mask | Dice/IoU tables and prediction figures |
| Lesion-level evaluation | 2D components, split/merge-aware matches, physical area and Feret diameter | `lesion_eval/` tables and recall plot |
| Local candidate comparison | 108 variants and three controls with one-to-one matching by lesion size | FROC, HD95, ASSD, center distance, and saved masks |
| Agreement and method statistics | Bland–Altman, Pearson/Spearman, paired Wilcoxon and Holm | Agreement plots and comparison CSVs |
| Stacked 3D prediction estimate | 2D segmenter applied to every axial slice, 6-connected cleanup | 3D volume, Dice/IoU and Feret comparison per case |
| GLCM texture | Contrast, homogeneity, energy, correlation, entropy and ASM | Feature tables, plots and level sensitivity |
| Compression comparison | 8-bit control, PNG/Huffman/RLE and JPEG-like DCT | `results/compression/<run-id>/` |
| Reconstruction comparison | Simulated acquisition of already reconstructed CT slices | `results/reconstruction/<run-id>/` |

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

To open the desktop GUI on Linux, activate the environment and run
`python3 scripts/gui.py`. If `liver_tumor.sh` is present locally, run
`./liver_tumor.sh` instead. This optional launcher activates `.venv` and is
ignored by Git. The GUI requires Tk and a graphical session.

### Windows (CMD)

Install 64-bit Python 3.12 and its `py` launcher. For automatic setup and a
GUI startup, double-click `PSIB.bat`. It creates `.venv` if needed and checks
whether each required library is installed at a compatible version. It only
runs `pip install -r requirements.txt` when a library is missing or incompatible,
then checks dependency consistency with `pip check`. Native Windows execution has not been tested
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
4. Uses the dataset liver mask to delimit the region on each selected slice.
5. Preprocesses that region and applies the configured classical segmenter,
   or passes the original HU and liver mask to the enabled HU detector.
6. Saves predictions and compares them with the tumor reference.

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
the automatic dependency checks are performed by `PSIB.bat`.

### Desktop GUI

From File Explorer, double-click `PSIB.bat`. From PowerShell at the
repository root, run:

```powershell
.\PSIB.bat
```

The launcher creates `.venv` if needed, checks installed dependencies, and only
installs or repairs them when a check fails. It opens the GUI without loading
a volume; choose a case in **Caso** when you are ready. A collapsible left sidebar holds five scrollable sections:
**Caso**, **Preprocesamiento**, **Segmentación**, **Cuantificación**, and
**Resultados**. The right side stays dedicated to the image viewer. Collapse the
sidebar with its chevron or **Ctrl+B**; in the narrow rail, click a section icon
to reopen directly there. Sidebar width, collapsed state, and active section are
saved under the ignored `.psib_gui/` profile folder.

Choose a selected case, inspect its volume information, move through axial
slices, or jump to the slice with the largest tumor area. The preprocessing and
segmentation controls update a current-slice preview; **Ejecutar flujo 2D** runs
the reproducible batch workflow with the current settings. The pinned footer
keeps the checklist, progress, cancel, and main run action visible. Results
figures and CSV tables appear in the **Resultados** list and use the same viewer.
The viewer opens in **Entrada**, which shows the image passed to the selected
segmenter. **Antes** shows the original liver intensities, and **Overlay** adds
the reference and predicted masks to **Entrada**. **Contorno hepático** toggles
the liver boundary. The status bar shows pixel coordinates, original HU and,
for classical segmentation, the normalized input value.

For classical segmentation, the order is liver mask → intensity scaling →
denoising → equalization → contrast → segmentation. The default uses a HU
window centered at 30 with width 150, Gaussian σ 0.6, CLAHE, and contrast 1.
Turning off **Aplicar ventana HU** scales the minimum and maximum within the
liver to `[0, 1]`. The preprocessed image is shown on that fixed scale. Sobel,
Prewitt, and Laplacian are optional batch comparisons. Their edge, watershed,
or sharpening outputs are separate from the main mask.

In **Segmentación → Presets del experimento**, choose `hu_sensitive`,
`default`, or `otsu_059` and click **Aplicar preset**. This fills the editable
preprocessing and segmentation fields. `hu_sensitive` activates the watershed
detector on original HU. Both Otsu presets use the preprocessed image. Use
**Aplicar al corte** or **Segmentar corte actual** to recompute the slice.

In **Caso**, set **Casos a evaluar** and **Cortes por caso (3–5)**, then use
**Aplicar y generar selección 2D**. These values also apply when the main 2D
workflow generates its selection. Cases are chosen reproducibly using the
configured seed; the case currently shown in the viewer is not forced into the
batch selection.

The GUI can also prepare `selected_cases_slices.csv` from **Caso → Aplicar y
generar selección 2D**, then launch the compression experiment and simulated
reconstruction from **Cuantificación**. Those two experiments require that
selection file; the GUI explains how to create it if it is missing. Their live
logs, cancellation, figures, and tables are handled in the same interface.
Each experiment writes to a separate timestamped directory under
`results/compression/` or `results/reconstruction/`.

The **Cuantificación** section runs the available 3D analysis on reference
masks, showing physical tumor volume, 6-connected component count, and the
largest component's RECIST-like maximum axial Feret diameter. The bounding-box
diagonal remains a separate measurement in the CSV. This reference-mask
analysis does not generate automatic 3D segmentation.

#### Application icon and desktop shortcut

The supplied 1024×1024 artwork is `assets/icon/liver_tumor_app_icon.png`.
Run `python scripts/build_icons.py` from the repository root; it recognizes that
filename and creates the canonical master copy. It deterministically
creates the PNG sizes, Windows multi-size `psib.ico`, and `assets/logo.png` used
in the sidebar. Commit the master and these generated assets so other users get
the icon without rebuilding it.

Create `PSIB.lnk` in both the project root and Desktop with:

```powershell
powershell -NoProfile -File .\scripts\create_shortcut.ps1
```

The shortcut runs the GUI launcher minimized. `PSIB.lnk` is local and ignored by
Git. On Linux, copy `scripts/psib.desktop.template`, replace its
`<project-root>` paths, and use the 256 px PNG as its `Icon` value.

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

The 2D workflow uses a seeded case sample and defaults to stratified slice roles
(largest, median, small, 75th percentile, and tumor-free). It enforces a 10 mm
gap using each scan's axial spacing and selects 5 slices per case (also accepts
3 or 4). Set `slice_strategy: area_top` to reproduce the previous tumor-area
ranking; that mode retains index-based `minimum_slice_gap`.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `selection.n_cases` | 1 | Cases sampled for 2D selection; temporarily reduced for a quick end-to-end check |
| `selection.slices_per_case` | 5 | Requested slices per case; 3–5 supported |
| `selection.slice_strategy` | `stratified` | Role-based strategy or legacy `area_top` |
| `selection.minimum_slice_gap_mm` | 10 | Minimum physical separation for stratified slices |
| `selection.min_tumor_pixels` | 20 | Minimum area for median/small/p75 tumor roles |
| `selection.seed` | 2026 | Reproducible case sample |
| `preprocessing.hu_window.enabled` | `true` | Apply the fixed HU window or scale the liver's intensity range |
| `preprocessing.enhancement.contrast` | `1.0` | Neutral at 1, lower contrast below 1, higher contrast above 1 |
| `dataset_context.density_tolerance_hu` | 5 | Absolute HU difference treated as isodense |
| `features.glcm_levels` | 16 | GLCM quantization levels; 32 was sparse on small ROIs |
| `lesion_eval.min_iou` | 0 | Minimum component IoU accepted as a lesion correspondence |
| `morphology3d_prediction.min_component_voxels` | 8 | Minimum 6-connected 3D prediction component size |
| `compression.qualities` | `[95,75,50,25,10]` | JPEG-like quantization qualities |
| `reconstruction.angles` | `[180,60,20]` | Simulated projection counts |
| `reconstruction.downsample` | 1 | Optional integer speedup factor |
| `reconstruction.sart_iterations` | 3 | SART iterations per reconstruction |
| `reconstruction.sart_relaxation` | 0.15 | SART relaxation factor |
| `reconstruction.cases` | `3` | First three selected cases; a list can name case IDs explicitly |
| `reconstruction.slices_per_case` | 3 | Role slices per case |
| `reconstruction.margin` | 8 | Liver ROI margin in pixels |

All selection uses reference masks. Segmentation also uses the known liver
region; tumor references are used for selection and evaluation, not as inputs
to the tumor segmentation algorithm. These results are exploratory examples,
not an independent full-volume evaluation.

## Scope based on the instructor's feedback

The **minimum objective** is to segment and quantify tumors in 2D on axial
slices from the configured labeled cases. The final project target is 10 cases;
the current configuration temporarily selects 1 case for a quick end-to-end
check. Restore `selection.n_cases: 10` before producing final comparative
results. It requests up to 5 slices per case, with a minimum gap between slices.
Segmentation variants are compared with Dice and IoU against reference masks;
2D geometric and GLCM features describe the selected regions. Lesion-level
evaluation adds split/merge-aware detection counts and automatic-versus-reference
area and Feret diameter. This remains exploratory evaluation on the same
selected slices, not an independent test set.

The **maximum objective** adds descriptive 3D morphology from reference masks:
tumor and liver volume, tumor burden, number of 6-connected components,
per-component 3D and maximum axial Feret diameters, equivalent-sphere diameter,
and an explicitly named bounding-box diagonal. The reported maximum diameter
is the largest axial Feret diameter; 3D Feret is secondary, and the bounding-box
diagonal is retained as a separate measurement because it overestimates
diameter. The optional maximum extension also stacks the 2D segmenter across
every slice in each selected volume, removes small 3D components, and compares
predicted volume and maximum axial Feret with the reference. It is exploratory:
it retains the 2D method and reference liver region, so it is not an independent
validation. Lesion evaluation, statistics, and texture remain 2D.

The selected slices are chosen using reference masks, and the same cases are
used for the exploratory comparison. Dice/IoU and feature summaries therefore
do not represent independent validation on unseen volumes.

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

### Dataset context, compression, and reconstruction experiments

```bash
python scripts/explore_dataset.py
python scripts/run_compression.py --run-id compression-base
python scripts/run_reconstruction.py --run-id reconstruction-base
```

The dataset inventory adds liver burden, signed tumor-to-liver HU difference,
density, and equivalent-sphere component-size counts. It writes
`dataset_context_table.csv` and `figures/delta_hu_vs_tumor_volume.png` in the
configured output directory. Inventory reads only paired labeled cases and
keeps one full volume in memory at a time.

The compression experiment evaluates every slice in
`selected_cases_slices.csv` and writes `results/compression/<run-id>/` with
per-slice, per-case, and condition-summary CSVs plus quality and example-grid
figures. PNG is checked for an exact 8-bit round trip; Huffman/RLE sizes and the
JPEG-like entropy size are estimators, and actual OpenCV JPEG sizes are also
recorded. Texture properties use reference-defined tumor and healthy-liver
regions for descriptive comparison. No tumor reference is passed to the
segmentation algorithm.

The reconstruction experiment is a **simulation**: LiTS images are already
reconstructed CTs. It selects the first three cases and the largest, small, and
tumor-free role slices by default. It writes `results/reconstruction/<run-id>/`
with per-slice, per-case, and summary metrics, a sinogram, a reconstruction grid,
and Dice versus angle count. Backprojection, ramp/Hann filtered backprojection,
and iterative SART are compared. The default run can take substantially longer
than 2D segmentation; use `reconstruction.downsample` to trade spatial detail
for speed. The requested attenuation mapping clamps HU below −1000 to zero,
so those air values cannot be recovered exactly. Case-scope segmentation also
requires the not-yet-integrated `psib.case_stats` API; the experiment stops with
a clear error rather than silently falling back to slice statistics. All
results are exploratory on reference-selected cases and slices.

## Compare denoising filters

After generating a selection CSV:

```bash
python scripts/compare_filters.py
```

Optional `--config path/to/config.yaml` uses the same configuration rules.
The script reads only the first selected slice, scales intensities within the
liver mask, and saves the input without denoising alongside bilateral, Gaussian,
and mean filtered views in
`<output_dir>/figures/filter_comparison_case_<id>_slice_<index>.png`, with
zero-padded identifiers. This comparison uses the configured window and filter
parameters and shows the outputs before equalization and contrast adjustment.
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
homogeneity, energy, correlation, entropy and ASM from a symmetric gray-level
co-occurrence matrix. The image must be normalized to `[0, 1]`, such as the
`denoised` output of preprocessing. The default uses 16 gray levels, distance 1,
and four in-plane directions. A pixel pair contributes only when both pixels
belong to `roi_mask`. GLCM energy is `sqrt(ASM)`; entropy is
`-sum(p * log2(p))` over nonzero entries. A region with no valid neighboring
pixel pairs returns NaN texture values and a pair count of zero.

These measurements describe individual 2D slices. They do not yet calculate a
3D lesion volume or track a lesion across slices. In the current experiment,
reference masks also guide slice and region selection, so feature comparisons
are descriptive and not an independent evaluation.

To build comparison tables for the selected slices, run:

```bash
python scripts/extract_features.py
python scripts/plot_features.py
python scripts/extract_features.py --levels 8 16 32
```

The command reads `<output_dir>/selected_cases_slices.csv` and writes
`texture_per_region.csv`, `texture_summary.csv`, and `tumor_geometry.csv` under
`<output_dir>/features/`. It also writes `texture_per_case.csv`, which averages
valid selected slices within each case and region, and `diagnostics.txt` for
cases with no computable tumor texture. `texture_per_region.csv` includes
`nan_reason` (`empty_roi`, `no_valid_pairs`, or empty). The second command writes
per-slice and per-case five-panel comparisons under
`<output_dir>/features/figures/`. Per-case means still describe selected slices,
not complete patient volumes. Paired exact sign tests compare five texture
properties across cases, with Holm adjustment; these exploratory results have
low power and are not independent validation. With `--levels 8 16 32`, one feature set is written per level
under `features/levels_<n>/`, and `texture_levels_sensitivity.csv` records the
median of every property by region and level. The default level changed from 32
to 16 because 32-level matrices are sparse for small lesions. The extractor
also writes per-case paired differences
(tumor minus healthy liver) to `texture_case_differences.csv` and their summary
to `texture_difference_summary.csv`. The summary includes an exact two-sided
sign test for contrast, homogeneity, energy, correlation and entropy, with
Holm-adjusted p-values across those five properties. ASM is reported but is
excluded from tests because energy is its square root.
These exploratory tests use paired case averages, exclude ties from the sign
test, and do not replace validation against independent segmentation ground
truth; interpretation is limited by the small number of selected cases.

The command accepts `--config path/to/config.yaml` and uses the configured
data directory directly, without launcher fallback. Feature extraction and
other non-equalized workflows use `prepare_denoised_slice` with the liver mask
for intensity scaling and denoising. The main classical runner adds equalization
and contrast via `prepare_enhanced_slice` before segmentation.

For lesion-level detection and size-aware recall, analyze a completed run:

```bash
python scripts/evaluate_lesions.py --run-id <run-id> --configuration-id best
python scripts/analyze_agreement.py --lesion-matches results/segmentation/<run-id>/lesion_eval/lesion_matches.csv
python scripts/compare_methods.py --run-id <run-id> --compare best:global,best:otsu
```

Lesion matching uses 8-connected 2D components. Each reference component first
maps to its highest-overlap predicted component; references mapping to the same
prediction are grouped, and remaining predictions are assigned to their best
overlapping reference group. The optional `lesion_eval.min_iou` setting defaults
to zero, meaning any positive overlap is accepted. `lesion_eval/` contains
`lesion_matches.csv`, `detection_per_case.csv`, `detection_summary.csv`, and
`recall_by_size_class.png`. Matched groups have signed relative area error,
automatic-minus-reference Feret error, and group Dice. Agreement analysis
reports Bland–Altman bias/95% limits, percent differences, and Pearson/Spearman
correlations for area and max Feret. `compare_methods.py` uses per-case Dice and
exact two-sided Wilcoxon tests; repeated `--compare A,B` pairs receive Holm
adjustment. Cases, not slices, are the comparison units; results with fewer than
six paired cases are flagged as low power.

The optional full-volume estimate processes every axial slice in each case
listed in the completed run and may take substantially longer than the
selected-slice workflow:

```bash
python scripts/evaluate_morphology_3d.py --run-id <run-id>
```

By default it uses the configuration with the best mean per-case Dice from
that run and removes 6-connected components smaller than
`morphology3d_prediction.min_component_voxels` (8 voxels). It writes per-case
and per-lesion CSVs (`morphology_3d_prediction_per_case.csv` and
`morphology_3d_prediction_per_lesion.csv`), `comparison_3d.png`, and compressed
reference/raw/clean masks in a new `morphology3d_prediction/` folder. Existing
output folders are preserved; use `--configuration-id cfg_000` to choose a
specific configuration. The baseline run completed, but its mean 3D Dice was
0.161; the current 2D threshold segmenter needs improvement before these
estimates can support strong conclusions.

## Optional volumetric exploration

For a machine with enough memory to load full volumes and count 3D components:

```bash
python scripts/explore_dataset.py
python scripts/select_cases.py
python scripts/preprocess_selected.py
```

Run in order. Exploration writes `candidate_pool.csv`, `dataset_inventory.csv`,
`dataset_context_table.csv`, and context figures. Selection balances cases across tumor-volume terciles,
requires at least one multi-lesion case, and writes the same selection CSV
used by the 2D workflow. It fails if the candidate pool cannot meet its criteria.
Slice roles use mm spacing and record shortage warnings in the CSV criteria.
Preprocessing saves scaled-input, denoised, and liver-crop figures for selected
slices. These scripts use the repository configuration and do not accept
`--config`. Regenerating selection replaces the shared selection CSV.

Crop boxes use `(x0, y0, x1, y1)` with exclusive upper bounds. Add the origin
to crop-local coordinates when mapping back to the full slice. Classical
segmentation uses an array aligned to the full slice with the values outside
the liver set to zero.

## Code layout and interfaces

| Location | Purpose |
| --- | --- |
| `src/psib/config.py` | Config precedence, absolute paths, output directories |
| `src/psib/data_io.py` | Paired NIfTI discovery, volume and bounded-memory slice loading |
| `src/psib/dataset.py` | Inventory and volume-based case/slice selection |
| `src/psib/compression.py` | Huffman/RLE estimators and block-DCT JPEG-like codec |
| `src/psib/reconstruction.py` | HU/attenuation conversion and Radon reconstruction helpers |
| `src/psib/preprocessing.py` | HU windowing, equalization, denoising, liver cropping |
| `src/psib/edge_analysis.py` | Sobel/Prewitt/Laplacian response, Laplacian sharpening, Sobel watershed refinement |
| `src/psib/segmentation.py` | Global, Otsu, and statistical thresholds with morphology |
| `src/psib/metrics.py` | Dice and IoU |
| `src/psib/features.py` | 2D lesion geometry and masked GLCM texture (entropy/ASM included) |
| `src/psib/lesion_eval.py` | Split/merge-aware 2D lesion matching and geometry errors |
| `src/psib/morphology3d.py` | Physical lesion volumes and 3D/axial Feret measurements |
| `src/psib/visualization.py` | Prediction/reference contour overlays |
| `src/psib/gui_model.py` | Display-independent sidebar state and profile persistence |
| `src/psib/app_icon.py` | Icon path discovery and safe Tk image loading |
| `scripts/run_tpi.py` | Automatic selection and timestamped baseline run |
| `scripts/select_slices_2d.py` | Seeded case selection and slice ranking using 2D reads |
| `scripts/run_segmentation.py` | Baseline experiments and parameter sweeps |
| `scripts/run_local_search.py` | Fixed 108-variant local candidate comparison with checkpointing and report regeneration |
| `scripts/evaluate_lesions.py` | Per-lesion detection metrics for a completed run |
| `scripts/evaluate_morphology_3d.py` | Full-volume estimate by stacking the 2D segmenter |
| `scripts/analyze_agreement.py` | Bland–Altman and correlation summaries |
| `scripts/compare_methods.py` | Paired per-case Wilcoxon comparisons with Holm correction |
| `scripts/extract_features.py` | Per-region texture, summary, and tumor geometry tables |
| `scripts/plot_features.py` | Descriptive GLCM boxplots by reference region |
| `scripts/compare_filters.py` | First-selected-slice denoising comparison |
| `scripts/explore_dataset.py` | Volume inventory and context figures |
| `scripts/select_cases.py` | Volume-stratified case and slice selection |
| `scripts/run_compression.py` | Lossless/lossy compression comparison on selected slices |
| `scripts/run_reconstruction.py` | Simulated acquisition and reconstruction comparison |
| `scripts/check_requirements.py` | Fast direct dependency check for the Windows launcher |
| `scripts/preprocess_selected.py` | Selected-slice preprocessing figures |
| `PSIB.bat` | Windows environment setup, dependency checks, and GUI launcher |
| `scripts/build_icons.py` | Deterministic icon asset builder from the supplied master |
| `scripts/create_shortcut.ps1` | Local minimized Windows shortcuts for Desktop and project root |
| `scripts/psib.desktop.template` | Optional Linux desktop entry template |
| `scripts/gui.py` | Desktop interface for the 2D baseline workflow |
| `tests/test_gui_model.py` | No-display tests for sidebar transitions and persistence |
| `tests/test_app_icon.py` | No-display tests for icon generation, paths and fallback |
| `tests/` | Synthetic `unittest` coverage |

Arrays use canonical RAS `(X, Y, Z)` order; axial slices are `array[:, :, k]`
with zero-based indices, and spacing is `(sx, sy, sz)` in millimetres. Figures
show `image.T` with `origin="lower"`.

- `load_case(case_id, data_dir=None)` returns `(ct_hu, seg, spacing)`.
- `iter_case_slices(...)` yields `(k, ct_2d, seg_2d, spacing)` using plane reads.
- `preprocess_slice(...)` returns `windowed` (scaled), `denoised`, `enhanced`
  (equalization and contrast), `cropped` (from `enhanced`), and `crop_box`.
- `prepare_denoised_slice(slice_hu, cfg, liver_mask=...)` returns the normalized
  and denoised full-size array. `prepare_enhanced_slice` adds equalization and
  contrast for classical segmentation. Outside the liver, the arrays are zero.
- `segment_slice(image, liver_mask, settings)` returns a full-slice boolean mask.
- `plot_overlay(...)` and `save_overlay(...)` draw prediction contours in cyan
  and reference contours in orange-red. The experiment runner instead saves a
  per-slice 2×3 composite with the selected edge-filter stages.

Scripts arrange their import paths automatically. For direct API use, add
`src/` to the Python path; the project is not currently installable with
`pip install .`.

## Tests

```bash
python -m unittest
python -m pip check
```

Tests use small synthetic datasets and do not require local CT files. They
cover loading, stratified and legacy selection, preprocessing, segmentation,
metrics, filter comparison, compression and reconstruction helpers, 2D geometry/texture, and launcher data fallback/timestamp behavior. The synthetic
launcher test exercises the Python workflow; it does not execute Windows CMD.

## Experimental HU detector (automatic, no trained model)

Opt in with `detection.enabled: true` in a complete config or apply the
`hu_sensitive` preset in the GUI.
`run_tpi.py` and `run_segmentation.py` dispatch to the same detector used in the
preview and volumetric evaluation. `detection.method` selects `hysteresis` or
`watershed`; that selected mask is saved, displayed and measured.

Analysis uses HU independently of display window/CLAHE. Mask-normalized Gaussian
smoothing uses `sigma_mm`; a 10 mm local background and the liver median supply
a darkness score normalized by `max(1.4826*MAD, noise_floor_hu)`. `high` creates
seeds and `low` permits connected growth. All connectivity is 8-neighbor in 2D.
Watershed uses the same seeds and low-threshold background on the physical HU
gradient, with a documented hysteresis fallback for missing marker classes.
`min_area_mm2` is the only removal rule; no opening, border exclusion, shape
filter or largest-component selection is applied. Inputs to the detector are
HU, a known binary liver mask, spacing and settings; never tumor references.

```bash
# Existing frozen slice selection, 48 experimental variants:
.venv/bin/python scripts/run_detection.py --run-id detector-sweep --sweep

# Freeze 10 patients / up to 5 stratified slices each, then compare 48 variants
# plus historical and coherent Otsu baselines. Outputs never overwrite a run.
.venv/bin/python scripts/run_detection.py --run-id detector-validation \
  --benchmark --historical-run results/segmentation/20261103-135329
```

The benchmark saves `selected_cases_slices.csv` and `folds.json` before scoring.
Five outer folds hold out two patients each; four inner folds partition the
remaining eight. These are fixed, non-fitted algorithms, so pooling their inner
validation predictions evaluates each parameter configuration exactly once per
training patient. No outer result participates in parameter choice.

`detection_per_slice.csv` reports one-to-one lesion matching at IoU 0.10 and
0.25, maximizing match cardinality before summed IoU. `selected_configurations.csv`
contains a choice for each outer fold and FP budget (1/2/4/8 per slice). Selection
maximizes lesion recall subject to the budget, then minimizes FP, then maximizes
case-averaged Dice, with stable ID tie-breaking. If no configuration qualifies,
the detector explicitly abstains (all reference lesions are FN), recorded in the
selection table. Budgets are enforced on inner patients; realized outer FP rates
are reported, not clipped or reselected.

`validation_summary.csv`, `validation_per_case.csv`, `outer_per_slice.csv`,
`inner_validation.csv` and `froc.png` report held-out results. Negatives contribute
FP, but not lesion-recall denominators. Empty prediction/reference masks score
Dice/IoU 1; undefined lesion precision/recall/F1 are NaN only when their count
denominator is zero. `promotion.json` checks improvement over **both** baselines,
no small-lesion recall loss and ≤8 outer FP/slice; it never edits defaults.

The historical baseline is rerun with its saved configuration on the new frozen
cohort (not compared by mixing different cohorts). The coherent Otsu control is
the prior audit's Gaussian sigma=1 pixel, no equalization, opening disk radius=1
pixel and minimum area=25 pixels; it is a fixed comparator, not an internally
tuned competitor. All detector spatial parameters use physical units.

Every variant saves full-size masks compatible with lesion evaluation and
component diagnostics (area, contrast, centroid, rejection reason), plus
per-slice effective parameters/fallback reasons. Single runs also save seeds,
candidates, score maps and comparison figures. For nested runs use an explicit
configuration ID in downstream evaluation: there is no global “best” model.
`evaluate_lesions.py` retains split/merge geometry tables and additionally exports
`one_to_one_matches.csv` and `one_to_one_per_case.csv`.

The 3D extension stacks the exact detector masks with no extra 3D component
removal. Legacy 3D reruns now use the same equalization as their saved 2D config;
existing artifacts are untouched. Reconstruction routes the original and
reconstructed HU through the detector when enabled. Compression analyzes HU
restored from its known window and 8-bit quantization; its control is therefore
the **windowed, quantized** image, not the original HU detector baseline.

This is an experimental hypodense-lesion detector inside a reference liver ROI.
Iso/hyperdense lesions remain in evaluation and are not silently excluded. Ten
patients and reference-selected slices provide a small internal comparison, not
external or full-volume validation. No claimed clinical accuracy follows from it.
## Cobertura de la referencia

La previsualización muestra **Cobertura referencia**: el porcentaje de píxeles
tumorales de referencia incluidos en la predicción, junto con el conteo cubierto/total.
Se calcula como `intersección / área de referencia`; 100 % significa que se cubrió
toda la referencia, aunque puede haber falsos positivos. Complementa Dice e IoU
y no equivale a sensibilidad por lesión. Sin referencia tumoral se muestra «—».

Las nuevas figuras y corridas 2D (clásicas y detector HU) incluyen este indicador.
En CSV, `reference_coverage` está en escala 0–1 y queda vacío si no hay referencia.
Por paciente se divide la suma de intersecciones por la suma de áreas de referencia;
`reference_coverage_mean` promedia pacientes con referencia y
`reference_coverage_n_cases` informa cuántos contribuyen. No modifica predicciones
ni resultados históricos. Reiniciar la GUI permite ver el nuevo indicador.

## Búsqueda clásica genérica y uniones automáticas

```bash
.venv/bin/python scripts/run_classical_search.py --run-id classical-search-20261104
# Para continuar una ejecución interrumpida con la misma configuración y selección:
.venv/bin/python scripts/run_classical_search.py --run-id classical-search-20261104 --resume
```

Este ejecutor separado evalúa 216 combinaciones: tres ventanas HU, cuatro
suavizados, tres ecualizaciones y seis perfiles de limpieza, siempre con Otsu.
Usa por defecto los 50 cortes y folds congelados de
`results/segmentation/classical-20261104/` (`--benchmark` permite indicar otra
carpeta con ese mismo protocolo). El detector HU y los filtros de bordes no
participan. El baseline se toma del `--config` actual, sin modificarlo.

La selección maximiza F2 por lesión (IoU ≥0,10), con desempates por sensibilidad
pequeña, menos FP/corte, Dice e ID. La unión comienza por la mejor variante
individual y agrega como máximo dos variantes si F2 mejora estrictamente. Se
recalculan los componentes y las correspondencias de cada máscara OR, sin
limpieza posterior. El procedimiento se evalúa en pacientes retenidos, junto a
la selección individual y a una alternativa de máxima sensibilidad. No hay un
presupuesto obligatorio de falsos positivos.

Los resultados se guardan en `results/classical_search/<run-id>/`: configuraciones
completas, métricas internas/externas, trazas de unión, figuras del caso 94,
comparación sensibilidad/FP y presets. Las máscaras individuales están en
`predictions/case_XXX_slice_YYY.npy`: filas en el orden de `configurations.json`,
seguidas por el default; cada fila contiene bits empaquetados con `np.packbits`.
`inputs/` contiene forma del recorte, origen, forma completa, ROI, referencia y
spacing para reconstruirlas. `case94/` incluye máscaras externas en tamaño
completo. `presets/union.yaml` describe un OR de los YAML relativos listados, sin
postprocesamiento; no es un YAML de configuración individual para la GUI.

`validation_summary.csv` estima el procedimiento con pacientes retenidos;
`exploratory_summary.csv` y `finalists.json` usan toda la cohorte y no son una
validación independiente de los presets finales. Los pacientes ya fueron
explorados: el estudio sigue siendo interno y exploratorio. Las figuras y los
conteos externos utilizan exactamente las mismas máscaras. `--resume` reutiliza
las máscaras completas y los conteos de uniones guardados; rechaza cambios en
la configuración o en la selección congelada.

## Comparación local automática y resultados recientes

Se incorporó una rama experimental sin ML con candidatos automáticos en HU,
crecimiento adaptativo, contornos activos locales y graph cuts. Compara contexto
vecino y candidatos hipodensos/hiperdensos en una grilla congelada de 108 variantes.
La máscara hepática es una entrada conocida; las etiquetas tumorales solo evalúan.

`local-v2-liver` completó las 108 variantes y los tres controles en **50 cortes
de 10 pacientes**, con Python 3.12.3 y cuatro trabajadores. Aplicó el
preprocesamiento enmascarado a los controles clásicos. La selección por Dice
retuvo `default` en los cinco folds. En la misma selección se probó Otsu con
suavizado bilateral (`sigma_color=0.5`, `sigma_spatial=10`) y contraste `0.9`.

| Configuración | Dice medio por paciente | Lesiones omitidas | Omitidas <10 mm | FP/corte |
| --- | ---: | ---: | ---: | ---: |
| `hu_sensitive` | 0,2275 | 10/83 | 8/32 | 156,54 |
| `default` | 0,3565 | 20/83 | 16/32 | 20,88 |
| `otsu_059` | 0,4417 | 21/83 | 19/32 | 10,82 |
| Otsu con bilateral | 0,3462 | 41/83 | 28/32 | 2,28 |

El emparejamiento usa IoU ≥ 0,10 en esta tabla. Frente a `otsu_059`, la variante
bilateral perdió 20 detecciones, todas menores de 30 mm. `hu_sensitive` dejó
2 de 29 cortes positivos sin una detección válida. Son resultados internos de
la cohorte explorada.

El informe del barrido está en `results/local_candidates/local-v2-liver/analysis/`.
El JSON de la corrida bilateral se llama `metrics_by_slice_and_size.json` y está
en `results/otsu-bilateral-50/segmentation/otsu-bilateral-50/`. Ambos quedan bajo
`results/`, que Git ignora. Ver el [protocolo del barrido](experiments/local_candidates/README.md)
y la [documentación de métricas](documentation.md#17-comparación-local-y-resultados-por-tamaño).

## Combinación adaptativa por tamaño

Se implementó `hybrid_size`: conserva una máscara clásica y agrega componentes
HU según su Feret **predicho** en mm, con incorporación completa o sin contacto
con la base. No utiliza la referencia tumoral para decidir. El experimento
completó 6 variantes iniciales, 144 ajustadas y 5 controles en los 50 cortes
congelados, seleccionando por paciente máxima sensibilidad con Dice medio ≥0,40.

**Ningún híbrido alcanzó ese piso en toda la cohorte.** El de mayor sensibilidad
recuperó 76/83 componentes (26/32 menores de 10 mm), con Dice 0,240 y 149,60
FP/corte; el híbrido de mayor Dice llegó a 0,327. El selector externo eligió
solo controles Otsu y obtuvo Dice 0,421, 61/83 detecciones y 15,10 FP/corte.
No cumplió el objetivo conjunto; los defaults permanecen sin cambios.

```bash
.venv/bin/python scripts/run_hybrid_search.py --preflight
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-nueva
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-nueva --resume
.venv/bin/python scripts/run_hybrid_search.py --run-id hybrid-v1 --report-only
```

El preflight verifica las corridas fuente `local-v2-liver` y `otsu-bilateral-50`.
Se reproducen los controles antes de reutilizarlos y se auditan las 7.750
máscaras guardadas. Salidas bajo `results/hybrid_candidates/<run-id>/`.
El preset fijo se admite en el ejecutor 2D; no se agregaron controles de GUI.
Ver [protocolo y comandos](experiments/hybrid_candidates/README.md) e
[informe de resultados](reports/experimento_hibrido_2026-11-05.md).
