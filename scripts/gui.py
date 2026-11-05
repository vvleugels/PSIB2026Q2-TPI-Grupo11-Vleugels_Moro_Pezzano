"""Small Windows-friendly graphical launcher for the 2D TPI workflow."""
from __future__ import annotations

from pathlib import Path
import os
import queue
import re
import json
import csv
import gc
import ctypes
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml

from PIL import Image, ImageTk
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psib.config import load_config
from psib.gui_model import ProfileStore, SidebarState, SECTIONS
from psib.data_io import list_cases, load_case, iter_case_slices
from psib.preprocessing import liver_bounding_box, normalize_slice_hu, prepare_enhanced_slice
from psib.detection import configured_prediction
from psib.segmentation import segment_slice
from psib.metrics import dice_score, iou_score, reference_coverage
from psib.app_icon import load_icon_images, warn_icon_once
from scipy import ndimage

DEFAULT_CONFIG = ROOT / "config.yaml"
EXPERIMENT_CONTROLS = ROOT / "experiments" / "local_candidates" / "controls.json"
# Frozen local-v2-liver comparison on 50 slices at IoU >= 0.10.
EXPERIMENT_PRESET_RESULTS = (
    ("hu_sensitive", 10, 8, "156,54"),
    ("default", 20, 16, "20,88"),
    ("otsu_059", 21, 19, "10,82"),
)
RESULT_PREFIX = "TPI run completed. Results are in "
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class TpiLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.pending_icon_warning: str | None = None
        self.icon_assets = load_icon_images(
            ROOT, lambda path: tk.PhotoImage(master=root, file=str(path)),
            warning_callback=self.queue_icon_warning,
        )
        self._icon_references: tuple[object, ...] = ()
        self._sidebar_logo = self._rail_icon = None
        if self.icon_assets is not None:
            try:
                self._icon_references = (*self.icon_assets["window"], self.icon_assets["logo"],
                                         self.icon_assets["rail"])
                self._sidebar_logo = self.icon_assets["logo"].subsample(2, 2)
                self._rail_icon = self.icon_assets["rail"]
                self._icon_references += (self._sidebar_logo,)
                root.iconphoto(True, *self.icon_assets["window"])
                if sys.platform == "win32":
                    try:
                        root.iconbitmap(str(self.icon_assets["ico"]))
                    except Exception as exc:
                        warn_icon_once(f"The Windows ICO could not be applied ({exc}); using the PNG window icons.",
                                       warning_callback=self.queue_icon_warning)
            except Exception as exc:
                self.icon_assets = None
                self._icon_references = ()
                self._sidebar_logo = self._rail_icon = None
                warn_icon_once(f"Application icon could not be applied ({exc}); continuing without a custom icon.",
                               warning_callback=self.queue_icon_warning)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.task_running = False
        self.task_buttons: list[ttk.Button] = []
        self.cancel_requested = False
        self.results_dir: Path | None = None
        self.figures: list[Path] = []
        self.figure_index = -1
        self.figure_image: Image.Image | None = None
        self.figure_photo: ImageTk.PhotoImage | None = None
        self.empty_message = ""
        self.figure_scale = 1.0
        self.figure_fit = True
        self.store = ProfileStore(ROOT / ".psib_gui")
        self.sidebar_state = self.store.load_sidebar()
        saved_sidebar = SidebarState.from_dict(self.sidebar_state.to_dict())
        self.case_volume = self.case_labels = None
        self.case_spacing = None
        self.current_hu = self.current_labels = self.current_prediction = None
        self.current_processed = None
        self.results_files: list[Path] = []
        self.analysis_case_ids: list[int] = []
        self.last_selection_signature: tuple | None = None
        self.view_mode = tk.StringVar(value="Entrada")
        self.show_reference = tk.BooleanVar(value=True)
        self.show_prediction = tk.BooleanVar(value=True)
        self.show_liver_mask = tk.BooleanVar(value=False)
        self.overlay_opacity = tk.DoubleVar(value=0.45)
        self.preprocessing_vars: dict[str, tk.Variable] = {}
        self.segmentation_vars: dict[str, tk.Variable] = {}

        root.title("PSIB TPI | Segmentación hepática")
        root.geometry("1180x760")
        root.minsize(1000, 640)
        root.configure(background="#F3F6FB")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.configure_styles()

        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self.sidebar = ttk.Frame(root, style="Panel.TFrame", width=self.sidebar_state.width)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.viewer = ttk.Frame(root, style="Viewer.TFrame")
        self.viewer.grid(row=0, column=1, sticky="nsew")
        self.build_sidebar()
        self.build_main_viewer()

        self.status_bar = ttk.Frame(root, style="StatusBar.TFrame", padding=(10, 4))
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.pixel_status = tk.StringVar(value="Píxel: —   HU: —")
        ttk.Label(self.status_bar, textvariable=self.pixel_status, style="StatusBar.TLabel").pack(side="left")
        self.status = tk.StringVar(value="Seleccioná un caso para empezar")
        self.status_pill = ttk.Frame(self.status_bar, style="StatusReady.TFrame", padding=(8, 3))
        self.status_pill.pack(side="right")
        self.status_dot = ttk.Label(self.status_pill, text="●", style="StatusReady.TLabel")
        self.status_dot.pack(side="left")
        self.status_label = ttk.Label(self.status_pill, textvariable=self.status, style="StatusReady.TLabel")
        self.status_label.pack(side="left", padx=(7, 0))
        self.log_meta = tk.StringVar(value="LISTO PARA EJECUTAR")
        self.build_log(self.sections["Resultados"])
        if self.pending_icon_warning:
            self.append_log(f"Advertencia: {self.pending_icon_warning}")
        self.bind_sidebar_shortcuts()
        self.refresh_cases()
        self.sidebar_state = saved_sidebar
        self.show_section("Caso", persist=False)
        self.apply_sidebar_layout()
        self.root.bind("<Configure>", self.on_window_resize, add="+")

    def build_sidebar(self) -> None:
        """Create persistent section pages, a compact rail, and pinned workflow controls."""
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar.rowconfigure(0, weight=1)
        self.sidebar_content = ttk.Frame(self.sidebar, style="Panel.TFrame")
        self.sidebar_content.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.sidebar_content.columnconfigure(0, weight=1)
        self.sidebar_content.rowconfigure(2, weight=1)
        header = ttk.Frame(self.sidebar_content, style="Panel.TFrame", padding=(12, 10, 8, 8))
        header.grid(row=0, column=0, sticky="ew")
        if self._sidebar_logo is not None:
            ttk.Label(header, image=self._sidebar_logo, style="Body.TLabel").pack(side="left", padx=(0, 7))
        ttk.Label(header, text="PSIB · TPI", style="SidebarTitle.TLabel").pack(side="left")
        self.collapse_button = ttk.Button(header, text="◀", width=3, style="Rail.TButton",
                                          command=self.toggle_sidebar)
        self.collapse_button.pack(side="right")

        self.nav = ttk.Frame(self.sidebar_content, style="Panel.TFrame", padding=(7, 0, 7, 7))
        self.nav.grid(row=1, column=0, sticky="ew")
        self.nav_buttons: dict[str, ttk.Button] = {}
        tab_labels = ("Caso", "Preproc.", "Segm.", "Cuant.", "Res.")
        for index, (section, glyph, label) in enumerate(zip(SECTIONS, ("▣", "◐", "◈", "▤", "▧"), tab_labels)):
            button = ttk.Button(self.nav, text=f"{glyph}\n{label}", style="Section.TButton",
                                command=lambda name=section: self.show_section(name))
            self.nav.columnconfigure(index, weight=1, uniform="section_tabs")
            button.grid(row=0, column=index, sticky="nsew", padx=1)
            self.nav_buttons[section] = button

        self.pages_host = ttk.Frame(self.sidebar_content, style="Panel.TFrame")
        self.pages_host.grid(row=2, column=0, sticky="nsew")
        self.pages_host.rowconfigure(0, weight=1)
        self.pages_host.columnconfigure(0, weight=1)
        self.sections: dict[str, ttk.Frame] = {}
        self.section_pages: dict[str, ttk.Frame] = {}
        self.section_canvases: dict[str, tk.Canvas] = {}
        for section in SECTIONS:
            page = ttk.Frame(self.pages_host, style="Panel.TFrame")
            canvas = tk.Canvas(page, background="#F3F6FB", highlightthickness=0, bd=0)
            scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
            body = ttk.Frame(canvas, style="Panel.TFrame", padding=(10, 2, 10, 12))
            body_id = canvas.create_window((0, 0), window=body, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.bind("<Configure>", lambda event, c=canvas, item=body_id: c.itemconfigure(item, width=event.width))
            body.bind("<Configure>", lambda _event, c=canvas: c.configure(scrollregion=c.bbox("all")))
            scrollbar.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_remove()
            self.section_pages[section] = page
            self.sections[section] = body
            self.section_canvases[section] = canvas

        self.build_case_section(self.sections["Caso"])
        self.build_preprocessing_section(self.sections["Preprocesamiento"])
        self.build_segmentation_section(self.sections["Segmentación"])
        self.build_quantification_section(self.sections["Cuantificación"])
        self.build_results_section(self.sections["Resultados"])

        self.footer = ttk.Frame(self.sidebar_content, style="Panel.TFrame", padding=(10, 5, 10, 10))
        self.footer.grid(row=3, column=0, sticky="ew")
        self.steps_text = tk.StringVar(value="1 Caso  ·  2 Preparar  ·  3 Segmentar  ·  4 Revisar")
        ttk.Label(self.footer, textvariable=self.steps_text, style="StepCaption.TLabel",
                  wraplength=300).pack(fill="x", pady=(0, 4))
        self.progress = ttk.Progressbar(self.footer, mode="indeterminate", style="Run.Horizontal.TProgressbar")
        self.cancel_button = ttk.Button(self.footer, text="Cancelar", style="Secondary.TButton",
                                        command=self.cancel_run, state="disabled")
        self.cancel_button.pack(side="right")
        self.run_button = ttk.Button(self.footer, text="▶  Ejecutar flujo 2D", style="Primary.TButton", command=self.run)
        self.run_button.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.sidebar_rail = ttk.Frame(self.sidebar, style="Panel.TFrame", padding=(3, 5))
        if self._rail_icon is not None:
            ttk.Label(self.sidebar_rail, image=self._rail_icon, style="Body.TLabel").pack(fill="x", pady=(0, 5))
        self.rail_toggle = ttk.Button(self.sidebar_rail, text="▶", width=4, style="Rail.TButton",
                                      command=self.toggle_sidebar)
        self.rail_toggle.pack(fill="x", pady=(0, 7))
        self.rail_activity = ttk.Label(self.sidebar_rail, text="", style="RailActivity.TLabel")
        self.rail_activity.pack(fill="x", pady=(0, 6))
        self.rail_buttons: dict[str, ttk.Button] = {}
        for section, glyph in zip(SECTIONS, ("▣", "◐", "◈", "▤", "▧")):
            button = ttk.Button(self.sidebar_rail, text=glyph, width=4, style="Rail.TButton",
                                command=lambda name=section: self.show_section(name))
            button.pack(fill="x", pady=2)
            self.rail_buttons[section] = button
        self.apply_sidebar_layout()

    def build_case_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Caso y corte", style="SectionTitle.TLabel").pack(anchor="w", pady=(4, 8))
        self.config_path = tk.StringVar(value=str(DEFAULT_CONFIG))
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Entry(row, textvariable=self.config_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3, style="Secondary.TButton", command=self.browse_config).pack(side="left", padx=(4, 0))
        ttk.Label(parent, text="Caso para previsualizar", style="Body.TLabel").pack(anchor="w", pady=(8, 2))
        self.case_var = tk.StringVar()
        self.case_combo = ttk.Combobox(parent, textvariable=self.case_var, state="readonly")
        self.case_combo.pack(fill="x")
        self.case_combo.bind("<<ComboboxSelected>>", lambda _event: self.select_case())
        ttk.Label(parent, text="Casos para el análisis (Ctrl/Shift para varios)",
                  style="Body.TLabel").pack(anchor="w", pady=(8, 2))
        analysis_case_frame = ttk.Frame(parent, style="Panel.TFrame")
        analysis_case_frame.pack(fill="x")
        self.analysis_case_list = tk.Listbox(
            analysis_case_frame, height=5, selectmode="extended", exportselection=False,
            background="#FFFFFF", foreground="#29405F", selectbackground="#D9E7FA",
            selectforeground="#20334F", highlightthickness=1, highlightbackground="#D7E0EC",
            relief="flat", activestyle="none",
        )
        analysis_case_scroll = ttk.Scrollbar(analysis_case_frame, orient="vertical",
                                             command=self.analysis_case_list.yview)
        self.analysis_case_list.configure(yscrollcommand=analysis_case_scroll.set)
        self.analysis_case_list.pack(side="left", fill="x", expand=True)
        analysis_case_scroll.pack(side="right", fill="y")
        self.analysis_case_list.bind("<<ListboxSelect>>", self.on_analysis_case_select)
        ttk.Label(parent, text="Sin casos marcados: usa la selección aleatoria con semilla.",
                  style="Body.TLabel", wraplength=280).pack(fill="x", pady=(2, 0))
        case_actions = ttk.Frame(parent, style="Panel.TFrame")
        case_actions.pack(fill="x", pady=(5, 0))
        ttk.Button(case_actions, text="Abrir volumen LiTS…", style="Secondary.TButton",
                   command=self.browse_volume).pack(fill="x")
        self.case_info = tk.StringVar(value="Elegí un caso para ver forma, espaciado y rango HU.")
        ttk.Label(parent, textvariable=self.case_info, style="Body.TLabel", wraplength=280).pack(fill="x", pady=7)
        ttk.Label(parent, text="Corte axial", style="Body.TLabel").pack(anchor="w", pady=(4, 0))
        slice_row = ttk.Frame(parent, style="Panel.TFrame")
        slice_row.pack(fill="x")
        self.slice_var = tk.IntVar(value=0)
        self.slice_scale = ttk.Scale(slice_row, from_=0, to=0, command=self.on_slice_scale)
        self.slice_scale.pack(side="left", fill="x", expand=True)
        self.slice_spin = ttk.Spinbox(slice_row, from_=0, to=0, width=6, textvariable=self.slice_var,
                                      command=self.show_case_slice)
        self.slice_spin.pack(side="left", padx=(7, 0))
        ttk.Button(parent, text="Ir al corte con mayor tumor", style="Secondary.TButton",
                   command=self.jump_largest_tumor).pack(fill="x", pady=(7, 3))
        ttk.Button(parent, text="Siguiente corte seleccionado", style="Secondary.TButton",
                   command=self.next_selected_slice).pack(fill="x")
        selection = load_config(DEFAULT_CONFIG).get("selection", {})
        self.selection_cases_var = tk.StringVar(value=str(selection.get("n_cases", 10)))
        self.selection_slices_var = tk.StringVar(value=str(selection.get("slices_per_case", 5)))
        self.selection_controls = ttk.Frame(parent, style="Panel.TFrame")
        self.selection_controls.pack(fill="x", pady=(9, 2))
        ttk.Label(self.selection_controls, text="Casos a evaluar", style="Body.TLabel").grid(
            row=0, column=0, sticky="w", pady=2)
        self.case_count_spin = ttk.Spinbox(self.selection_controls, from_=1, to=131, width=6, textvariable=self.selection_cases_var)
        self.case_count_spin.grid(row=0, column=1, sticky="e", padx=(6, 0), pady=2)
        ttk.Label(self.selection_controls, text="Cortes por caso (3–5)", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=2)
        self.slices_count_spin = ttk.Spinbox(self.selection_controls, from_=3, to=5, width=6,
                                             textvariable=self.selection_slices_var)
        self.slices_count_spin.grid(row=1, column=1, sticky="e", padx=(6, 0), pady=2)
        self.selection_controls.columnconfigure(0, weight=1)
        self.selection_button = ttk.Button(parent, text="Aplicar y generar selección 2D",
                                           style="Secondary.TButton", command=self.run_slice_selection)
        self.selection_button.pack(fill="x", pady=(3, 3))

    def build_preprocessing_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Preprocesamiento", style="SectionTitle.TLabel").pack(anchor="w", pady=(4, 8))
        ttk.Label(parent, text="Los filtros usan la máscara hepática. El detector HU recibe los HU originales.",
                  wraplength=280).pack(anchor="w", pady=(0, 6))
        cfg = load_config(DEFAULT_CONFIG)
        prep = cfg.get("preprocessing", {})
        window = prep.get("hu_window", {})
        denoise = prep.get("denoising", {})
        enhancement = prep.get("enhancement", {})
        edge_settings = cfg.get("edge_analysis", {})
        denoise_method = str(denoise.get("method", "gaussian"))
        if denoise_method == "median":
            denoise_method = "bilateral"
        parameter_keys = {"bilateral": "bilateral_sigma_color", "gaussian": "gaussian_sigma",
                          "mean": "mean_size"}
        parameter_defaults = {"bilateral": 0.08, "gaussian": 0.6, "mean": 3}
        parameter_key = parameter_keys.get(denoise_method, "bilateral_sigma_color")
        parameter_default = parameter_defaults.get(denoise_method, 0.08)
        self.preprocessing_vars = {
            "hu_window_enabled": tk.BooleanVar(value=window.get("enabled", True)),
            "center": tk.StringVar(value=str(window.get("center", 60))),
            "width": tk.StringVar(value=str(window.get("width", 400))),
            "denoising": tk.StringVar(value=denoise_method),
            "parameter": tk.StringVar(value=str(denoise.get(parameter_key, parameter_default))),
            "bilateral_sigma_spatial": tk.StringVar(
                value=str(denoise.get("bilateral_sigma_spatial", 3.0))),
            "sharpen_alpha": tk.StringVar(value=str(edge_settings.get("sharpen_alpha", 0.5))),
            "edge_operator": tk.StringVar(value=str(edge_settings.get("operator", "sobel"))),
            "equalization": tk.StringVar(value=str(enhancement.get("method", "clahe"))),
            "contrast": tk.StringVar(value=str(enhancement.get("contrast", 1.0))),
            "clahe_clip_limit": tk.StringVar(),
            "clahe_kernel_size": tk.StringVar(),
            "crop": tk.BooleanVar(value=True),
        }
        ttk.Checkbutton(parent, text="Aplicar ventana HU", variable=self.preprocessing_vars["hu_window_enabled"]).pack(anchor="w")
        self.labeled_entry(parent, "Centro de ventana (HU)", self.preprocessing_vars["center"])
        self.labeled_entry(parent, "Ancho de ventana (HU)", self.preprocessing_vars["width"])
        self.labeled_combo(parent, "Denoising", self.preprocessing_vars["denoising"],
                           ("none", "bilateral", "gaussian", "mean"))
        self.labeled_entry(parent, "Parámetro (tamaño o σ color)", self.preprocessing_vars["parameter"])
        self.labeled_entry(parent, "σ espacial bilateral", self.preprocessing_vars["bilateral_sigma_spatial"])
        self.labeled_combo(parent, "Filtro de bordes (uno a la vez)",
                           self.preprocessing_vars["edge_operator"], ("none", "sobel", "prewitt", "laplace"))
        self.labeled_entry(parent, "α realce Laplaciano (solo Laplace)", self.preprocessing_vars["sharpen_alpha"])
        self.labeled_combo(parent, "Ecualización", self.preprocessing_vars["equalization"],
                           ("none", "histogram_equalization", "clahe"))
        self.labeled_entry(parent, "Límite CLAHE (vacío: YAML)", self.preprocessing_vars["clahe_clip_limit"])
        self.labeled_entry(parent, "Ventana CLAHE (vacío: YAML)", self.preprocessing_vars["clahe_kernel_size"])
        self.labeled_entry(parent, "Contraste (1 = neutro)", self.preprocessing_vars["contrast"])
        ttk.Checkbutton(parent, text="Recortar a hígado (visualización)", variable=self.preprocessing_vars["crop"]).pack(anchor="w", pady=6)
        ttk.Button(parent, text="Aplicar al corte", style="Secondary.TButton",
                   command=self.apply_preprocessing).pack(fill="x", pady=6)

    def build_segmentation_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Segmentación 2D", style="SectionTitle.TLabel").pack(anchor="w", pady=(4, 8))
        self.build_experiment_presets(parent)
        defaults = load_config(DEFAULT_CONFIG).get("segmentation", {})
        detection = load_config(DEFAULT_CONFIG).get("detection", {})
        self.detection_enabled = tk.BooleanVar(value=detection.get("enabled", False))
        self.detection_method = tk.StringVar(value=detection.get("method", "hysteresis"))
        self.labeled_combo(parent, "Salida del detector", self.detection_method,
                           ("hysteresis", "watershed", "local_growth", "local_contour", "local_graphcut"))
        self.local_high = tk.StringVar(value=str(detection.get("high", 2.) if str(detection.get("method", "")).startswith("local_") else 2.))
        self.local_refinement = tk.StringVar(value=str(detection.get("refinement", 2.)))
        self.local_polarity = tk.StringVar(value=detection.get("polarity", "dark"))
        self.local_context = tk.BooleanVar(value=detection.get("context", True))
        self.local_config_pipeline = tk.BooleanVar(
            value=load_config(DEFAULT_CONFIG).get("local_pipeline", "raw_hu") == "configured")
        self.local_overlays = tk.BooleanVar(value=False)
        self.local_parameters = ({k: v for k, v in detection.items() if k != "enabled"}
                                 if str(detection.get("method", "")).startswith("local_") else {})
        self.labeled_entry(parent, "Semillas locales (1,5 / 2 / 2,5)", self.local_high)
        self.labeled_entry(parent, "Tolerancia crecimiento / peso contorno-grafo", self.local_refinement)
        self.labeled_combo(parent, "Candidatos locales", self.local_polarity, ("dark", "both"))
        ttk.Checkbutton(parent, text="Usar cortes vecinos (métodos locales)", variable=self.local_context).pack(anchor="w")
        ttk.Checkbutton(parent, text="Mostrar semillas y candidatos", variable=self.local_overlays,
                        command=self.redraw_slice_overlay).pack(anchor="w")
        ttk.Checkbutton(parent, text="Locales: usar preprocesamiento y morfología del config",
                        variable=self.local_config_pipeline).pack(anchor="w")
        ttk.Button(parent, text="Cargar preset local YAML", command=self.load_local_preset).pack(fill="x")
        self.detection_vars = {}
        for key, label, default in (("sigma_mm", "Suavizado (mm)", 0.6),
                                     ("high", "Umbral de semillas", 1.0),
                                     ("low", "Umbral de crecimiento", 0.5),
                                     ("min_area_mm2", "Área mínima detector (mm²)", 0.0)):
            self.detection_vars[key] = tk.StringVar(value=str(detection.get(key, default)))
            self.labeled_entry(parent, label, self.detection_vars[key])
        for key, label in (("background_sigma_mm", "σ de fondo (mm, vacío: YAML)"),
                           ("noise_floor_hu", "Piso de ruido (HU, vacío: YAML)")):
            self.detection_vars[key] = tk.StringVar()
            self.labeled_entry(parent, label, self.detection_vars[key])
        ttk.Label(parent, text="Umbralización: baseline histórico").pack(anchor="w")
        self.segmentation_vars = {
            "method": tk.StringVar(value=str(defaults.get("method", "global"))),
            "threshold": tk.StringVar(value=str(defaults.get("threshold", 0.45))),
            "k": tk.StringVar(value=str(defaults.get("k", 1.0))),
            "direction": tk.StringVar(value=str(defaults.get("direction", "lower"))),
            "opening": tk.BooleanVar(value=bool(defaults.get("opening", False))),
            "closing": tk.BooleanVar(value=bool(defaults.get("closing", False))),
            "fill_holes": tk.BooleanVar(value=bool(defaults.get("fill_holes", False))),
            "footprint": tk.StringVar(value=str(defaults.get("footprint", "disk"))),
            "radius": tk.StringVar(value=str(defaults.get("radius", 2))),
            "min_area": tk.StringVar(value=str(defaults.get("min_area", 10))),
        }
        self.labeled_combo(parent, "Método", self.segmentation_vars["method"], ("global", "otsu", "statistical"))
        self.labeled_entry(parent, "Umbral [0–1]", self.segmentation_vars["threshold"])
        self.labeled_entry(parent, "k estadístico", self.segmentation_vars["k"])
        self.labeled_combo(parent, "Dirección", self.segmentation_vars["direction"], ("lower", "upper"))
        ttk.Label(parent, text="Filtrado final: baseline y locales con config").pack(anchor="w")
        for key, title in (("opening", "Apertura morfológica"), ("closing", "Cierre morfológico"),
                           ("fill_holes", "Rellenar huecos")):
            ttk.Checkbutton(parent, text=title, variable=self.segmentation_vars[key]).pack(anchor="w")
        self.labeled_combo(parent, "Elemento morfológico", self.segmentation_vars["footprint"], ("disk", "square"))
        self.labeled_entry(parent, "Radio morfológico", self.segmentation_vars["radius"])
        self.labeled_entry(parent, "Área mínima (px)", self.segmentation_vars["min_area"])
        ttk.Separator(parent).pack(fill="x", pady=8)
        ttk.Checkbutton(parent, text="Superponer referencia", variable=self.show_reference,
                        command=self.show_case_slice).pack(anchor="w")
        ttk.Checkbutton(parent, text="Superponer automática", variable=self.show_prediction,
                        command=self.show_case_slice).pack(anchor="w")
        self.labeled_entry(parent, "Opacidad máscara (0–1)", self.overlay_opacity)
        ttk.Button(parent, text="Segmentar corte actual", style="Secondary.TButton",
                   command=self.segment_current_slice).pack(fill="x", pady=(6, 7))
        self.metrics_text = tk.StringVar(value="Cobertura referencia —\nDice —  ·  IoU —  ·  área —  ·  lesiones —")
        ttk.Label(parent, textvariable=self.metrics_text, style="Body.TLabel", wraplength=280).pack(fill="x")
        columns = ("lesion", "area", "centroid")
        self.lesion_table = ttk.Treeview(parent, columns=columns, show="headings", height=4)
        for key, title, width in (("lesion", "Lesión", 55), ("area", "px", 55), ("centroid", "Centroide", 125)):
            self.lesion_table.heading(key, text=title)
            self.lesion_table.column(key, width=width, anchor="center", stretch=True)
        self.lesion_table.pack(fill="x", pady=(5, 0))

    def build_experiment_presets(self, parent: ttk.Frame) -> None:
        self.experiment_preset = tk.StringVar(value="hu_sensitive")
        self.experiment_preset_applied = False
        self.labeled_combo(parent, "Presets del experimento", self.experiment_preset,
                           tuple(row[0] for row in EXPERIMENT_PRESET_RESULTS))
        ttk.Label(parent, text="local-v2-liver · IoU ≥ 0,10 · FN: lesiones omitidas",
                  style="Body.TLabel", wraplength=255).pack(anchor="w", pady=(4, 2))
        columns = ("preset", "fn", "fn_small", "fp_per_slice")
        table = ttk.Treeview(parent, columns=columns, show="headings", height=3, selectmode="none")
        for key, label, width in zip(columns, ("Preset", "FN", "FN <10", "FP/corte"), (94, 32, 52, 64)):
            table.heading(key, text=label)
            table.column(key, width=width, minwidth=width, anchor="center")
        for row in EXPERIMENT_PRESET_RESULTS:
            table.insert("", "end", values=row)
        table.pack(fill="x")
        ttk.Button(parent, text="Aplicar preset", style="Secondary.TButton",
                   command=self.apply_experiment_preset).pack(fill="x", pady=(4, 2))
        self.preset_status = tk.StringVar(value="Aplicá un preset para cargar sus campos editables.")
        ttk.Label(parent, textvariable=self.preset_status, style="Body.TLabel",
                  wraplength=255).pack(anchor="w")
        ttk.Separator(parent).pack(fill="x", pady=8)

    def build_quantification_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Cuantificación 3D", style="SectionTitle.TLabel").pack(anchor="w", pady=(4, 8))
        ttk.Label(parent, text="Analiza las máscaras de referencia seleccionadas: volumen, diámetro máximo y componentes 6-conexos. La segmentación automática 3D aún no está implementada.",
                  style="Body.TLabel", wraplength=280).pack(fill="x", pady=(0, 8))
        controls = ttk.Frame(parent, style="Panel.TFrame")
        controls.pack(fill="x")
        self.morphology_button = ttk.Button(controls, text="Analizar 3D", style="Secondary.TButton",
                                            command=self.run_morphology)
        self.morphology_button.pack(side="left", fill="x", expand=True)
        ttk.Button(controls, text="Cancelar", style="Secondary.TButton", command=self.cancel_run).pack(side="left", padx=(5, 0))
        self.compression_button = ttk.Button(parent, text="Ejecutar experimento de compresión",
                                             style="Secondary.TButton", command=self.run_compression)
        self.compression_button.pack(fill="x", pady=(4, 2))
        self.reconstruction_button = ttk.Button(parent, text="Ejecutar reconstrucción simulada",
                                                style="Secondary.TButton", command=self.run_reconstruction)
        self.reconstruction_button.pack(fill="x", pady=2)
        self.task_buttons = [self.selection_button, self.morphology_button,
                             self.compression_button, self.reconstruction_button]
        columns = ("case_id", "lesions", "volume", "diameter")
        self.morphology_table = ttk.Treeview(parent, columns=columns, show="headings", height=6)
        for key, title, width in (("case_id", "Caso", 50), ("lesions", "Lesiones", 60),
                                  ("volume", "Vol. cm³", 75), ("diameter", "Diám. axial mm", 85)):
            self.morphology_table.heading(key, text=title)
            self.morphology_table.column(key, width=width, anchor="center", stretch=True)
        self.morphology_table.pack(fill="x", pady=(8, 4))
        ttk.Button(parent, text="Abrir resultados 3D", style="Secondary.TButton",
                   command=self.open_morphology_results).pack(fill="x")

    def build_results_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Resultados guardados", style="SectionTitle.TLabel").pack(anchor="w", pady=(4, 8))
        columns = ("kind", "name")
        self.results_tree = ttk.Treeview(parent, columns=columns, show="headings", height=6)
        self.results_tree.heading("kind", text="Tipo")
        self.results_tree.heading("name", text="Archivo")
        self.results_tree.column("kind", width=64, anchor="center", stretch=False)
        self.results_tree.column("name", width=190, anchor="w")
        self.results_tree.pack(fill="x")
        self.results_tree.bind("<<TreeviewSelect>>", self.select_result_item)
        self.note_title = tk.StringVar(value="Descripción")
        self.note_body = tk.StringVar(value="Seleccioná una figura o tabla para verla en el visor.")
        ttk.Label(parent, textvariable=self.note_title, style="SectionTitle.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Label(parent, textvariable=self.note_body, style="Body.TLabel", wraplength=280).pack(fill="x")
        ttk.Button(parent, text="Abrir carpeta de resultados", style="Secondary.TButton",
                   command=self.open_results).pack(fill="x", pady=(7, 4))
        ttk.Button(parent, text="Elegir carpeta…", style="Secondary.TButton",
                   command=self.browse_results_path).pack(fill="x")
        ttk.Label(parent, text="Registro", style="Body.TLabel").pack(anchor="w", pady=(6, 2))

    def labeled_entry(self, parent, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").pack(anchor="w", pady=(5, 1))
        ttk.Entry(parent, textvariable=variable).pack(fill="x")

    def labeled_combo(self, parent, label: str, variable: tk.StringVar, values) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").pack(anchor="w", pady=(5, 1))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly").pack(fill="x")

    def bind_sidebar_shortcuts(self) -> None:
        self.root.bind("<Control-b>", lambda _event: self.toggle_sidebar())
        self.root.bind("<Control-B>", lambda _event: self.toggle_sidebar())
        self.root.bind_all("<MouseWheel>", self.sidebar_wheel_current, add="+")

    def sidebar_wheel_current(self, event: tk.Event) -> str:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget is not None and str(widget).startswith(str(self.sidebar)):
            self.section_canvases[self.sidebar_state.active_section].yview_scroll(
                -1 if event.delta > 0 else 1, "units")
            return "break"
        return ""

    def show_section(self, section: str, persist: bool = True) -> None:
        was_collapsed = self.sidebar_state.collapsed
        self.sidebar_state.select_section(section)
        if was_collapsed:
            self.apply_sidebar_layout()
        for page in self.sections:
            widget = self.section_pages[page]
            if page == section:
                widget.grid()
            else:
                widget.grid_remove()
        for name, button in self.nav_buttons.items():
            button.state(["selected"] if name == section else ["!selected"])
        for name, button in self.rail_buttons.items():
            button.state(["selected"] if name == section else ["!selected"])
        if persist:
            self.store.save_sidebar(self.sidebar_state)

    def toggle_sidebar(self) -> None:
        center = self.capture_view_center()
        self.sidebar_state.toggle()
        self.apply_sidebar_layout()
        self.root.after_idle(lambda: self.restore_view(center))
        self.store.save_sidebar(self.sidebar_state)

    def apply_sidebar_layout(self) -> None:
        collapsed = self.sidebar_state.collapsed
        self.sidebar_state.set_width(self.sidebar_state.width, self.root.winfo_width() or 1180)
        self.sidebar.configure(width=44 if collapsed else self.sidebar_state.width)
        if collapsed:
            self.sidebar_content.grid_remove()
            self.sidebar_rail.grid(row=0, column=0, sticky="nsew")
            self.collapse_button.configure(text="▶")
            self.rail_activity.configure(text="●" if self.process is not None else "")
        else:
            self.sidebar_rail.grid_remove()
            self.sidebar_content.grid(row=0, column=0, sticky="nsew")
            self.collapse_button.configure(text="◀")
        self.root.after_idle(self.render_figure)

    def on_window_resize(self, event: tk.Event) -> None:
        if event.widget is self.root:
            previous = self.sidebar_state.width
            clamped = self.sidebar_state.set_width(previous, event.width)
            if not self.sidebar_state.collapsed and clamped != previous:
                self.sidebar.configure(width=clamped)
                self.store.save_sidebar(self.sidebar_state)

    def capture_view_center(self) -> tuple[float, float]:
        if self.figure_image is None:
            return (0.5, 0.5)
        display_width = max(1, round(self.figure_image.width * self.figure_scale))
        display_height = max(1, round(self.figure_image.height * self.figure_scale))
        canvas_width, canvas_height = max(self.figure_canvas.winfo_width(), 1), max(self.figure_canvas.winfo_height(), 1)
        region_width, region_height = max(display_width, canvas_width), max(display_height, canvas_height)
        left, top = (region_width - display_width) / 2, (region_height - display_height) / 2
        x = (self.figure_canvas.canvasx(canvas_width / 2) - left) / display_width
        y = (self.figure_canvas.canvasy(canvas_height / 2) - top) / display_height
        return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))

    def restore_view(self, center: tuple[float, float]) -> None:
        if self.figure_fit:
            self.fit_figure()
        elif self.figure_image is not None:
            self.render_figure()
            width, height = max(self.figure_canvas.winfo_width(), 1), max(self.figure_canvas.winfo_height(), 1)
            image_width = max(1, round(self.figure_image.width * self.figure_scale))
            image_height = max(1, round(self.figure_image.height * self.figure_scale))
            region_width, region_height = max(image_width, width), max(image_height, height)
            left = (region_width - image_width) / 2 + center[0] * image_width - width / 2
            top = (region_height - image_height) / 2 + center[1] * image_height - height / 2
            self.figure_canvas.xview_moveto(max(0.0, left / region_width))
            self.figure_canvas.yview_moveto(max(0.0, top / region_height))

    def refresh_cases(self) -> None:
        try:
            config = load_config(self.config_path.get())
            cases = list_cases(self.external_data_dir if getattr(self, "external_data_dir", None) else config["data_dir"])
            available_case_count = len(cases)
            if hasattr(self, "case_count_spin"):
                self.case_count_spin.configure(to=max(1, available_case_count))
                try:
                    requested = int(self.selection_cases_var.get())
                except ValueError:
                    requested = 1
                if requested > available_case_count > 0:
                    self.selection_cases_var.set(str(available_case_count))
        except Exception as exc:
            cases = []
            self.case_info.set(f"No se pudo leer la base: {exc}")
        self.case_combo["values"] = [f"{case:03d}" for case in cases]
        if hasattr(self, "analysis_case_list"):
            available_ids = set(cases)
            self.analysis_case_ids = [case_id for case_id in self.analysis_case_ids
                                      if case_id in available_ids]
            self.analysis_case_list.delete(0, tk.END)
            for index, case_id in enumerate(cases):
                self.analysis_case_list.insert(tk.END, f"{case_id:03d}")
                if case_id in self.analysis_case_ids:
                    self.analysis_case_list.selection_set(index)
            if self.analysis_case_ids:
                self.selection_cases_var.set(str(len(self.analysis_case_ids)))
                self.case_count_spin.configure(state="disabled")
            else:
                self.case_count_spin.configure(state="normal")
        if not self.case_var.get():
            self.case_info.set(
                f"{len(cases)} casos disponibles. Elegí un caso para cargar su volumen y etiquetas."
                if cases else "No se encontraron casos disponibles."
            )

    def on_analysis_case_select(self, _event=None) -> None:
        """Use the explicitly highlighted cases instead of the seeded sample."""
        self.analysis_case_ids = [
            int(self.analysis_case_list.get(index))
            for index in self.analysis_case_list.curselection()
        ]
        if self.analysis_case_ids:
            self.selection_cases_var.set(str(len(self.analysis_case_ids)))
            self.case_count_spin.configure(state="disabled")
        else:
            self.case_count_spin.configure(state="normal")

    def current_selection_signature(self) -> tuple:
        """Describe the data and case/slice choices used to create a selection."""
        config = load_config(self.config_path.get())
        case_ids = tuple(self.analysis_case_ids)
        n_cases = len(case_ids) if case_ids else int(self.selection_cases_var.get())
        seed = int(config.get("selection", {}).get("seed", 2026))
        data_dir = getattr(self, "external_data_dir", None) or config["data_dir"]
        return (str(Path(data_dir).resolve()), case_ids, n_cases,
                int(self.selection_slices_var.get()), seed)

    def browse_volume(self) -> None:
        selected = filedialog.askopenfilename(title="Elegir volumen LiTS",
            filetypes=(("NIfTI", "*.nii *.nii.gz"), ("Todos", "*.*")))
        if not selected:
            return
        path = Path(selected)
        parent = path.parent.parent if path.parent.name.lower() == "imagestr" else path.parent
        self.external_data_dir = parent
        try:
            import re
            match = re.search(r"(?:volume-|liver_)(\d+)", path.name)
            if not match:
                raise ValueError("Nombre de volumen no reconocido (se espera volume-N o liver_N).")
            case = int(match.group(1))
            self.refresh_cases()
            self.case_var.set(f"{case:03d}")
            self.select_case()
        except Exception as exc:
            messagebox.showerror("No se pudo abrir el caso", str(exc))

    def select_case(self) -> None:
        value = self.case_var.get().strip()
        if not value:
            return
        try:
            config = load_config(self.config_path.get())
            data_dir = self.external_data_dir if getattr(self, "external_data_dir", None) else config["data_dir"]
            self.case_volume = self.case_labels = self.current_hu = self.current_labels = self.current_prediction = None
            self.current_processed = None
            gc.collect()
            self.case_volume, self.case_labels, self.case_spacing = load_case(int(value), data_dir)
            hu_min, hu_max = float(np.nanmin(self.case_volume)), float(np.nanmax(self.case_volume))
            self.case_info.set(f"Forma: {self.case_volume.shape}\nEspaciado: {self.case_spacing} mm\nHU: {hu_min:.1f} … {hu_max:.1f}")
            maximum = self.case_volume.shape[2] - 1
            self.slice_scale.configure(to=maximum)
            self.slice_spin.configure(to=maximum)
            self.slice_var.set(maximum // 2)
            self.show_case_slice()
        except Exception as exc:
            self.case_volume = self.case_labels = None
            self.case_info.set(f"No se pudo cargar el caso {value}: {exc}")

    def on_slice_scale(self, value: str) -> None:
        self.slice_var.set(int(float(value)))
        self.show_case_slice()

    def selected_case_id(self) -> int | None:
        try:
            return int(self.case_var.get())
        except ValueError:
            return None

    def jump_largest_tumor(self) -> None:
        if self.case_labels is not None:
            areas = np.count_nonzero(self.case_labels == 2, axis=(0, 1))
            self.slice_var.set(int(np.argmax(areas)))
            self.show_case_slice()

    def next_selected_slice(self) -> None:
        case_id = self.selected_case_id()
        if case_id is None:
            return
        try:
            config = load_config(self.config_path.get())
            path = Path(config["_output_dir"]) / "selected_cases_slices.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                rows = [int(row["slice_index"]) for row in csv.DictReader(stream)
                        if int(row["case_id"]) == case_id]
            later = [item for item in rows if item > int(self.slice_var.get())]
            self.slice_var.set(later[0] if later else (rows[0] if rows else int(self.slice_var.get())))
            self.show_case_slice()
        except (OSError, KeyError, ValueError) as exc:
            self.append_log(f"No se pudo avanzar al corte seleccionado: {exc}")

    def current_settings(self) -> tuple[dict, dict]:
        config = load_config(self.config_path.get())
        selection = config.setdefault("selection", {})
        n_cases = int(self.selection_cases_var.get())
        slices_per_case = int(self.selection_slices_var.get())
        if n_cases < 1:
            raise ValueError("La cantidad de casos debe ser al menos 1.")
        if not 3 <= slices_per_case <= 5:
            raise ValueError("La cantidad de cortes por caso debe estar entre 3 y 5.")
        selection["n_cases"] = n_cases
        selection["slices_per_case"] = slices_per_case
        if self.analysis_case_ids:
            selection["case_ids"] = list(self.analysis_case_ids)
            selection["n_cases"] = len(self.analysis_case_ids)
        else:
            selection.pop("case_ids", None)
        prep = config.setdefault("preprocessing", {})
        window = prep.setdefault("hu_window", {})
        window["enabled"] = self.preprocessing_vars["hu_window_enabled"].get()
        if window["enabled"]:
            window.update(center=float(self.preprocessing_vars["center"].get()),
                          width=float(self.preprocessing_vars["width"].get()))
        method = self.preprocessing_vars["denoising"].get()
        denoise = {"method": method}
        if method != "none":
            parameter = float(self.preprocessing_vars["parameter"].get())
            parameter_key = {"bilateral": "bilateral_sigma_color", "gaussian": "gaussian_sigma",
                             "mean": "mean_size"}[method]
            if method == "mean":
                if not parameter.is_integer():
                    raise ValueError("El tamaño del filtro de media debe ser un número entero.")
                parameter = int(parameter)
            denoise[parameter_key] = parameter
        if method == "bilateral":
            denoise["bilateral_sigma_spatial"] = float(
                self.preprocessing_vars["bilateral_sigma_spatial"].get())
        prep["denoising"] = denoise
        enhancement = prep.setdefault("enhancement", {})
        enhancement["method"] = self.preprocessing_vars["equalization"].get()
        enhancement["contrast"] = float(self.preprocessing_vars["contrast"].get())
        for key, convert in (("clahe_clip_limit", float), ("clahe_kernel_size", int)):
            value = self.preprocessing_vars[key].get().strip()
            if value:
                enhancement[key] = convert(value)
        config.setdefault("edge_analysis", {})["sharpen_alpha"] = float(
            self.preprocessing_vars["sharpen_alpha"].get())
        config.setdefault("edge_analysis", {})["operator"] = self.preprocessing_vars["edge_operator"].get()
        segment = {key: var.get() for key, var in self.segmentation_vars.items()}
        segment["threshold"] = float(segment["threshold"])
        segment["k"] = float(segment["k"])
        segment["radius"] = int(segment["radius"])
        segment["min_area"] = int(segment["min_area"])
        if self.experiment_preset_applied:
            config["detection"] = {}
        config["local_pipeline"] = "configured" if self.local_config_pipeline.get() else "raw_hu"
        detector = config.setdefault("detection", {})
        detector.update(enabled=self.detection_enabled.get(), method=self.detection_method.get())
        if not detector["enabled"]:
            return config, segment
        if detector["method"].startswith("local_"):
            from psib.local_segmentation import validate_settings
            parameters = validate_settings({**self.local_parameters, "method": detector["method"],
                "high": float(self.local_high.get()), "refinement": float(self.local_refinement.get()),
                "polarity": self.local_polarity.get(), "context": self.local_context.get()})
            config["detection"] = dict(enabled=True, **parameters)
        else:
            for key, variable in self.detection_vars.items():
                value = variable.get().strip()
                if value or key not in ("background_sigma_mm", "noise_floor_hu"):
                    detector[key] = float(value)
        return config, segment

    def apply_experiment_preset(self) -> None:
        name = self.experiment_preset.get()
        try:
            controls = json.loads(EXPERIMENT_CONTROLS.read_text(encoding="utf-8"))
            baseline, preset = controls["default"], controls[name]
            base_prep, prep = baseline["preprocessing"], preset.get("preprocessing", {})
            window = {**base_prep["hu_window"], **prep.get("hu_window", {})}
            denoising = {**base_prep["denoising"], **prep.get("denoising", {})}
            enhancement = {**base_prep["enhancement"], **prep.get("enhancement", {})}
            edges = {**baseline["edge_analysis"], **preset.get("edge_analysis", {})}
            segmentation = {**baseline["segmentation"], **preset.get("segmentation", {})}
            detection = {**baseline["detection"], **preset.get("detection", {})}
            preprocessing = dict(center=window["center"], width=window["width"], denoising=denoising["method"],
                                 parameter=denoising["gaussian_sigma"], equalization=enhancement["method"],
                                 contrast=enhancement.get("contrast", 1.0),
                                 clahe_clip_limit=enhancement["clahe_clip_limit"],
                                 clahe_kernel_size=enhancement["clahe_kernel_size"],
                                 edge_operator=edges["operator"], sharpen_alpha=edges["sharpen_alpha"])
            updates = [(self.preprocessing_vars[key], str(value)) for key, value in preprocessing.items()]
            updates.append((self.preprocessing_vars["hu_window_enabled"], window.get("enabled", True)))
            updates.extend((variable, segmentation[key]) for key, variable in self.segmentation_vars.items())
            updates.extend((variable, str(detection[key])) for key, variable in self.detection_vars.items())
            updates.extend(((self.detection_enabled, detection["enabled"]), (self.detection_method, detection["method"])))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            messagebox.showerror("Preset del experimento inválido", str(exc))
            return
        for variable, value in updates:
            variable.set(value)
        self.experiment_preset_applied = True
        self.preset_status.set(f"{name} aplicado. Podés editar cualquier campo.")
        self.append_log(f"Preset {name} aplicado a Preprocesamiento y Segmentación.")

    def load_local_preset(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml")])
        if not path:
            return
        try:
            from psib.local_segmentation import validate_settings
            preset = yaml.safe_load(Path(path).read_text())
            values = preset["detection"]
            settings = validate_settings({k: v for k, v in values.items() if k != "enabled"})
            self.local_parameters = settings
            self.local_config_pipeline.set(preset.get("local_pipeline", "raw_hu") == "configured")
            self.detection_enabled.set(True)
            self.detection_method.set(settings["method"])
            self.local_high.set(str(settings["high"]))
            self.local_refinement.set(str(settings["refinement"]))
            self.local_polarity.set(settings["polarity"])
            self.local_context.set(settings["context"])
            self.append_log("Preset local cargado; aplicar al corte para calcularlo.")
        except Exception as exc:
            messagebox.showerror("Preset local inválido", str(exc))

    def redraw_slice_overlay(self) -> None:
        if self.current_hu is not None:
            source = self.current_hu if self.view_mode.get() == "Antes" or self.current_processed is None else self.current_processed
            self.draw_case_image(source)

    def start_local_preview(self, config, settings, k, token):
        from psib.local_segmentation import context_from_volume
        if not hasattr(self, "local_executor"):
            self.local_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-preview")
        if getattr(self, "local_future", None) is not None:
            self.local_future.cancel()
        volume, labels, spacing = self.case_volume, self.case_labels, self.case_spacing
        def compute():
            liver = labels > 0
            context = context_from_volume(volume, liver, k, spacing) if config["detection"]["context"] else None
            return configured_prediction(volume[:, :, k], liver[:, :, k], spacing, config, settings, context=context)
        future = self.local_executor.submit(compute)
        self.local_future = future
        self.metrics_text.set("Calculando segmentación local experimental…")
        self.root.after(50, lambda: self.finish_local_preview(token, future))

    def finish_local_preview(self, token, future):
        if token != getattr(self, "local_preview_token", None):
            return
        if not future.done():
            self.root.after(50, lambda: self.finish_local_preview(token, future))
            return
        try:
            result = future.result()
            self.current_local_result = result
            self.current_processed = result.get("processed_image")
            self.current_prediction = result["mask"]
            self.redraw_slice_overlay()
            self.update_slice_metrics()
            if result["failed_candidates"]:
                self.append_log(f"Método local: {result['failed_candidates']} candidatos fallidos; no se aplicó otro método.")
            self.refresh_steps()
        except Exception as exc:
            self.current_prediction = None
            self.metrics_text.set("Falló la segmentación local; consultar el registro.")
            self.append_log(f"Segmentación local: {exc}")

    def apply_preprocessing(self) -> None:
        self.show_case_slice(segment=False)

    def segment_current_slice(self) -> None:
        self.show_case_slice(segment=True)

    def show_case_slice(self, segment: bool = False) -> None:
        self.local_preview_token = getattr(self, "local_preview_token", 0) + 1
        self.current_local_result = None
        if self.case_volume is None or self.case_labels is None:
            return
        k = max(0, min(int(self.slice_var.get()), self.case_volume.shape[2] - 1))
        self.slice_var.set(k)
        self.current_hu = self.case_volume[:, :, k]
        self.current_labels = self.case_labels[:, :, k]
        self.current_processed = None
        variant_label = "baseline"
        try:
            config, settings = self.current_settings()
            liver = self.current_labels > 0
            detector = config.get("detection", {})
            if detector.get("enabled"):
                method = detector.get("method", "hysteresis")
                configured_local = method.startswith("local_") and config.get("local_pipeline") == "configured"
                variant_label = ("Preprocesamiento configurado / " if configured_local else "HU originales / ") + method
                if method.startswith("local_"):
                    self.current_prediction = None
                    self.start_local_preview(config, settings, k, self.local_preview_token)
                else:
                    self.current_prediction = configured_prediction(self.current_hu, liver, self.case_spacing, config, settings)["mask"]
            else:
                self.current_processed = prepare_enhanced_slice(self.current_hu, config, liver_mask=liver)
                self.current_prediction = segment_slice(self.current_processed, liver, settings)
        except (ValueError, KeyError) as exc:
            self.current_prediction = None
            variant_label = "parámetros inválidos"
            self.append_log(f"Parámetros inválidos: {exc}")
        self.figure_index = -1
        self.figure_title.set(f"Caso {self.selected_case_id():03d} · corte {k} · {variant_label}")
        self.figure_fit = True
        self.redraw_slice_overlay()
        if self.current_prediction is not None:
            self.update_slice_metrics()
        self.refresh_steps()

    def refresh_steps(self) -> None:
        prepared = self.current_processed is not None or self.current_prediction is not None
        states = (self.case_volume is not None, prepared,
                  self.current_prediction is not None, bool(self.results_dir and self.results_dir.is_dir()))
        labels = ("Caso", "Preparar", "Segmentar", "Resultados")
        self.steps_text.set("  ·  ".join(("✓ " if done else "○ ") + label
                                        for done, label in zip(states, labels)))

    def draw_case_image(self, image: np.ndarray) -> None:
        source = np.asarray(image, dtype=np.float32)
        self.current_crop_box = (0, 0, source.shape[0], source.shape[1])
        if self.preprocessing_vars["crop"].get() and np.any(self.current_labels > 0):
            try:
                self.current_crop_box = liver_bounding_box(self.current_labels > 0, margin=8)
                x0, y0, x1, y1 = self.current_crop_box
                source = source[x0:x1, y0:y1]
            except ValueError:
                pass
        x0, y0, x1, y1 = self.current_crop_box
        liver = self.current_labels[x0:x1, y0:y1] > 0
        if image is not self.current_processed:
            # Map raw HU to grayscale for display only, using the same range as the no-window input.
            source = normalize_slice_hu(source, {"enabled": False}, liver_mask=liver)
        gray = np.clip(np.where(liver, source, 0), 0, 1)
        rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
        opacity = max(0.0, min(1.0, float(self.overlay_opacity.get())))
        if self.view_mode.get() == "Overlay":
            if self.show_reference.get():
                mask = (self.current_labels == 2)[x0:x1, y0:y1]
                rgb[mask] = (rgb[mask] * (1 - opacity) + np.array((40, 180, 255)) * opacity).astype(np.uint8)
            if self.show_prediction.get() and self.current_prediction is not None:
                mask = self.current_prediction[x0:x1, y0:y1]
                rgb[mask] = (rgb[mask] * (1 - opacity) + np.array((255, 155, 40)) * opacity).astype(np.uint8)
            result = getattr(self, "current_local_result", None)
            if result is not None and self.local_overlays.get():
                candidate = result["candidates"][x0:x1, y0:y1]
                edge = candidate & ~ndimage.binary_erosion(candidate)
                rgb[edge] = (180, 80, 255)
                seed = result["seeds"][x0:x1, y0:y1]
                rgb[seed] = (80, 255, 80)
        if self.show_liver_mask.get():
            liver = self.current_labels > 0
            boundary = liver & ~ndimage.binary_erosion(liver)
            rgb[boundary[x0:x1, y0:y1]] = (255, 221, 80)
        self.figure_image = Image.fromarray(rgb.transpose(1, 0, 2))
        self.figure_photo = None
        self.empty_message = ""
        self.figure_counter.set(f"Corte {int(self.slice_var.get()) + 1}")
        self.render_figure()

    def update_slice_metrics(self) -> None:
        prediction = self.current_prediction
        reference = self.current_labels == 2
        labels, count = ndimage.label(prediction, structure=np.ones((3, 3), dtype=bool))
        coverage = reference_coverage(prediction, reference)
        coverage_text = (f"{coverage:.1%} ({int((prediction & reference).sum())}/{int(reference.sum())} px)"
                         if np.isfinite(coverage) else "— (sin referencia tumoral)")
        self.metrics_text.set(f"Cobertura referencia: {coverage_text}\nDice {dice_score(prediction, reference):.3f} · IoU {iou_score(prediction, reference):.3f} · área {int(prediction.sum())} px · lesiones {count}")
        for item in self.lesion_table.get_children():
            self.lesion_table.delete(item)
        for lesion in range(1, count + 1):
            coords = np.argwhere(labels == lesion)
            centroid = coords.mean(axis=0)
            self.lesion_table.insert("", "end", values=(lesion, len(coords), f"{centroid[0]:.1f}, {centroid[1]:.1f}"))

    def pixel_readout(self, event: tk.Event) -> None:
        if self.figure_image is None or self.current_hu is None:
            return
        width, height = max(self.figure_canvas.winfo_width(), 1), max(self.figure_canvas.winfo_height(), 1)
        x0, y0, x1, y1 = getattr(self, "current_crop_box", (0, 0, *self.current_hu.shape))
        display_width, display_height = (x1 - x0) * self.figure_scale, (y1 - y0) * self.figure_scale
        region_width, region_height = max(display_width, width), max(display_height, height)
        left, top = (region_width - display_width) / 2, (region_height - display_height) / 2
        x = int((self.figure_canvas.canvasx(event.x) - left) / max(self.figure_scale, 1e-6)) + x0
        y = int((self.figure_canvas.canvasy(event.y) - top) / max(self.figure_scale, 1e-6)) + y0
        if 0 <= x < self.current_hu.shape[0] and 0 <= y < self.current_hu.shape[1]:
            value = f"Píxel: ({x}, {y})   HU: {self.current_hu[x, y]:.6g}"
            if self.view_mode.get() != "Antes" and self.current_processed is not None:
                value += f"   Entrada: {self.current_processed[x, y]:.9g}"
            self.pixel_status.set(value)

    def zoom_100(self) -> None:
        if self.figure_image is not None:
            self.figure_fit = False
            self.figure_scale = 1.0
            self.render_figure()

    def select_result_item(self, _event=None) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        if not 0 <= index < len(self.results_files):
            return
        path = self.results_files[index]
        if path.suffix.lower() in IMAGE_SUFFIXES:
            self.figure_index = self.figures.index(path)
            self.show_path(path)
        else:
            self.note_title.set(path.name)
            self.note_body.set("Vista de tabla CSV. El archivo completo permanece guardado.")
            self.show_csv(path)

    def show_csv(self, path: Path) -> None:
        try:
            self.current_hu = None
            with path.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.reader(stream))[:35]
            if not rows:
                raise ValueError("El archivo no contiene filas.")
            from PIL import ImageDraw, ImageFont
            columns = min(max(len(row) for row in rows), 7)
            rows = [row[:columns] for row in rows]
            widths = [min(260, max(80, max(len(row[i]) if i < len(row) else 0 for row in rows) * 8))
                      for i in range(columns)]
            image = Image.new("RGB", (sum(widths) + 24, len(rows) * 24 + 20), "white")
            draw, font = ImageDraw.Draw(image), ImageFont.load_default()
            for row_index, row in enumerate(rows):
                x, y = 8, row_index * 24 + 5
                for col, width in enumerate(widths):
                    value = row[col] if col < len(row) else ""
                    draw.text((x + 3, y), value[:34], fill="#20334F" if row_index == 0 else "#53647B", font=font)
                    x += width
                draw.line((8, y + 18, image.width - 8, y + 18), fill="#D7E0EC")
            self.figure_image = image
            self.figure_fit = True
            self.figure_title.set(path.name)
            self.figure_counter.set(f"Tabla · {len(rows)} filas mostradas")
            self.render_figure()
        except (OSError, csv.Error, ValueError) as exc:
            self.show_figure_message(f"No se pudo mostrar {path.name}: {exc}")

    def refresh_results_list(self) -> None:
        if not hasattr(self, "results_tree"):
            return
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.results_files = []
        base = self.results_dir
        if base and base.is_dir():
            self.results_files = sorted((p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES | {".csv"}), key=lambda p: (p.suffix.lower() not in IMAGE_SUFFIXES, str(p).lower()))
        self.figures = [path for path in self.results_files if path.suffix.lower() in IMAGE_SUFFIXES]
        self.figure_index = -1
        for index, path in enumerate(self.results_files):
            kind = "Figura" if path.suffix.lower() in IMAGE_SUFFIXES else "Tabla"
            self.results_tree.insert("", "end", iid=str(index), values=(kind, path.name))

    def browse_results_path(self) -> None:
        selected = filedialog.askdirectory(title="Elegir carpeta de resultados")
        if selected:
            self.results_dir = Path(selected)
            self.refresh_results_list()

    def write_effective_run_config(self) -> Path:
        """Save current sidebar parameters to an ignored config consumed by CLI workflows."""
        config, segmentation = self.current_settings()
        config["segmentation"] = segmentation
        if getattr(self, "external_data_dir", None):
            config["data_dir"] = str(self.external_data_dir.resolve())
        config["output_dir"] = str(Path(config["_output_dir"]).resolve())
        for key in tuple(config):
            if key.startswith("_"):
                config.pop(key)
        profile_dir = ROOT / ".psib_gui"
        profile_dir.mkdir(parents=True, exist_ok=True)
        path = profile_dir / "gui-run.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def build_main_viewer(self) -> None:
        toolbar = ttk.Frame(self.viewer, style="Viewer.TFrame", padding=(10, 8))
        toolbar.pack(fill="x")
        self.previous_button = ttk.Button(toolbar, text="◀", width=3, style="Secondary.TButton",
                                          command=lambda: self.show_figure(-1), state="disabled")
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(toolbar, text="▶", width=3, style="Secondary.TButton",
                                      command=lambda: self.show_figure(1), state="disabled")
        self.next_button.pack(side="left", padx=(4, 10))
        self.figure_counter = tk.StringVar(value="Sin imagen")
        ttk.Label(toolbar, textvariable=self.figure_counter, style="Viewer.TLabel").pack(side="left")
        self.view_toggle = ttk.Combobox(toolbar, textvariable=self.view_mode,
                                        values=("Entrada", "Antes", "Overlay"), state="readonly", width=10)
        self.view_toggle.pack(side="right", padx=(5, 0))
        self.view_toggle.bind("<<ComboboxSelected>>", lambda _event: (
            self.redraw_slice_overlay() if self.current_hu is not None else self.show_case_slice()))
        ttk.Checkbutton(toolbar, text="Contorno hepático", variable=self.show_liver_mask,
                        command=self.redraw_slice_overlay).pack(side="right", padx=5)
        ttk.Button(toolbar, text="100%", style="Secondary.TButton", command=self.zoom_100).pack(side="right", padx=3)
        ttk.Button(toolbar, text="Ajustar", style="Secondary.TButton", command=self.fit_figure).pack(side="right", padx=3)
        ttk.Button(toolbar, text="+", width=3, style="Secondary.TButton", command=lambda: self.zoom_figure(1.25)).pack(side="right", padx=3)
        ttk.Button(toolbar, text="−", width=3, style="Secondary.TButton", command=lambda: self.zoom_figure(0.8)).pack(side="right", padx=3)
        self.figure_title = tk.StringVar(value="Visor de imágenes")
        ttk.Label(self.viewer, textvariable=self.figure_title, style="ViewerTitle.TLabel",
                  padding=(12, 2)).pack(fill="x")
        self.figure_canvas = tk.Canvas(self.viewer, background="#111C2E", highlightthickness=0, cursor="fleur")
        self.figure_canvas.pack(fill="both", expand=True, padx=1, pady=(2, 1))
        self.figure_canvas.bind("<Configure>", self.on_figure_resize)
        self.figure_canvas.bind("<ButtonPress-1>", lambda event: self.figure_canvas.scan_mark(event.x, event.y))
        self.figure_canvas.bind("<B1-Motion>", lambda event: self.figure_canvas.scan_dragto(event.x, event.y, gain=1))
        self.figure_canvas.bind("<MouseWheel>", self.on_figure_wheel)
        self.figure_canvas.bind("<Motion>", self.pixel_readout)
        self.show_figure_message(
            "Seleccioná un caso para previsualizarlo, o definí cuántos casos y cortes evaluar para comenzar.")

    def build_log(self, parent: ttk.Frame) -> None:
        surface = ttk.Frame(parent, style="Log.TFrame", padding=1)
        surface.pack(fill="x", pady=(7, 0))
        self.log = tk.Text(surface, wrap="word", height=7, state="disabled", font=("Cascadia Mono", 8),
                           background="#111C2E", foreground="#D7E0ED", insertbackground="#FFFFFF",
                           relief="flat", borderwidth=0, padx=8, pady=7)
        scrollbar = ttk.Scrollbar(surface, orient="vertical", command=self.log.yview, style="Log.Vertical.TScrollbar")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.log.tag_configure("success", foreground="#71D6A0")
        self.log.tag_configure("error", foreground="#FF8D95")
        self.log.tag_configure("accent", foreground="#83B8FF")

    def configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#F3F6FB")
        style.configure("Panel.TFrame", background="#F3F6FB")
        style.configure("Viewer.TFrame", background="#111C2E")
        style.configure("Viewer.TLabel", background="#111C2E", foreground="#D7E0ED", font=("Segoe UI", 9))
        style.configure("StatusBar.TFrame", background="#E6ECF4", bordercolor="#D7E0EC", relief="solid", borderwidth=1)
        style.configure("StatusBar.TLabel", background="#E6ECF4", foreground="#53647B", font=("Segoe UI", 9))
        style.configure("SidebarTitle.TLabel", background="#F3F6FB", foreground="#20334F", font=("Segoe UI", 12, "bold"))
        style.configure("ViewerTitle.TLabel", background="#111C2E", foreground="#D7E0ED", font=("Segoe UI", 10, "bold"))
        style.configure("Section.TButton", background="#F3F6FB", foreground="#53647B", anchor="w", padding=(9, 7), borderwidth=0)
        style.map("Section.TButton", background=[("active", "#E6ECF4"), ("selected", "#F5A524")],
                  foreground=[("selected", "#17262B"), ("active", "#20334F")])
        style.configure("Rail.TButton", background="#F3F6FB", foreground="#62767C", padding=(3, 6), borderwidth=0)
        style.map("Rail.TButton", background=[("active", "#E6ECF4"), ("selected", "#F5A524")])
        style.configure("RailActivity.TLabel", background="#F3F6FB", foreground="#F5A524", anchor="center")
        style.configure("StepCaption.TLabel", background="#F3F6FB", foreground="#77849A", font=("Segoe UI", 8))
        style.configure("App.TNotebook", background="#F3F6FB", borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("App.TNotebook.Tab", background="#E6ECF4", foreground="#53647B", padding=(18, 9), font=("Segoe UI", 9, "bold"))
        style.map("App.TNotebook.Tab", background=[("selected", "#FFFFFF"), ("active", "#F5F8FC")], foreground=[("selected", "#20334F")])
        style.configure("Card.TFrame", background="#FFFFFF")
        style.configure("Hero.TFrame", background="#172B4D")
        style.configure("Step.TFrame", background="#FFFFFF", borderwidth=1, relief="solid")
        style.configure("Log.TFrame", background="#D9E2EF")
        style.configure("Eyebrow.Hero.TLabel", background="#172B4D", foreground="#A9C7F0", font=("Segoe UI", 9, "bold"))
        style.configure("Badge.Hero.TLabel", background="#24446F", foreground="#DAE8FC", font=("Segoe UI", 9, "bold"))
        style.configure("Title.Hero.TLabel", background="#172B4D", foreground="#FFFFFF", font=("Segoe UI", 23, "bold"))
        style.configure("Subtitle.Hero.TLabel", background="#172B4D", foreground="#C3D2E6", font=("Segoe UI", 10))
        style.configure("StepTitle.TLabel", background="#FFFFFF", foreground="#20334F", font=("Segoe UI", 10, "bold"))
        style.configure("StepCaption.TLabel", background="#FFFFFF", foreground="#77849A", font=("Segoe UI", 9))
        style.configure("SectionTitle.TLabel", background="#F3F6FB", foreground="#1D2F49", font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background="#F3F6FB", foreground="#68778D", font=("Segoe UI", 9))
        style.configure("LogMeta.TLabel", background="#FFFFFF", foreground="#8A96A8", font=("Segoe UI", 8, "bold"))
        style.configure("Config.TEntry", fieldbackground="#F8FAFD", foreground="#26364D", padding=(10, 9), bordercolor="#D8E1ED")
        style.map("Config.TEntry", bordercolor=[("focus", "#5D8ED6")])
        style.configure("Primary.TButton", background="#2864C5", foreground="#FFFFFF", font=("Segoe UI", 10, "bold"), padding=(16, 10), borderwidth=0)
        style.configure("Primary.TButton", background="#F5A524", foreground="#17262B", font=("Segoe UI", 10, "bold"), padding=(12, 8), borderwidth=0)
        style.map("Primary.TButton", background=[("disabled", "#E8D9B8"), ("pressed", "#DD9013"), ("active", "#DD9013")])
        style.configure("Secondary.TButton", background="#FFFFFF", foreground="#29405F", font=("Segoe UI", 9, "bold"), padding=(13, 9), borderwidth=1, bordercolor="#D7E0EC")
        style.map("Secondary.TButton", background=[("disabled", "#F4F6F9"), ("pressed", "#E6EDF6"), ("active", "#F5F8FC")])
        style.configure("StatusReady.TFrame", background="#E8F4ED")
        style.configure("StatusReady.TLabel", background="#E8F4ED", foreground="#28734B", font=("Segoe UI", 9, "bold"))
        style.configure("StatusRunning.TFrame", background="#EAF1FC")
        style.configure("StatusRunning.TLabel", background="#EAF1FC", foreground="#2859A2", font=("Segoe UI", 9, "bold"))
        style.configure("StatusSuccess.TFrame", background="#E8F4ED")
        style.configure("StatusSuccess.TLabel", background="#E8F4ED", foreground="#28734B", font=("Segoe UI", 9, "bold"))
        style.configure("StatusError.TFrame", background="#FCEBEC")
        style.configure("StatusError.TLabel", background="#FCEBEC", foreground="#A83B43", font=("Segoe UI", 9, "bold"))
        style.configure("Run.Horizontal.TProgressbar", background="#3475D7", troughcolor="#DFE8F5", bordercolor="#DFE8F5", lightcolor="#3475D7", darkcolor="#3475D7")
        style.configure("Log.Vertical.TScrollbar", background="#E1E8F1", troughcolor="#111C2E", bordercolor="#111C2E", arrowcolor="#64758D")

    def load_morphology_results(self) -> None:
        try:
            config = load_config(self.config_path.get())
            output = Path(config["_output_dir"]) / "morphology_3d"
            summary_path = output / "morphology_3d_per_case.csv"
            if summary_path.is_file():
                self.show_morphology_results(output)
        except (OSError, ValueError, KeyError):
            return

    def show_morphology_results(self, output: Path) -> None:
        self.morphology_output = output
        summary_path = output / "morphology_3d_per_case.csv"
        for item in self.morphology_table.get_children():
            self.morphology_table.delete(item)
        try:
            with summary_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    case_id = int(row["case_id"])
                    diameter_value = row.get("max_axial_feret_mm")
                    diameter = f"{float(diameter_value):.1f}" if diameter_value else "—"
                    self.morphology_table.insert("", "end", values=(
                        f"{case_id:03d}", row["connected_components_6"],
                        f"{float(row['tumor_volume_cm3']):.2f}", diameter))
        except (OSError, ValueError, KeyError) as exc:
            self.append_log(f"No se pudo leer el resumen 3D: {exc}")
            return
        self.morphology_output = output
        self.results_dir = output
        self.refresh_results_list()
        self.show_section("Cuantificación")

    def open_morphology_results(self) -> None:
        if getattr(self, "morphology_output", None) and self.morphology_output.is_dir():
            os.startfile(self.morphology_output)

    def set_status(self, text: str, state: str) -> None:
        self.status.set(text)
        style = f"Status{state.title()}.TFrame"
        label_style = f"Status{state.title()}.TLabel"
        self.status_pill.configure(style=style)
        self.status_dot.configure(style=label_style)
        self.status_label.configure(style=label_style)

    def queue_icon_warning(self, message: str) -> None:
        """Hold the single icon warning until the GUI log widget exists."""
        self.pending_icon_warning = message

    def load_latest_results(self) -> None:
        """Show figures from the latest completed run for the selected output directory."""
        try:
            config = load_config(self.config_path.get())
            segmentation_dir = Path(config["_output_dir"]) / "segmentation"
            runs = sorted(
                (path for path in segmentation_dir.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except (OSError, ValueError, KeyError):
            return
        for run_dir in runs:
            try:
                state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if state.get("status") == "completed":
                figure_count = self.load_run_figures(run_dir)
                self.append_log(f"Última corrida completada: {run_dir.name}")
                self.append_log(
                    f"Casos: {state.get('n_cases', '—')}  ·  "
                    f"Cortes: {state.get('n_slices', '—')}  ·  "
                    f"Configuraciones: {state.get('n_configurations', '—')}"
                )
                self.append_log(f"Figuras disponibles: {figure_count}")
                self.append_log(f"Carpeta: {run_dir}")
                self.log_meta.set("RESUMEN DE LA ÚLTIMA CORRIDA")
                if figure_count:
                    self.set_status(f"Última corrida · {figure_count} figuras", "success")
                    self.show_section("Resultados")
                else:
                    self.set_status("Última corrida sin figuras", "error")
                return

        self.append_log("Todavía no hay una corrida completada. Ejecutá el flujo 2D para ver aquí el progreso y el resultado.")

    def load_run_figures(self, run_dir: Path) -> int:
        """Collect this run's exported images and open the first one in the viewer."""
        self.results_dir = run_dir
        self.figures = sorted(
            path for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.figure_index = -1
        if not self.figures:
            self.show_figure_message("Esta corrida terminó, pero no exportó figuras")
            self.previous_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
            return 0
        self.refresh_results_list()
        self.previous_button.configure(state="normal" if len(self.figures) > 1 else "disabled")
        self.next_button.configure(state="normal" if len(self.figures) > 1 else "disabled")
        self.show_figure(1)
        return len(self.figures)

    def show_figure_message(self, message: str) -> None:
        self.empty_message = message
        self.current_hu = None
        self.figure_image = None
        self.figure_photo = None
        self.figure_title.set("Visor de figuras")
        self.note_title.set("Visor")
        self.note_body.set(message)
        self.figure_counter.set("Sin figuras")
        if hasattr(self, "previous_button"):
            self.previous_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
        self.render_figure()

    def show_figure(self, step: int) -> None:
        if not self.figures:
            return
        self.figure_index = (self.figure_index + step) % len(self.figures)
        path = self.figures[self.figure_index]
        self.show_path(path)

    def show_path(self, path: Path) -> None:
        self.current_hu = None
        self.empty_message = ""
        try:
            with Image.open(path) as source:
                self.figure_image = source.convert("RGB")
        except OSError as exc:
            self.show_figure_message(f"No se pudo abrir {path.name}: {exc}")
            return

        match = re.search(r"case_(\d+)_slice_(\d+)", path.stem, flags=re.IGNORECASE)
        if match:
            case_id, slice_index = match.groups()
            self.figure_title.set(f"Caso {int(case_id):03d} · corte axial {int(slice_index)}")
            description = f"{path.parent.name} · Figura de segmentación del corte {int(slice_index)}."
        else:
            self.figure_title.set(path.stem.replace("_", " "))
            description = f"Archivo: {path.name}"
        description = self.figure_note(path, description)
        self.note_title.set(path.stem.replace("_", " "))
        self.note_body.set(description)
        self.figure_counter.set(f"Figura {self.figure_index + 1} de {len(self.figures)}")
        self.figure_fit = True
        self.render_figure()
        self.figure_canvas.xview_moveto(0)
        self.figure_canvas.yview_moveto(0)

    def figure_note(self, path: Path, fallback: str) -> str:
        """Read optional figure-specific explanations from figure_notes.yaml."""
        try:
            notes = yaml.safe_load((ROOT / "figure_notes.yaml").read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return fallback
        note = notes.get(path.stem) if isinstance(notes, dict) else None
        if note is None and isinstance(notes, dict):
            candidates = [str(key) for key in notes if str(key).endswith("*") and path.stem.startswith(str(key)[:-1])]
            if candidates:
                note = notes[max(candidates, key=len)]
        return str(note).strip() if note else fallback

    def render_figure(self) -> None:
        canvas = self.figure_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        if self.figure_image is None:
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_text(
                width / 2,
                height / 2,
                text=self.empty_message,
                fill="#AAB7C8",
                font=("Segoe UI", 11),
                justify="center",
                width=max(width - 40, 100),
            )
            return
        image_width, image_height = self.figure_image.size
        if self.figure_fit:
            self.figure_scale = min((width - 24) / image_width, (height - 24) / image_height, 1.0)
        display_size = (
            max(1, round(image_width * self.figure_scale)),
            max(1, round(image_height * self.figure_scale)),
        )
        if display_size == self.figure_image.size:
            rendered = self.figure_image
        else:
            rendered = self.figure_image.resize(display_size, Image.Resampling.LANCZOS)
        self.figure_photo = ImageTk.PhotoImage(rendered)
        scroll_width, scroll_height = max(display_size[0], width), max(display_size[1], height)
        canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
        canvas.create_image(scroll_width / 2, scroll_height / 2, image=self.figure_photo, anchor="center")

    def fit_figure(self) -> None:
        if self.figure_image is not None:
            self.figure_fit = True
            self.render_figure()
            self.figure_canvas.xview_moveto(0)
            self.figure_canvas.yview_moveto(0)

    def zoom_figure(self, factor: float) -> None:
        if self.figure_image is None:
            return
        if self.figure_fit:
            width, height = self.figure_image.size
            self.figure_scale = min(
                max(self.figure_canvas.winfo_width() - 24, 50) / width,
                max(self.figure_canvas.winfo_height() - 24, 50) / height,
                1.0,
            )
        width, height = self.figure_image.size
        limit = min(8.0, (40e6 / (width * height)) ** 0.5)
        self.figure_fit = False
        self.figure_scale = max(0.05, min(self.figure_scale * factor, limit))
        self.render_figure()

    def on_figure_resize(self, _event: tk.Event) -> None:
        if self.figure_image is None or self.figure_fit:
            self.render_figure()

    def on_figure_wheel(self, event: tk.Event) -> str:
        if event.state & 0x4:
            self.zoom_figure(1.25 if event.delta > 0 else 0.8)
        else:
            self.figure_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def browse_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Elegir configuración del TPI",
            initialdir=str(Path(self.config_path.get()).parent),
            filetypes=(("Archivos YAML", "*.yaml *.yml"), ("Todos los archivos", "*.*")),
        )
        if selected:
            self.config_path.set(selected)
            self.external_data_dir = None
            self.analysis_case_ids = []
            self.last_selection_signature = None
            try:
                selection = load_config(selected).get("selection", {})
                self.selection_cases_var.set(str(selection.get("n_cases", 10)))
                self.selection_slices_var.set(str(selection.get("slices_per_case", 5)))
                edge_settings = load_config(selected).get("edge_analysis", {})
                operator = str(edge_settings.get("operator", "sobel")).lower()
                if operator in ("none", "sobel", "prewitt", "laplace"):
                    self.preprocessing_vars["edge_operator"].set(operator)
                self.preprocessing_vars["sharpen_alpha"].set(
                    str(edge_settings.get("sharpen_alpha", 0.5)))
            except Exception as exc:
                messagebox.showerror("Configuración inválida", str(exc))
                return
            self.refresh_cases()
            self.reset_results_view("Seleccioná un caso para empezar una nueva sesión.")

    def reset_results_view(self, message: str) -> None:
        """Clear displayed results without deleting prior files from disk."""
        self.local_preview_token = getattr(self, "local_preview_token", 0) + 1
        self.results_dir = None
        self.figures = []
        self.results_files = []
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        for item in self.morphology_table.get_children():
            self.morphology_table.delete(item)
        self.show_figure_message(message)

    def append_log(self, line: str) -> None:
        tag = None
        lowered = line.lower()
        if line.startswith(RESULT_PREFIX):
            tag = "success"
        elif "error" in lowered or "traceback" in lowered:
            tag = "error"
        elif "evaluado caso" in lowered or "selección 2d" in lowered or "selecting 2d" in lowered:
            tag = "accent"
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n", (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def run(self) -> None:
        if self.task_running or self.process is not None:
            return
        config = Path(self.config_path.get()).expanduser()
        if not config.is_file():
            messagebox.showerror("Configuración no encontrada", f"No existe este archivo:\n{config}")
            return
        try:
            config = self.write_effective_run_config()
        except Exception as exc:
            messagebox.showerror("Configuración inválida", str(exc))
            return
        self.results_dir = None
        self.cancel_requested = False
        self.task_running = True
        self.run_button.configure(state="disabled")
        for button in self.task_buttons:
            button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.log_meta.set("SALIDA EN VIVO")
        self.set_status("Ejecutando…", "running")
        self.show_figure_message("La corrida está generando sus figuras…")
        self.show_section("Resultados")
        self.progress.pack(fill="x", pady=(0, 5))
        self.progress.start(12)
        self.append_log(f"\n> Iniciando flujo con {config.resolve()}")
        thread = threading.Thread(target=self.run_worker, args=(config.resolve(),), daemon=True)
        thread.start()
        self.root.after(100, self.poll_events)

    def run_morphology(self) -> None:
        if self.task_running or self.process is not None:
            return
        config = Path(self.config_path.get()).expanduser()
        if not config.is_file():
            messagebox.showerror("Configuración no encontrada", f"No existe este archivo:\n{config}")
            return
        try:
            config = self.write_effective_run_config()
        except Exception as exc:
            messagebox.showerror("Configuración inválida", str(exc))
            return
        self.cancel_requested = False
        self.task_running = True
        self.run_button.configure(state="disabled")
        for button in self.task_buttons:
            button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.log_meta.set("ANÁLISIS MORFOLÓGICO 3D")
        self.set_status("Analizando máscaras 3D…", "running")
        self.show_section("Cuantificación")
        self.progress.pack(fill="x", pady=(0, 5))
        self.progress.start(12)
        self.append_log(f"\n> Iniciando morfología 3D con {config.resolve()}")
        threading.Thread(target=self.morphology_worker, args=(config.resolve(),), daemon=True).start()
        self.root.after(100, self.poll_events)

    def run_slice_selection(self) -> None:
        """Generate the configured reproducible 2D slice selection from the GUI."""
        self.start_cli_task("select_slices_2d.py", "Generando selección 2D…", "Caso",
                            task="selection")

    def run_compression(self) -> None:
        """Run the compression experiment for the currently selected slice CSV."""
        self.start_experiment("run_compression.py", "compression", "Experimento de compresión…")

    def run_reconstruction(self) -> None:
        """Run the simulated reconstruction experiment for selected slices."""
        self.start_experiment("run_reconstruction.py", "reconstruction", "Reconstrucción simulada…")

    def start_experiment(self, script: str, kind: str, label: str) -> None:
        try:
            if self.last_selection_signature != self.current_selection_signature():
                messagebox.showinfo("Generá la selección actual",
                                    "Primero aplicá y generá la selección para los casos y cortes elegidos.")
                return
        except Exception as exc:
            messagebox.showerror("Configuración inválida", str(exc))
            return
        try:
            cfg = load_config(self.config_path.get())
            selection = Path(cfg["_output_dir"]) / "selected_cases_slices.csv"
        except Exception as exc:
            messagebox.showerror("Configuración inválida", str(exc))
            return
        if not selection.is_file():
            messagebox.showinfo("Falta la selección de cortes",
                                "Primero generá selected_cases_slices.csv con «Generar selección de cortes 2D» o ejecutá el flujo 2D.")
            return
        run_id = datetime.now().strftime("gui-%Y%m%d-%H%M%S-%f")
        result_dir = Path(cfg["_output_dir"]) / kind / run_id
        self.start_cli_task(script, label, "Cuantificación", task=kind,
                            extra_args=("--run-id", run_id), result_dir=result_dir)

    def start_cli_task(self, script: str, label: str, section: str, *, task: str,
                       extra_args: tuple[str, ...] = (), result_dir: Path | None = None) -> None:
        """Run one repository CLI workflow asynchronously and stream its output to the GUI."""
        if self.task_running or self.process is not None:
            return
        source_config = Path(self.config_path.get()).expanduser()
        if not source_config.is_file():
            messagebox.showerror("Configuración no encontrada", f"No existe este archivo:\n{source_config}")
            return
        try:
            config = self.write_effective_run_config()
        except Exception as exc:
            messagebox.showerror("Configuración inválida", str(exc))
            return
        self.cancel_requested = False
        self.task_running = True
        self.run_button.configure(state="disabled")
        for button in self.task_buttons:
            button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.log_meta.set(label.upper())
        self.set_status(label, "running")
        self.show_section(section)
        self.progress.pack(fill="x", pady=(0, 5))
        self.progress.start(12)
        self.append_log(f"\n> Iniciando {label} con {config.resolve()}")
        args = (sys.executable, str(ROOT / "scripts" / script), "--config", str(config.resolve()), *extra_args)
        threading.Thread(target=self.cli_task_worker,
                         args=(args, task, result_dir), daemon=True).start()
        self.root.after(100, self.poll_events)

    def cli_task_worker(self, args: tuple[str, ...], task: str,
                        result_dir: Path | None) -> None:
        try:
            interpreter = Path(args[0])
            if interpreter.name.lower() == "pythonw.exe":
                console_python = interpreter.with_name("python.exe")
                if console_python.is_file():
                    args = (str(console_python), *args[1:])
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, env=environment)
            if self.cancel_requested:
                self.process.terminate()
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.events.put(("task_line", line.rstrip()))
            self.events.put(("task_done", (task, result_dir, self.process.wait())))
        except Exception as exc:
            self.events.put(("task_error", (task, str(exc))))

    def morphology_worker(self, config: Path) -> None:
        try:
            interpreter = Path(sys.executable)
            if interpreter.name.lower() == "pythonw.exe":
                console_python = interpreter.with_name("python.exe")
                if console_python.is_file():
                    interpreter = console_python
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(
                [str(interpreter), str(ROOT / "scripts" / "analyze_morphology_3d.py"), "--config", str(config)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", bufsize=1, env=environment,
            )
            if self.cancel_requested:
                self.process.terminate()
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.events.put(("morph_line", line.rstrip()))
            self.events.put(("morph_done", self.process.wait()))
        except Exception as exc:
            self.events.put(("morph_error", str(exc)))

    def run_worker(self, config: Path) -> None:
        try:
            interpreter = Path(sys.executable)
            if interpreter.name.lower() == "pythonw.exe":
                console_python = interpreter.with_name("python.exe")
                if console_python.is_file():
                    interpreter = console_python
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(
                [str(interpreter), str(ROOT / "scripts" / "run_tpi.py"), "--config", str(config)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )
            if self.cancel_requested:
                self.process.terminate()
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.events.put(("line", line.rstrip()))
            self.events.put(("done", self.process.wait()))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def poll_events(self) -> None:
        finished = False
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                line = str(value)
                self.append_log(line)
                if line.startswith(RESULT_PREFIX):
                    result = line[len(RESULT_PREFIX):].removesuffix(".")
                    self.results_dir = Path(result)
            elif kind == "task_line":
                self.append_log(str(value))
            elif kind == "task_error":
                task, error = value
                self.append_log(f"Error al iniciar {task}: {error}")
                self.log_meta.set(f"ERROR AL INICIAR {str(task).upper()}")
                self.set_status("Error al iniciar", "error")
                finished = True
            elif kind == "task_done":
                task, result_dir, code = value
                if int(code) == 0:
                    if task == "selection":
                        self.refresh_cases()
                        try:
                            cfg = load_config(self.config_path.get())
                            selection_path = Path(cfg["_output_dir"]) / "selected_cases_slices.csv"
                            with selection_path.open(newline="", encoding="utf-8") as stream:
                                rows = list(csv.DictReader(stream))
                            self.last_selection_signature = self.current_selection_signature()
                            case_count = len({row["case_id"] for row in rows})
                            requested = (int(self.selection_cases_var.get())
                                         * int(self.selection_slices_var.get()))
                            message = f"Selección generada: {case_count} casos, {len(rows)} cortes"
                            self.append_log(message)
                            complete = len(rows) >= requested
                            self.log_meta.set("SELECCIÓN 2D LISTA" if complete else "SELECCIÓN 2D INCOMPLETA")
                            self.set_status(message, "success" if complete else "error")
                            if not complete:
                                self.append_log(f"Se solicitaron {requested} cortes; revisá las advertencias de selección por caso.")
                        except (OSError, ValueError, KeyError) as exc:
                            self.append_log(f"No se pudo resumir la selección generada: {exc}")
                            self.set_status("Selección generada; no se pudo leer su resumen", "error")
                    elif result_dir and result_dir.is_dir():
                        self.results_dir = result_dir
                        self.refresh_results_list()
                        count = self.load_run_figures(result_dir)
                        self.log_meta.set(f"{str(task).upper()} FINALIZADO")
                        description = f"{count} figuras" if count else "resultados guardados"
                        self.set_status(f"{task.capitalize()} listo · {description}", "success")
                        self.show_section("Resultados")
                    else:
                        self.set_status(f"{task.capitalize()} finalizado", "success")
                else:
                    self.log_meta.set(f"{str(task).upper()} CON ERROR")
                    self.set_status(f"Finalizó con error ({code})", "error")
                finished = True
            elif kind == "morph_line":
                self.append_log(str(value))
            elif kind == "morph_error":
                self.append_log(f"Error al iniciar el análisis 3D: {value}")
                self.log_meta.set("ERROR EN MORFOLOGÍA 3D")
                self.set_status("Error al iniciar", "error")
                finished = True
            elif kind == "morph_done":
                code = int(value)
                if code == 0:
                    try:
                        cfg = load_config(self.config_path.get())
                        output = Path(cfg["_output_dir"]) / "morphology_3d"
                        self.show_morphology_results(output)
                        self.log_meta.set("ANÁLISIS 3D FINALIZADO")
                        self.set_status("Morfología 3D lista", "success")
                        self.show_section("Cuantificación")
                    except Exception as exc:
                        self.append_log(f"No se pudieron cargar los resultados 3D: {exc}")
                        self.log_meta.set("ERROR AL CARGAR MORFOLOGÍA 3D")
                        self.set_status("Error al cargar", "error")
                else:
                    self.log_meta.set("ANÁLISIS 3D CON ERROR")
                    self.set_status(f"Análisis 3D con error ({code})", "error")
                finished = True
            elif kind == "error":
                self.append_log(f"Error al iniciar el proceso: {value}")
                self.log_meta.set("ERROR AL INICIAR")
                self.set_status("Error al iniciar", "error")
                finished = True
            elif kind == "done":
                code = int(value)
                if code == 0:
                    self.last_selection_signature = self.current_selection_signature()
                    count = self.load_run_figures(self.results_dir) if self.results_dir else 0
                    self.log_meta.set("CORRIDA FINALIZADA")
                    if count:
                        self.set_status(f"Listo · {count} figuras", "success")
                        self.show_section("Resultados")
                    else:
                        self.set_status("Finalizó sin figuras", "error")
                else:
                    self.log_meta.set("CORRIDA CON ERROR")
                    self.set_status(f"Finalizó con error ({code})", "error")
                self.refresh_steps()
                finished = True
        if finished:
            self.progress.stop()
            self.progress.pack_forget()
            self.process = None
            self.task_running = False
            self.cancel_requested = False
            self.run_button.configure(state="normal")
            self.morphology_button.configure(state="normal")
            for button in self.task_buttons:
                button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.rail_activity.configure(text="")
        else:
            if self.sidebar_state.collapsed:
                self.rail_activity.configure(text="●")
            self.root.after(100, self.poll_events)

    def cancel_run(self) -> None:
        self.cancel_requested = True
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            self.set_status("Cancelando…", "running")

    def open_results(self) -> None:
        if self.results_dir and self.results_dir.is_dir():
            os.startfile(self.results_dir)  # Windows desktop launcher.

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Ejecución en curso", "¿Cerrar y detener la ejecución actual?"):
                return
            self.process.terminate()
        self.local_preview_token = getattr(self, "local_preview_token", 0) + 1
        if hasattr(self, "local_executor"):
            self.local_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ITBA.PSIB.LiverTumorGUI")
        except Exception:
            pass
    root = tk.Tk()
    TpiLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
