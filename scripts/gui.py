"""Small Windows-friendly graphical launcher for the 2D TPI workflow."""
from __future__ import annotations

from pathlib import Path
import os
import queue
import re
import json
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psib.config import load_config

DEFAULT_CONFIG = ROOT / "config.yaml"
RESULT_PREFIX = "TPI run completed. Results are in "
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class TpiLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.results_dir: Path | None = None
        self.figures: list[Path] = []
        self.figure_index = -1
        self.figure_image: Image.Image | None = None
        self.figure_photo: ImageTk.PhotoImage | None = None
        self.empty_message = ""
        self.figure_scale = 1.0
        self.figure_fit = True

        root.title("TPI | Segmentación hepática")
        root.geometry("1000x640")
        root.minsize(800, 540)
        root.configure(background="#F3F6FB")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.configure_styles()

        frame = ttk.Frame(root, style="App.TFrame", padding=(24, 15))
        frame.pack(fill="both", expand=True)

        hero = ttk.Frame(frame, style="Hero.TFrame", padding=(22, 13))
        hero.pack(fill="x")
        hero_top = ttk.Frame(hero, style="Hero.TFrame")
        hero_top.pack(fill="x")
        ttk.Label(hero_top, text="PSIB  /  PROYECTO INTEGRADOR", style="Eyebrow.Hero.TLabel").pack(side="left")
        ttk.Label(hero_top, text="FLUJO 2D", style="Badge.Hero.TLabel", padding=(11, 5)).pack(side="right")
        ttk.Label(hero, text="Segmentación hepática", style="Title.Hero.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Label(
            hero,
            text="Selección reproducible de cortes · segmentación base · resultados listos para revisar",
            style="Subtitle.Hero.TLabel",
        ).pack(anchor="w")

        config_card = ttk.Frame(frame, style="Card.TFrame", padding=(15, 10))
        config_card.pack(fill="x")
        ttk.Label(config_card, text="Configuración", style="SectionTitle.TLabel").pack(anchor="w")
        entry_row = ttk.Frame(config_card, style="Card.TFrame")
        entry_row.pack(fill="x", pady=(7, 0))
        self.config_path = tk.StringVar(value=str(DEFAULT_CONFIG))
        ttk.Entry(entry_row, textvariable=self.config_path, style="Config.TEntry").pack(side="left", fill="x", expand=True)
        ttk.Button(entry_row, text="Examinar…", style="Secondary.TButton", command=self.browse_config).pack(side="left", padx=(9, 0))

        controls = ttk.Frame(frame, style="App.TFrame")
        controls.pack(fill="x", pady=(10, 10))
        self.run_button = ttk.Button(controls, text="▶   Ejecutar flujo 2D", style="Primary.TButton", command=self.run)
        self.run_button.pack(side="left")
        self.open_button = ttk.Button(controls, text="Abrir carpeta de resultados", style="Secondary.TButton", command=self.open_results, state="disabled")
        self.open_button.pack(side="left", padx=(9, 0))
        self.status = tk.StringVar(value="Listo para ejecutar")
        self.status_pill = ttk.Frame(controls, style="StatusReady.TFrame", padding=(12, 8))
        self.status_pill.pack(side="right")
        self.status_dot = ttk.Label(self.status_pill, text="●", style="StatusReady.TLabel")
        self.status_dot.pack(side="left")
        self.status_label = ttk.Label(self.status_pill, textvariable=self.status, style="StatusReady.TLabel")
        self.status_label.pack(side="left", padx=(7, 0))
        self.progress = ttk.Progressbar(frame, mode="indeterminate", style="Run.Horizontal.TProgressbar")

        self.tabs = ttk.Notebook(frame, style="App.TNotebook")
        self.tabs.pack(fill="both", expand=True)
        self.viewer_tab = ttk.Frame(self.tabs, style="App.TFrame", padding=(0, 0, 0, 2))
        self.log_tab = ttk.Frame(self.tabs, style="App.TFrame", padding=(0, 0, 0, 2))
        self.tabs.add(self.viewer_tab, text="Figuras")
        self.tabs.add(self.log_tab, text="Registro")
        self.build_figure_viewer()

        log_card = ttk.Frame(self.log_tab, style="Card.TFrame", padding=(16, 13))
        log_card.pack(fill="both", expand=True)
        log_header = ttk.Frame(log_card, style="Card.TFrame")
        log_header.pack(fill="x", pady=(0, 10))
        ttk.Label(log_header, text="Actividad", style="SectionTitle.TLabel").pack(side="left")
        self.log_meta = tk.StringVar(value="LISTO PARA EJECUTAR")
        ttk.Label(log_header, textvariable=self.log_meta, style="LogMeta.TLabel").pack(side="right")
        log_surface = ttk.Frame(log_card, style="Log.TFrame", padding=1)
        log_surface.pack(fill="both", expand=True)
        self.log = tk.Text(
            log_surface,
            wrap="word",
            state="disabled",
            font=("Cascadia Mono", 9),
            background="#111C2E",
            foreground="#D7E0ED",
            insertbackground="#FFFFFF",
            selectbackground="#315B91",
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=12,
            spacing1=2,
            spacing3=3,
        )
        scrollbar = ttk.Scrollbar(log_surface, orient="vertical", command=self.log.yview, style="Log.Vertical.TScrollbar")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.log.tag_configure("success", foreground="#71D6A0")
        self.log.tag_configure("error", foreground="#FF8D95")
        self.log.tag_configure("accent", foreground="#83B8FF")
        self.load_latest_results()

    def configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#F3F6FB")
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
        style.configure("SectionTitle.TLabel", background="#FFFFFF", foreground="#1D2F49", font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background="#FFFFFF", foreground="#68778D", font=("Segoe UI", 9))
        style.configure("LogMeta.TLabel", background="#FFFFFF", foreground="#8A96A8", font=("Segoe UI", 8, "bold"))
        style.configure("Config.TEntry", fieldbackground="#F8FAFD", foreground="#26364D", padding=(10, 9), bordercolor="#D8E1ED")
        style.map("Config.TEntry", bordercolor=[("focus", "#5D8ED6")])
        style.configure("Primary.TButton", background="#2864C5", foreground="#FFFFFF", font=("Segoe UI", 10, "bold"), padding=(16, 10), borderwidth=0)
        style.map("Primary.TButton", background=[("disabled", "#AEBBD0"), ("pressed", "#1E4D99"), ("active", "#3475D7")])
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

    def build_figure_viewer(self) -> None:
        toolbar = ttk.Frame(self.viewer_tab, style="App.TFrame")
        toolbar.pack(fill="x", pady=(0, 9))
        self.previous_button = ttk.Button(toolbar, text="◀  Anterior", style="Secondary.TButton", command=lambda: self.show_figure(-1), state="disabled")
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(toolbar, text="Siguiente  ▶", style="Secondary.TButton", command=lambda: self.show_figure(1), state="disabled")
        self.next_button.pack(side="left", padx=(7, 0))
        ttk.Button(toolbar, text="Ajustar", style="Secondary.TButton", command=self.fit_figure).pack(side="right")
        ttk.Button(toolbar, text="+", style="Secondary.TButton", command=lambda: self.zoom_figure(1.25), width=3).pack(side="right", padx=(6, 0))
        ttk.Button(toolbar, text="−", style="Secondary.TButton", command=lambda: self.zoom_figure(0.8), width=3).pack(side="right", padx=(6, 0))
        self.figure_counter = tk.StringVar(value="Sin figuras")
        ttk.Label(toolbar, textvariable=self.figure_counter, style="Body.TLabel").pack(side="left", padx=(13, 0))

        info = ttk.Frame(self.viewer_tab, style="Card.TFrame", padding=(14, 10))
        info.pack(fill="x", pady=(0, 9))
        self.figure_title = tk.StringVar(value="Las figuras de la corrida aparecerán aquí")
        self.figure_description = tk.StringVar(
            value="Ejecutá el flujo para explorar las imágenes de entrada, las máscaras de referencia y las predicciones."
        )
        ttk.Label(info, textvariable=self.figure_title, style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(info, textvariable=self.figure_description, style="Body.TLabel", wraplength=880).pack(anchor="w", pady=(3, 0))

        viewer = ttk.Frame(self.viewer_tab, style="Card.TFrame", padding=1)
        viewer.pack(fill="both", expand=True)
        self.figure_canvas = tk.Canvas(viewer, background="#111C2E", highlightthickness=0, cursor="fleur")
        vertical = ttk.Scrollbar(viewer, orient="vertical", command=self.figure_canvas.yview, style="Log.Vertical.TScrollbar")
        horizontal = ttk.Scrollbar(viewer, orient="horizontal", command=self.figure_canvas.xview, style="Log.Vertical.TScrollbar")
        self.figure_canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side="right", fill="y")
        horizontal.pack(side="bottom", fill="x")
        self.figure_canvas.pack(fill="both", expand=True)
        self.figure_canvas.bind("<Configure>", self.on_figure_resize)
        self.figure_canvas.bind("<ButtonPress-1>", lambda event: self.figure_canvas.scan_mark(event.x, event.y))
        self.figure_canvas.bind("<B1-Motion>", lambda event: self.figure_canvas.scan_dragto(event.x, event.y, gain=1))
        self.figure_canvas.bind("<MouseWheel>", self.on_figure_wheel)
        self.show_figure_message("Todavía no hay figuras cargadas")

    def set_status(self, text: str, state: str) -> None:
        self.status.set(text)
        style = f"Status{state.title()}.TFrame"
        label_style = f"Status{state.title()}.TLabel"
        self.status_pill.configure(style=style)
        self.status_dot.configure(style=label_style)
        self.status_label.configure(style=label_style)

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
                    self.open_button.configure(state="normal")
                    self.tabs.select(self.viewer_tab)
                else:
                    self.set_status("Última corrida sin figuras", "error")
                    self.tabs.select(self.log_tab)
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
        self.previous_button.configure(state="normal" if len(self.figures) > 1 else "disabled")
        self.next_button.configure(state="normal" if len(self.figures) > 1 else "disabled")
        self.show_figure(1)
        return len(self.figures)

    def show_figure_message(self, message: str) -> None:
        self.empty_message = message
        self.figure_image = None
        self.figure_photo = None
        self.figure_title.set("Visor de figuras")
        self.figure_description.set(message)
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
            self.figure_description.set(
                f"{path.parent.name} · De izquierda a derecha: imagen preprocesada, referencia tumoral y predicción. "
                "La imagen incluye Dice e IoU para este corte."
            )
        else:
            self.figure_title.set(path.stem.replace("_", " "))
            self.figure_description.set(f"Archivo: {path.name}")
        self.figure_counter.set(f"Figura {self.figure_index + 1} de {len(self.figures)}")
        self.figure_fit = True
        self.render_figure()
        self.figure_canvas.xview_moveto(0)
        self.figure_canvas.yview_moveto(0)

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
        config = Path(self.config_path.get()).expanduser()
        if not config.is_file():
            messagebox.showerror("Configuración no encontrada", f"No existe este archivo:\n{config}")
            return
        self.results_dir = None
        self.open_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.log_meta.set("SALIDA EN VIVO")
        self.set_status("Ejecutando…", "running")
        self.show_figure_message("La corrida está generando sus figuras…")
        self.tabs.select(self.log_tab)
        self.progress.pack(fill="x", pady=(0, 12), before=self.tabs)
        self.progress.start(12)
        self.append_log(f"\n> Iniciando flujo con {config.resolve()}")
        thread = threading.Thread(target=self.run_worker, args=(config.resolve(),), daemon=True)
        thread.start()
        self.root.after(100, self.poll_events)

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
            elif kind == "error":
                self.append_log(f"Error al iniciar el proceso: {value}")
                self.log_meta.set("ERROR AL INICIAR")
                self.set_status("Error al iniciar", "error")
                finished = True
            elif kind == "done":
                code = int(value)
                if code == 0:
                    count = self.load_run_figures(self.results_dir) if self.results_dir else 0
                    self.log_meta.set("CORRIDA FINALIZADA")
                    if count:
                        self.set_status(f"Listo · {count} figuras", "success")
                        self.open_button.configure(state="normal")
                        self.tabs.select(self.viewer_tab)
                    else:
                        self.set_status("Finalizó sin figuras", "error")
                        self.tabs.select(self.log_tab)
                else:
                    self.log_meta.set("CORRIDA CON ERROR")
                    self.set_status(f"Finalizó con error ({code})", "error")
                    self.tabs.select(self.log_tab)
                finished = True
        if finished:
            self.progress.stop()
            self.progress.pack_forget()
            self.process = None
            self.run_button.configure(state="normal")
        else:
            self.root.after(100, self.poll_events)

    def open_results(self) -> None:
        if self.results_dir and self.results_dir.is_dir():
            os.startfile(self.results_dir)  # Windows desktop launcher.

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Ejecución en curso", "¿Cerrar y detener la ejecución actual?"):
                return
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TpiLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
