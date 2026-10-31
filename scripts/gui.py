"""Small Windows-friendly graphical launcher for the 2D TPI workflow."""
from __future__ import annotations

from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
RESULT_PREFIX = "TPI run completed. Results are in "


class TpiLauncher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.results_dir: Path | None = None

        root.title("TPI | Segmentación hepática")
        root.geometry("1000x640")
        root.minsize(800, 540)
        root.configure(background="#F3F6FB")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.configure_styles()

        frame = ttk.Frame(root, style="App.TFrame", padding=(28, 24))
        frame.pack(fill="both", expand=True)

        hero = ttk.Frame(frame, style="Hero.TFrame", padding=(24, 21))
        hero.pack(fill="x")
        hero_top = ttk.Frame(hero, style="Hero.TFrame")
        hero_top.pack(fill="x")
        ttk.Label(hero_top, text="PSIB  /  PROYECTO INTEGRADOR", style="Eyebrow.Hero.TLabel").pack(side="left")
        ttk.Label(hero_top, text="FLUJO 2D", style="Badge.Hero.TLabel", padding=(11, 5)).pack(side="right")
        ttk.Label(hero, text="Segmentación hepática", style="Title.Hero.TLabel").pack(anchor="w", pady=(13, 3))
        ttk.Label(
            hero,
            text="Selección reproducible de cortes · segmentación base · resultados listos para revisar",
            style="Subtitle.Hero.TLabel",
        ).pack(anchor="w")

        steps = ttk.Frame(frame, style="App.TFrame")
        steps.pack(fill="x", pady=(18, 18))
        for index, (title, caption) in enumerate((
            ("01  Selección", "Cortes con lesiones"),
            ("02  Segmentación", "Método base 2D"),
            ("03  Resultados", "Métricas y máscaras"),
        )):
            card = ttk.Frame(steps, style="Step.TFrame", padding=(15, 12))
            card.pack(side="left", fill="x", expand=True, padx=(0 if index == 0 else 8, 0))
            ttk.Label(card, text=title, style="StepTitle.TLabel").pack(anchor="w")
            ttk.Label(card, text=caption, style="StepCaption.TLabel").pack(anchor="w", pady=(3, 0))

        config_card = ttk.Frame(frame, style="Card.TFrame", padding=(19, 16))
        config_card.pack(fill="x")
        ttk.Label(config_card, text="Configuración", style="SectionTitle.TLabel").pack(anchor="w")
        ttk.Label(
            config_card,
            text="Elegí el archivo YAML que define la ubicación de los datos y los parámetros del flujo.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(4, 11))
        entry_row = ttk.Frame(config_card, style="Card.TFrame")
        entry_row.pack(fill="x")
        self.config_path = tk.StringVar(value=str(DEFAULT_CONFIG))
        ttk.Entry(entry_row, textvariable=self.config_path, style="Config.TEntry").pack(side="left", fill="x", expand=True)
        ttk.Button(entry_row, text="Examinar…", style="Secondary.TButton", command=self.browse_config).pack(side="left", padx=(9, 0))

        controls = ttk.Frame(frame, style="App.TFrame")
        controls.pack(fill="x", pady=(15, 15))
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

        log_card = ttk.Frame(frame, style="Card.TFrame", padding=(16, 13))
        self.log_card = log_card
        log_card.pack(fill="both", expand=True)
        log_header = ttk.Frame(log_card, style="Card.TFrame")
        log_header.pack(fill="x", pady=(0, 10))
        ttk.Label(log_header, text="Actividad", style="SectionTitle.TLabel").pack(side="left")
        ttk.Label(log_header, text="SALIDA EN VIVO", style="LogMeta.TLabel").pack(side="right")
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

    def configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#F3F6FB")
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

    def set_status(self, text: str, state: str) -> None:
        self.status.set(text)
        style = f"Status{state.title()}.TFrame"
        label_style = f"Status{state.title()}.TLabel"
        self.status_pill.configure(style=style)
        self.status_dot.configure(style=label_style)
        self.status_label.configure(style=label_style)

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
        self.set_status("Ejecutando…", "running")
        self.progress.pack(fill="x", pady=(0, 12), before=self.log_card)
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
                self.set_status("Error al iniciar", "error")
                finished = True
            elif kind == "done":
                code = int(value)
                if code == 0:
                    self.set_status("Finalizado correctamente", "success")
                    if self.results_dir and self.results_dir.is_dir():
                        self.open_button.configure(state="normal")
                else:
                    self.set_status(f"Finalizó con error ({code})", "error")
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
