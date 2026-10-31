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

        root.title("TPI — Segmentación hepática")
        root.geometry("780x560")
        root.minsize(620, 420)
        root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Procesamiento de imágenes médicas", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Ejecuta selección 2D de cortes y segmentación base. Los resultados se guardan según la configuración.",
            wraplength=720,
        ).pack(anchor="w", pady=(5, 16))

        config_row = ttk.Frame(frame)
        config_row.pack(fill="x", pady=(0, 12))
        ttk.Label(config_row, text="Configuración YAML:").pack(anchor="w")
        entry_row = ttk.Frame(config_row)
        entry_row.pack(fill="x", pady=(4, 0))
        self.config_path = tk.StringVar(value=str(DEFAULT_CONFIG))
        ttk.Entry(entry_row, textvariable=self.config_path).pack(side="left", fill="x", expand=True)
        ttk.Button(entry_row, text="Buscar…", command=self.browse_config).pack(side="left", padx=(8, 0))

        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(0, 10))
        self.run_button = ttk.Button(actions, text="Ejecutar flujo completo", command=self.run)
        self.run_button.pack(side="left")
        self.open_button = ttk.Button(actions, text="Abrir resultados", command=self.open_results, state="disabled")
        self.open_button.pack(side="left", padx=(8, 0))
        self.status = tk.StringVar(value="Listo")
        ttk.Label(actions, textvariable=self.status).pack(side="right")

        log_frame = ttk.LabelFrame(frame, text="Registro de ejecución", padding=6)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, wrap="word", state="disabled", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def browse_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Elegir configuración del TPI",
            initialdir=str(Path(self.config_path.get()).parent),
            filetypes=(("Archivos YAML", "*.yaml *.yml"), ("Todos los archivos", "*.*")),
        )
        if selected:
            self.config_path.set(selected)

    def append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
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
        self.status.set("Ejecutando…")
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
                self.status.set("Error")
                finished = True
            elif kind == "done":
                code = int(value)
                if code == 0:
                    self.status.set("Finalizado correctamente")
                    if self.results_dir and self.results_dir.is_dir():
                        self.open_button.configure(state="normal")
                else:
                    self.status.set(f"Finalizó con error ({code})")
                finished = True
        if finished:
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
