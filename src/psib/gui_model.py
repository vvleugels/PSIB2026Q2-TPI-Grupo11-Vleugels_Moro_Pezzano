"""Display-independent state and persistence for the desktop GUI."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


SECTIONS = ("Caso", "Preprocesamiento", "Segmentación", "Cuantificación", "Resultados")
DEFAULT_SIDEBAR_WIDTH = 340
MIN_SIDEBAR_WIDTH = 300


@dataclass
class SidebarState:
    """Sidebar state with width limits tied to the current window geometry."""

    collapsed: bool = False
    active_section: str = "Caso"
    width: int = DEFAULT_SIDEBAR_WIDTH

    def __post_init__(self) -> None:
        if self.active_section not in SECTIONS:
            self.active_section = SECTIONS[0]
        self.width = max(MIN_SIDEBAR_WIDTH, int(self.width))

    def toggle(self) -> bool:
        """Toggle collapsed state and return the new value."""
        self.collapsed = not self.collapsed
        return self.collapsed

    def select_section(self, section: str) -> str:
        """Select a known section; selecting from the rail expands the sidebar."""
        if section not in SECTIONS:
            raise ValueError(f"Unknown sidebar section: {section!r}")
        self.active_section = section
        self.collapsed = False
        return section

    def set_width(self, width: int | float, window_width: int | float) -> int:
        """Clamp width to 300 px .. 45% of the window and save it in state."""
        upper = max(MIN_SIDEBAR_WIDTH, int(float(window_width) * 0.45))
        self.width = max(MIN_SIDEBAR_WIDTH, min(int(round(float(width))), upper))
        return self.width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "SidebarState":
        values = values if isinstance(values, dict) else {}
        return cls(bool(values.get("collapsed", False)),
                   str(values.get("active_section", SECTIONS[0])),
                   int(values.get("width", DEFAULT_SIDEBAR_WIDTH)))


class ProfileStore:
    """JSON profile persisted under the ignored ``.psib_gui/`` directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "profile.json"

    def load_sidebar(self) -> SidebarState:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return SidebarState()
        return SidebarState.from_dict(data.get("sidebar") if isinstance(data, dict) else None)

    def save_sidebar(self, state: SidebarState) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"sidebar": state.to_dict()}, indent=2), encoding="utf-8")
        temporary.replace(self.path)
