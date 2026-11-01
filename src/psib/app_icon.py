"""Pure path discovery and defensive loading for the optional desktop icon set."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Any


PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
WINDOW_SIZES = (16, 32, 48, 64)
_warned = False
LOGGER = logging.getLogger(__name__)


def icon_paths(root_dir: str | Path) -> dict[str, Any]:
    """Return the expected master, generated icon, ICO and sidebar-logo paths."""
    root = Path(root_dir)
    icon_dir = root / "assets" / "icon"
    return {
        "master": icon_dir / "psib_icon_master.png",
        "pngs": {size: icon_dir / f"psib_icon_{size}.png" for size in PNG_SIZES},
        "ico": icon_dir / "psib.ico",
        "logo": root / "assets" / "logo.png",
    }


def warn_icon_once(message: str = "Application icon could not be loaded; continuing without a custom icon.",
                   logger: logging.Logger | None = None,
                   warning_callback: Callable[[str], None] | None = None) -> None:
    """Log a single icon warning per process, even if multiple widgets retry."""
    global _warned
    if _warned:
        return
    _warned = True
    if warning_callback is not None:
        warning_callback(message)
    else:
        (logger or LOGGER).warning(message)


def load_icon_images(
    root_dir: str | Path,
    photo_image_factory: Callable[[Path], Any],
    *,
    logger: logging.Logger | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Load Tk-compatible images; return ``None`` on any missing or invalid asset.

    The factory is injected so path and failure behavior can be tested without a
    display. All generated files are checked before loading any of them.
    """
    paths = icon_paths(root_dir)
    expected = [paths["master"], *paths["pngs"].values(), paths["ico"], paths["logo"]]
    missing = [path for path in expected if not path.is_file()]
    if missing:
        warn_icon_once(
            "Application icon assets are incomplete; continuing without a custom icon. "
            "Place psib_icon_master.png in assets/icon/ and run scripts/build_icons.py.",
            logger, warning_callback,
        )
        return None
    try:
        photos = {size: photo_image_factory(paths["pngs"][size]) for size in (*WINDOW_SIZES, 24)}
        logo = photo_image_factory(paths["logo"])
    except Exception as exc:  # Tk image decoding can raise TclError as well as OSError.
        warn_icon_once(f"Application icon assets could not be loaded ({exc}); continuing without a custom icon.",
                       logger, warning_callback)
        return None
    return {"window": tuple(photos[size] for size in WINDOW_SIZES),
            "rail": photos[24], "logo": logo, "ico": paths["ico"]}
