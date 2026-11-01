"""Build deterministic PNG, ICO and sidebar-logo assets from the supplied master."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from PIL import Image


PNG_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
LOGO_SIZE = 56  # 2x display size for the 28 px sidebar-header mark.
RESAMPLE = Image.Resampling.LANCZOS


def build_icons(project_root: str | Path | None = None,
                master_path: str | Path | None = None) -> list[Path]:
    """Generate the icon set, overwriting only expected outputs and preserving input.

    ``project_root`` and ``master_path`` are injectable for isolated tests. The
    user's master design is never modified.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    icon_dir = root / "assets" / "icon"
    canonical_master = icon_dir / "psib_icon_master.png"
    if master_path is not None:
        master = Path(master_path)
    elif canonical_master.is_file():
        master = canonical_master
    else:
        # Compatibility with the already supplied, same-design 1024px artwork.
        master = icon_dir / "liver_tumor_app_icon.png"
    if not master.is_file():
        raise FileNotFoundError(
            f"Master icon not found: {canonical_master}. Place psib_icon_master.png in assets/icon/ and rerun."
        )
    icon_dir.mkdir(parents=True, exist_ok=True)
    logo_path = root / "assets" / "logo.png"
    logo_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(master) as source:
        if source.size != (1024, 1024):
            raise ValueError(f"Master icon must be 1024x1024 pixels; got {source.size} at {master}")
        master_image = source.convert("RGBA")
    outputs: list[Path] = []
    if master.resolve() != canonical_master.resolve():
        shutil.copyfile(master, canonical_master)
        outputs.append(canonical_master)
    for size in PNG_SIZES:
        image = master_image.resize((size, size), RESAMPLE)
        path = icon_dir / f"psib_icon_{size}.png"
        image.save(path, format="PNG", optimize=False, compress_level=9)
        outputs.append(path)

    ico_path = icon_dir / "psib.ico"
    master_image.save(ico_path, format="ICO", sizes=list((size, size) for size in ICO_SIZES))
    outputs.append(ico_path)

    logo = master_image.resize((LOGO_SIZE, LOGO_SIZE), RESAMPLE)
    logo.save(logo_path, format="PNG", optimize=False, compress_level=9)
    outputs.append(logo_path)

    print("Generated icon files:")
    for path in outputs:
        print(f"  {path.relative_to(root)} — {path.stat().st_size:,} bytes")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--master", type=Path, help="Optional path to the 1024x1024 master PNG")
    args = parser.parse_args()
    build_icons(args.project_root, args.master)


if __name__ == "__main__":
    main()
