"""Icon asset build and fallback tests that do not need Tk or a display."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from psib.app_icon import PNG_SIZES, icon_paths, load_icon_images
from scripts.build_icons import ICO_SIZES, PNG_SIZES as BUILD_PNG_SIZES, build_icons


def make_test_master(path: Path) -> None:
    """Create an isolated fixture; no fixture art is written into project assets."""
    image = Image.new("RGBA", (1024, 1024), (18, 28, 46, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((190, 230, 820, 830), fill=(100, 160, 120, 255))
    draw.ellipse((560, 430, 710, 580), fill=(230, 70, 70, 255))
    image.save(path, format="PNG", optimize=False, compress_level=9)


class AppIconTests(unittest.TestCase):
    def test_icon_paths(self):
        root = Path("/fake/project")
        paths = icon_paths(root)
        self.assertEqual(paths["master"], root / "assets" / "icon" / "psib_icon_master.png")
        self.assertEqual(paths["pngs"], {size: root / "assets" / "icon" / f"psib_icon_{size}.png"
                                         for size in PNG_SIZES})
        self.assertEqual(paths["ico"], root / "assets" / "icon" / "psib.ico")
        self.assertEqual(paths["logo"], root / "assets" / "logo.png")

    def test_missing_icon_assets_return_none_and_warn_without_display(self):
        with tempfile.TemporaryDirectory() as directory:
            warnings = []
            result = load_icon_images(directory, lambda path: self.fail("loader should check paths first"),
                                      warning_callback=warnings.append)
            self.assertIsNone(result)
            self.assertEqual(len(warnings), 1)
            self.assertIn("psib_icon_master.png", warnings[0])

    def test_missing_master_does_not_create_derived_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                build_icons(root)
            self.assertFalse((root / "assets" / "icon").exists())
            self.assertFalse((root / "assets" / "logo.png").exists())

    def test_build_outputs_all_sizes_and_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            master = root / "assets" / "icon" / "psib_icon_master.png"
            master.parent.mkdir(parents=True)
            make_test_master(master)
            outputs = build_icons(root)
            self.assertEqual(len(outputs), len(BUILD_PNG_SIZES) + 2)
            expected_pngs = [root / "assets" / "icon" / f"psib_icon_{size}.png"
                             for size in BUILD_PNG_SIZES]
            for size, path in zip(BUILD_PNG_SIZES, expected_pngs):
                with Image.open(path) as image:
                    self.assertEqual(image.size, (size, size))
            with Image.open(root / "assets" / "icon" / "psib.ico") as image:
                self.assertEqual(image.ico.sizes(), {(size, size) for size in ICO_SIZES})
            with Image.open(root / "assets" / "logo.png") as image:
                self.assertEqual(image.size, (56, 56))
            first_bytes = {path: path.read_bytes() for path in outputs}
            second_outputs = build_icons(root)
            self.assertEqual(first_bytes, {path: path.read_bytes() for path in second_outputs})

    def test_build_recognizes_supplied_legacy_master_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_master = root / "assets" / "icon" / "liver_tumor_app_icon.png"
            legacy_master.parent.mkdir(parents=True)
            make_test_master(legacy_master)
            outputs = build_icons(root)
            canonical_master = root / "assets" / "icon" / "psib_icon_master.png"
            self.assertIn(canonical_master, outputs)
            self.assertEqual(canonical_master.read_bytes(), legacy_master.read_bytes())
            self.assertTrue((root / "assets" / "icon" / "psib_icon_16.png").is_file())


if __name__ == "__main__":
    unittest.main()
