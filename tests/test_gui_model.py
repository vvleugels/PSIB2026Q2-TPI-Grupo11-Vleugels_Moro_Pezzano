"""No-display tests for persistent GUI sidebar state."""
import tempfile
import unittest
from pathlib import Path

from psib.gui_model import ProfileStore, SidebarState


class SidebarStateTests(unittest.TestCase):
    def test_toggle(self):
        state = SidebarState()
        self.assertTrue(state.toggle())
        self.assertFalse(state.toggle())

    def test_select_section_while_collapsed_expands_sidebar(self):
        state = SidebarState(collapsed=True)
        state.select_section("Segmentación")
        self.assertEqual(state.active_section, "Segmentación")
        self.assertFalse(state.collapsed)

    def test_profile_store_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / ".psib_gui")
            state = SidebarState(collapsed=True, active_section="Resultados", width=420)
            store.save_sidebar(state)
            self.assertEqual(store.load_sidebar(), state)

    def test_width_clamping(self):
        state = SidebarState()
        self.assertEqual(state.set_width(180, 1000), 300)
        self.assertEqual(state.set_width(800, 1000), 450)
        self.assertEqual(state.set_width(800, 500), 300)


if __name__ == "__main__":
    unittest.main()
