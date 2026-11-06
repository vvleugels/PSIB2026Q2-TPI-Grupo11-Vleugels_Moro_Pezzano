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
        state.select_section("Ajustar")
        self.assertEqual(state.active_section, "Ajustar")
        self.assertFalse(state.collapsed)

    def test_legacy_section_names_map_to_the_four_steps(self):
        self.assertEqual(SidebarState.from_dict({"active_section": "Segmentación"}).active_section, "Ajustar")
        self.assertEqual(SidebarState(active_section="Caso").active_section, "Casos")
        state = SidebarState()
        state.select_section("Cuantificación")
        self.assertEqual(state.active_section, "Resultados")
        with self.assertRaises(ValueError):
            state.select_section("Inexistente")

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
