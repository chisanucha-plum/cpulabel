"""Regression tests for configuration parsing."""

import unittest

from schemas import ConfigLoader


FULL = {
    "models": {"dino_config": "a.py", "dino_weights": "b.pth", "sam_weights": "c.pt"},
    "paths": {"input_folder": "in", "output_folder": "out"},
    "detection": {"box_threshold": 0.45, "text_threshold": 0.30},
    "helmet": {"iou_threshold": 0.25},
    "classes": {
        "0": {"name": "helmet", "box_threshold": 0.40, "text_threshold": 0.20},
        "1": {"name": "head"},
        "2": {"name": "no_helmet", "detect": False},
    },
    "colors": {"0": [0, 255, 0], "2": [0, 0, 255]},
}

MINIMAL = {}


class TestParseConfig(unittest.TestCase):
    def test_full_document(self):
        cfg = ConfigLoader._parse_config(FULL)
        self.assertEqual(cfg.input_folder, "in")
        self.assertEqual(cfg.output_folder, "out")
        self.assertAlmostEqual(cfg.box_threshold, 0.45)
        self.assertAlmostEqual(cfg.text_threshold, 0.30)
        self.assertAlmostEqual(cfg.helmet_iou_threshold, 0.25)
        self.assertEqual(cfg.classes, {0: "helmet", 1: "head", 2: "no_helmet"})
        self.assertEqual(cfg.class_colors, {0: [0, 255, 0], 2: [0, 0, 255]})
        # Raw per-class settings must pass through for per-class thresholds
        # and "detect": false flags used by the DINO model
        self.assertEqual(cfg.class_settings, FULL["classes"])

    def test_minimal_document_uses_defaults(self):
        cfg = ConfigLoader._parse_config(MINIMAL)
        self.assertEqual(cfg.input_folder, "images")
        self.assertEqual(cfg.output_folder, "dataset")
        self.assertEqual(cfg.classes, {})
        self.assertEqual(cfg.class_colors, {})
        self.assertEqual(cfg.class_settings, {})

    def test_old_string_class_format(self):
        cfg = ConfigLoader._parse_config({"classes": {"1": "person"}})
        self.assertEqual(cfg.classes, {1: "person"})


if __name__ == "__main__":
    unittest.main()
