"""Regression tests for GroundingDINO per-class threshold resolution."""

import unittest

from schemas import Config
from models.dino import GroundingDINOModel


def make_model(class_settings):
    """Build a GroundingDINOModel without loading weights (skips __init__)."""
    model = GroundingDINOModel.__new__(GroundingDINOModel)
    model.config = Config(box_threshold=0.35, text_threshold=0.25,
                          class_settings=class_settings)
    return model


class TestGetThreshold(unittest.TestCase):
    def test_per_class_overrides(self):
        model = make_model({"0": {"box_threshold": 0.40, "text_threshold": 0.20}})
        self.assertAlmostEqual(model._get_threshold(0, "box"), 0.40)
        self.assertAlmostEqual(model._get_threshold(0, "text"), 0.20)

    def test_fallback_to_global_when_class_missing(self):
        model = make_model({})
        self.assertAlmostEqual(model._get_threshold(7, "box"), 0.35)
        self.assertAlmostEqual(model._get_threshold(7, "text"), 0.25)

    def test_partial_settings_fall_back_per_key(self):
        model = make_model({"3": {"name": "person"}})
        self.assertAlmostEqual(model._get_threshold(3, "box"), 0.35)
        self.assertAlmostEqual(model._get_threshold(3, "text"), 0.25)

    def test_non_dict_class_value_uses_global(self):
        model = make_model({"2": "safety vest"})
        self.assertAlmostEqual(model._get_threshold(2, "box"), 0.35)


if __name__ == "__main__":
    unittest.main()
