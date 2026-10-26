"""Regression tests for output file writers (path layout + content)."""

import json
import tempfile
import unittest
from pathlib import Path

from schemas import Config
from utils.annotations import save_coco_json, save_yolo_yaml


class TestSaveOutputs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        # Writers expect the dataset structure to already exist
        (self.out / "coco" / "annotations").mkdir(parents=True)
        (self.out / "yolo").mkdir(parents=True)
        self.config = Config(output_folder=str(self.out),
                             classes={0: "helmet", 1: "head"},
                             class_settings={})

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_coco_json_writes_instances(self):
        data = {"images": [], "annotations": [], "categories": []}
        save_coco_json(data, self.config)
        written = json.loads(
            (self.out / "coco" / "annotations" / "instances.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(written, data)

    def test_save_yolo_yaml_lists_classes(self):
        save_yolo_yaml(self.config)
        text = (self.out / "yolo" / "data.yaml").read_text(encoding="utf-8")
        self.assertIn("nc: 2", text)
        self.assertIn("['helmet', 'head']", text)
        self.assertIn("train: images", text)
        self.assertTrue(text.rstrip().endswith("]"))  # names line complete


if __name__ == "__main__":
    unittest.main()
