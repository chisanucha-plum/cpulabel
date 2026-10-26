"""Regression tests for COCO annotation creation."""

import unittest

import numpy as np

from schemas import Detection
from utils.annotations import create_coco_annotation


def square_mask():
    m = np.zeros((100, 200), dtype=bool)  # h=100, w=200
    m[10:30, 40:80] = True
    return m


class TestCreateCocoAnnotation(unittest.TestCase):
    def test_bbox_math(self):
        # det cx=0.5, cy=0.5, bw=0.25, bh=0.1 on 200x100
        d = Detection(class_id=0, box=(0.5, 0.5, 0.25, 0.1), confidence=0.9)
        ann = create_coco_annotation(
            image_id=3,
            annotation_id=7,
            detection=d,
            mask=square_mask(),
            width=200,
            height=100,
        )
        self.assertEqual(ann["bbox"], [75, 45, 50, 10])
        self.assertEqual(ann["area"], 500)
        self.assertEqual(ann["image_id"], 3)
        self.assertEqual(ann["id"], 7)
        self.assertEqual(ann["category_id"], 1)  # class_id + 1
        self.assertEqual(ann["iscrowd"], 0)

    def test_segmentation_from_mask_contours(self):
        d = Detection(class_id=2, box=(0.3, 0.2, 0.2, 0.2), confidence=0.5)
        ann = create_coco_annotation(1, 1, d, square_mask(), width=200, height=100)
        seg = ann["segmentation"]
        self.assertEqual(len(seg), 1)  # one rectangle contour
        self.assertGreaterEqual(len(seg[0]), 6)


if __name__ == "__main__":
    unittest.main()
