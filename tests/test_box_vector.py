"""Regression tests for BoundingBoxVector.iou (used by helmet logic)."""

import unittest

from schemas import Detection
from utils.boxes import BoundingBoxVector


def det(cx, cy, bw, bh):
    return Detection(class_id=0, box=(cx, cy, bw, bh), confidence=1.0)


class TestIoU(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(BoundingBoxVector.iou(det(0.5, 0.5, 0.2, 0.2),
                                                     det(0.5, 0.5, 0.2, 0.2)), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(BoundingBoxVector.iou(det(0.1, 0.1, 0.1, 0.1),
                                               det(0.9, 0.9, 0.1, 0.1)), 0.0)

    def test_partial_overlap(self):
        # [0,0.2]x[0,0.2] vs [0.1,0.3]x[0.1,0.3]: inter 0.01, union 0.07
        iou = BoundingBoxVector.iou(det(0.1, 0.1, 0.2, 0.2), det(0.2, 0.2, 0.2, 0.2))
        self.assertAlmostEqual(iou, 0.01 / 0.07)

    def test_touching_edges_only(self):
        self.assertEqual(BoundingBoxVector.iou(det(0.0, 0.5, 0.2, 0.2),
                                               det(0.2, 0.5, 0.2, 0.2)), 0.0)


if __name__ == "__main__":
    unittest.main()
