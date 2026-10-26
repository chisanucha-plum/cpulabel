"""Regression tests for helmet reclassification logic."""

import unittest

from pipeline.helmet import apply_helmet
from schemas import Detection


def det(class_id, cx, cy, bw=0.2, bh=0.2, confidence=0.9):
    return Detection(class_id=class_id, box=(cx, cy, bw, bh), confidence=confidence)


HELMET, HEAD = 0, 1          # config class ids for raw DINO queries
OUT_HELMET, OUT_NONE = 0, 2  # output ids


class TestApplyHelmet(unittest.TestCase):
    def test_head_with_overlapping_helmet_relabeled(self):
        dets = [
            det(HEAD, 0.5, 0.5, confidence=0.8),
            det(HELMET, 0.5, 0.5, confidence=0.6),
        ]
        out = apply_helmet(dets, HELMET, HEAD, OUT_HELMET, OUT_NONE)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].class_id, OUT_HELMET)
        self.assertAlmostEqual(out[0].confidence, (0.8 + 0.6) / 2)

    def test_head_without_helmet_labeled_no_helmet(self):
        dets = [det(HEAD, 0.5, 0.5, confidence=0.7)]
        out = apply_helmet(dets, HELMET, HEAD, OUT_HELMET, OUT_NONE)
        self.assertEqual(out[0].class_id, OUT_NONE)
        self.assertAlmostEqual(out[0].confidence, 0.7)

    def test_unmatched_helmet_dropped(self):
        dets = [
            det(HEAD, 0.5, 0.5),
            det(HELMET, 0.05, 0.05),  # far away, below IoU threshold
        ]
        out = apply_helmet(dets, HELMET, HEAD, OUT_HELMET, OUT_NONE)
        self.assertEqual([d.class_id for d in out], [OUT_NONE])

    def test_no_heads_returns_input_unchanged(self):
        dets = [det(HELMET, 0.5, 0.5)]
        out = apply_helmet(dets, HELMET, HEAD, OUT_HELMET, OUT_NONE)
        self.assertEqual(out, dets)

    def test_best_matching_helmet_wins(self):
        dets = [
            det(HEAD, 0.5, 0.5),
            det(HELMET, 0.52, 0.5, confidence=0.3),   # strong overlap
            det(HELMET, 0.90, 0.9, confidence=0.99),  # negligible overlap
        ]
        out = apply_helmet(dets, HELMET, HEAD, OUT_HELMET, OUT_NONE, iou_threshold=0.1)
        self.assertEqual(out[0].class_id, OUT_HELMET)
        self.assertAlmostEqual(out[0].confidence, (0.9 + 0.3) / 2)


if __name__ == "__main__":
    unittest.main()
