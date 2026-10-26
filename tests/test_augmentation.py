"""Regression tests for ImageAugmenter coordinate transforms.

Box/mask coordinates must follow the same transform that warpAffine applies
to the image content (verified numerically against OpenCV).
"""

import unittest

import numpy as np

from schemas import Detection
from utils.augment import ImageAugmenter


def make_image():
    img = np.zeros((100, 100, 3), np.uint8)
    img[50, 80] = 255  # bright pixel at x=80, y=50
    return img


def square_mask(size=100):
    mask = np.zeros((size, size), dtype=bool)
    mask[40:60, 40:60] = True
    return mask


def det(cx=0.8, cy=0.5, bw=0.1, bh=0.2, class_id=0, confidence=0.9):
    return Detection(class_id=class_id, box=(cx, cy, bw, bh), confidence=confidence)


class TestRotateWithCoords(unittest.TestCase):
    def test_box_follows_image_rotation(self):
        # Content pixel (80,50) lands near (79,42) after rotate(15);
        # the box center at (0.8, 0.5) must follow it within ~2px of the
        # detected pixel (interpolation smears the bright pixel).
        img, dets, _ = ImageAugmenter.rotate_with_coords(make_image(), 15, [det()], [])
        ys, xs = np.nonzero(img[:, :, 0])
        self.assertAlmostEqual(dets[0].box[0], xs[0] / 100, delta=0.02)
        self.assertAlmostEqual(dets[0].box[1], ys[0] / 100, delta=0.02)
        # Analytic position from cv2.getRotationMatrix2D forward map
        self.assertAlmostEqual(dets[0].box[0], 0.790, places=3)
        self.assertAlmostEqual(dets[0].box[1], 0.422, places=3)
        self.assertLess(dets[0].box[1], 0.46)  # rotated up, not mirrored down
        self.assertAlmostEqual(dets[0].box[2], 0.100, places=6)
        self.assertAlmostEqual(dets[0].box[3], 0.200, places=6)

    def test_mask_rotation_matches_content(self):
        # Mask and image content go through the same warpAffine call,
        # so a mask region covering the bright pixel keeps covering it.
        mask = square_mask()
        mask[:] = False
        mask[48:53, 78:83] = True  # small patch around (80,50)
        img, _, masks = ImageAugmenter.rotate_with_coords(make_image(), 15, [], [mask])
        ys, xs = np.nonzero(img[:, :, 0])
        self.assertTrue(masks[0][ys[0], xs[0]])

    def test_preserves_class_and_confidence(self):
        _, dets, _ = ImageAugmenter.rotate_with_coords(
            make_image(), 15, [det(class_id=2, confidence=0.42)], []
        )
        self.assertEqual(dets[0].class_id, 2)
        self.assertAlmostEqual(dets[0].confidence, 0.42)

    def test_masks_rotated_to_bool(self):
        out, _, masks = ImageAugmenter.rotate_with_coords(
            make_image(), 15, [], [square_mask()]
        )
        self.assertEqual(out.shape, (100, 100, 3))
        self.assertEqual(masks[0].dtype, np.bool_)

    def test_zero_angle_is_identity_for_boxes(self):
        _, dets, _ = ImageAugmenter.rotate_with_coords(
            make_image(), 0, [det()], [square_mask()]
        )
        self.assertAlmostEqual(dets[0].box[0], 0.8, places=6)
        self.assertAlmostEqual(dets[0].box[1], 0.5, places=6)


class TestFlipHWithCoords(unittest.TestCase):
    def test_box_x_mirrored(self):
        _, dets, _ = ImageAugmenter.flip_h_with_coords(
            make_image(), [det()], [square_mask()]
        )
        self.assertAlmostEqual(dets[0].box[0], 0.2, places=6)
        self.assertAlmostEqual(dets[0].box[1], 0.5, places=6)

    def test_content_follows_flip(self):
        # pixel at x=80 must land at x=w-1-80=19; box cx 0.8 -> 0.2 agrees
        out, _, _ = ImageAugmenter.flip_h_with_coords(
            make_image(), [det()], [square_mask()]
        )
        ys, xs = np.nonzero(out[:, :, 0])
        self.assertEqual(int(xs[0]), 19)

    def test_masks_flipped_bool(self):
        _, _, masks = ImageAugmenter.flip_h_with_coords(
            make_image(), [], [square_mask()]
        )
        self.assertTrue(masks[0][40:60, 40:60].any())
        self.assertFalse(masks[0][0:10, 90:100].all())
        self.assertEqual(masks[0].dtype, np.bool_)


class TestNoiseAndDarken(unittest.TestCase):
    def test_noise_changes_pixels_keeps_coords_objects(self):
        d, m = det(), square_mask()
        img, dets, masks = ImageAugmenter.add_noise_with_coords(
            make_image(), 0.5, [d], [m]
        )
        self.assertTrue((img != make_image()).any())
        self.assertEqual(dets[0], d)  # same values, untouched
        self.assertIs(masks[0], m)  # pass-through, same object

    def test_darken_scales_pixels(self):
        out, dets, _ = ImageAugmenter.darken_with_coords(make_image(), 0.5, [det()], [])
        self.assertEqual(int(out[50, 80, 0]), 127)  # 255 * 0.5 clipped
        self.assertAlmostEqual(dets[0].box[0], 0.8)


if __name__ == "__main__":
    unittest.main()
