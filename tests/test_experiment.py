import unittest

import numpy as np
import torch

from count_apples import AppleCenters, center_targets, count_color, count_peaks, focal_loss, has_overlap, summarize
from prepare import official_test
from prepare_row_holdout import assign_rows


class ExperimentTest(unittest.TestCase):
    def test_center_collision_and_peak_plateau(self):
        target, collisions = center_targets([[0.5, 0.5], [0.501, 0.501], [0.8, 0.7]])
        self.assertEqual(collisions, 1)
        self.assertEqual(np.count_nonzero(target == 1), 2)
        heat = np.array([[0.8, 0.8, 0], [0, 0, 0], [0, 0, 0.9]], dtype=np.float32)
        self.assertEqual(count_peaks(heat, 0.5), 2)
        self.assertEqual(count_peaks(np.zeros((3, 3), dtype=np.float32), 0.5), 0)

    def test_color_rejects_small_fragments_without_uint_underflow(self):
        image = np.full((5, 5, 3), [10, 90, 90], dtype=np.uint8)
        image[:2, :2] = [90, 15, 10]
        image[4, 4] = [90, 15, 10]
        self.assertEqual(count_color(image, 25, 4), 1)

    def test_qualifier_uses_box_iou_and_handles_empty_cohort(self):
        self.assertTrue(has_overlap([[0, 0, 10, 10], [5, 1, 10, 10]]))
        self.assertFalse(has_overlap([[0, 0, 10, 10], [9, 9, 10, 10]]))
        self.assertFalse(has_overlap([]))
        self.assertEqual(summarize([]), {"images": 0, "apples": 0})

    def test_official_filename_field_and_zero_apple_images(self):
        info = {"width": 720, "height": 1280}
        coco = {
            "categories": [{"id": 0, "name": "apple"}],
            "images": [{**info, "id": 0, "filename": "dataset1_front_1.png"},
                       {**info, "id": 1, "filename": "dataset2_back_2.png"}],
            "annotations": [{"category_id": 0, "image_id": 0, "iscrowd": 0, "bbox": [1, 1, 2, 2]}],
        }
        selected = official_test(coco, 12)
        self.assertEqual([image["group"] for image in selected], ["dataset1_front", "dataset2_back"])
        self.assertEqual([len(image["boxes"]) for image in selected], [1, 0])

    def test_row_holdout_never_shares_an_orchard_row_between_splits(self):
        groups = ("dataset1_back", "dataset1_front", "dataset2_back", "dataset2_front",
                  "dataset3_back", "dataset3_front", "dataset4_front")
        images = [{"id": j * 24 + i, "filename": f"{group}_{i}.png", "width": 720, "height": 1280}
                  for j, group in enumerate(groups) for i in range(24)]
        picked = assign_rows({"categories": [{"id": 0, "name": "apple"}], "images": images, "annotations": []})
        self.assertEqual([len(value) for value in picked.values()], [30, 12, 24])
        self.assertEqual([{item["group"].split("_")[0] for item in picked[split]} for split in picked],
                         [{"dataset1"}, {"dataset2"}, {"dataset3", "dataset4"}])

    def test_actual_model_forward_and_backward(self):
        torch.set_num_threads(1)
        model = AppleCenters(pretrained=False)
        logits = model(torch.zeros((2, 3, 32, 32)))
        self.assertEqual(tuple(logits.shape), (2, 8, 8))
        labels = torch.zeros_like(logits)
        labels[0, 2, 3] = 1
        loss = focal_loss(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.head[-1].bias.grad)
