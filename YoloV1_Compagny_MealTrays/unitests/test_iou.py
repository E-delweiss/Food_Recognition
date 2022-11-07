import unittest
import os, sys
from pathlib import Path

import torch
import numpy as np

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from IoU import intersection_over_union

class TestIoU(unittest.TestCase):
    def __init__(self, TestIoU) -> None:
        super().__init__(TestIoU)
        self.size = 448
        self.S = 7
        self.B = 2
        self.C = 8
        self.channel_img = 3
        self.BATCH_SIZE = 32
        self.prob_threshold = 0.5
        self.iou_threshold = 0.5

    def test_iou(self):
        box1 = torch.rand(self.BATCH_SIZE, 4)
        box2 = box1.clone() * 0.9

        iou = intersection_over_union(box1, box2)
        
        idx = np.random.randint(0,self.BATCH_SIZE)
        self.assertEqual(iou.size(), torch.Size([self.BATCH_SIZE]))
        self.assertAlmostEqual(abs(iou[idx].item()), 0)

if __name__ == "__main__":
    unittest.main()
        
