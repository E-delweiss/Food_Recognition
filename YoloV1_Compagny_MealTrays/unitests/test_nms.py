import unittest
import os, sys
from pathlib import Path

import torch
import numpy as np

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from NMS import non_max_suppression

class TestNMS(unittest.TestCase):
    def __init__(self, TestNMS) -> None:
        super().__init__(TestNMS)
        self.size = 448
        self.S = 7
        self.B = 2
        self.C = 8
        self.channel_img = 3
        self.BATCH_SIZE = 32
        self.prob_threshold = 0.5
        self.iou_threshold = 0.5

    def test_NMS(self):      
        prediction = torch.rand(1, self.S, self.S, self.B*(4+1) + self.C)
        box_nms = non_max_suppression(prediction, self.prob_threshold, self.iou_threshold)
        
        idx = np.random.randint(0, len(box_nms))
        self.assertIs(type(box_nms), list)
        self.assertEqual(len(box_nms[idx]), 6)


if __name__ == "__main__":
    unittest.main()
        
