import unittest
import os, sys
from pathlib import Path

import torch

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from loss import YoloLoss

class TestLoss(unittest.TestCase):
    def __init__(self, TestLoss) -> None:
        super().__init__(TestLoss)
        self.size = 448
        self.S = 7
        self.B = 2
        self.C = 8
        self.channel_img = 3
        self.BATCH_SIZE = 32

    def test_loss(self):      
        box_pred = torch.rand(self.BATCH_SIZE, self.S, self.S, self.B*(4+1) + self.C)
        box_true = torch.rand(self.BATCH_SIZE, self.S, self.S, 5 + self.C)

        criterion = YoloLoss(lambd_coord=5, lambd_noobj=0.5, S=self.S, device=torch.device('cpu'))
        losses, loss = criterion(box_pred, box_true)

        self.assertIs(type(losses), dict)
        self.assertEqual(len(losses), 5)
        self.assertIs(type(loss), torch.Tensor)
        for value in losses.values():
            self.assertIs(type(value), torch.Tensor)



if __name__ == "__main__":
    unittest.main()
        
