import unittest
import os, sys
from pathlib import Path

import torch
import numpy as np

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from utils import get_cells_with_object, tensor2boxlist
from mealtrays_dataset import get_validation_dataset

class TestUtils(unittest.TestCase):
    def __init__(self, TestUtils) -> None:
        super().__init__(TestUtils)
        self.size = 448
        self.S = 7
        self.B = 2
        self.C = 8
        self.channel_img = 3
        self.BATCH_SIZE = 32
        self.validation_loader = get_validation_dataset(self.BATCH_SIZE, isAugment=False)
        _, self.target = next(iter(self.validation_loader))

    def test_cellswithobject(self):  
        N, cells_i, cells_j = get_cells_with_object(self.target)
        target = self.target[N, cells_i, cells_j]
        idx1 = np.random.randint(0, len(target))
        idx2 = np.random.randint(0, len(target[idx1][:5]))

        self.assertIs(type(N), torch.Tensor)
        self.assertIs(type(cells_i), torch.Tensor)
        self.assertIs(type(cells_j), torch.Tensor)
        self.assertEqual(len(target.size()), 2)
        self.assertIsNot(target[idx1][idx2],torch.tensor([0]))
        self.assertEqual(target[idx1].size(), torch.Size([5+self.C]))
        
    def test_tensor2boxlist(self):
        #TODO
        pass

if __name__ == "__main__":
    unittest.main()
        
