import unittest
import os, sys
from pathlib import Path

import torch
import numpy as np

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from utils import get_cells_with_object, tensor2boxlist

class TestUtils(unittest.TestCase):
    def __init__(self, TestUtils) -> None:
        super().__init__(TestUtils)
        self.size = 448
        self.S = 7
        self.B = 2
        self.C = 8
        self.channel_img = 3
        self.BATCH_SIZE = 32

    def test_cellswithobject(self):      
        prediction = torch.rand(1, self.S, self.S, self.B*(4+1) + self.C)
        
        
    def test_tensor2boxlist(self):
        pass

if __name__ == "__main__":
    unittest.main()
        
