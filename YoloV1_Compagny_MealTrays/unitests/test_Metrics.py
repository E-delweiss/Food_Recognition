import os
import unittest
import os, sys
from pathlib import Path

import torch

current_folder = os.path.dirname(locals().get("__file__"))
parent_folder = Path(current_folder).parent
sys.path.append(str(parent_folder))

from metrics import class_acc, hard_class_acc

class TestYololoss(unittest.TestCase):
    def __init__(self, TestYololoss) -> None:
        super().__init__(TestYololoss)
        S = 7
        B = 2
        C = 8
        BATCH_SIZE = 32

        self.target = torch.zeros(BATCH_SIZE, S, S, 5+C)
        N = range(BATCH_SIZE)
        i = torch.randint(S, (1,BATCH_SIZE))
        j = torch.randint(S, (1,BATCH_SIZE))
        box_value = torch.rand((BATCH_SIZE, 5))
        self.target[N,i,j,:5] = box_value

        label_true = torch.zeros(BATCH_SIZE, C)
        idx = torch.randint(C, (1, BATCH_SIZE))
        label_true[N, idx] = 1
        self.target[N,i,j,5:] = label_true
        
        self.prediction = torch.zeros(BATCH_SIZE, S, S, 5*B+C)
        self.prediction[:,:,:,10:] = torch.rand(8)
        self.prediction[N,i,j,10:] = label_true

        self.N, self.i, self.j = N, i, j

    def test_class_acc(self):
        prediction = self.prediction.clone()
        prediction[self.N, self.i, self.j, 10:] *= 0.999
        acc = class_acc(self.target, prediction)
        self.assertIs(type(acc), float)
        self.assertGreaterEqual(acc, 0.)
        self.assertLessEqual(acc,1.)
        self.assertAlmostEqual(acc, 1)

    def test_class_hard_acc(self):
        prediction = self.prediction.clone()
        prediction[self.N, self.i, self.j, 10:] *= 0.999
        acc = hard_class_acc(self.target, prediction)
        self.assertIs(type(acc), float)
        self.assertGreaterEqual(acc, 0.)
        self.assertLessEqual(acc,1.)
        self.assertAlmostEqual(acc, 1)



if __name__ == "__main__":
    unittest.main()
        
