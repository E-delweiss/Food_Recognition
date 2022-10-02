import unittest
import torch
import numpy as np

from Metrics import class_acc, MSE, MSE_confidenceScore

class TestYololoss(unittest.TestCase):
    """
    TODO
    """
    def __init__(self, TestYololoss) -> None:
        super().__init__(TestYololoss)
        size = 75
        S = 6
        B = 1
        C = 10
        channel_img = 1
        BATCH_SIZE = 32

        self.box_true = torch.zeros(BATCH_SIZE, S, S, B+4)
        N = range(BATCH_SIZE)
        i = torch.randint(S, (1,BATCH_SIZE))
        j = torch.randint(S, (1,BATCH_SIZE))
        box_value = torch.rand((BATCH_SIZE, B+4))
        self.box_true[N,i,j] = box_value

        self.label_true = torch.zeros(BATCH_SIZE, C)
        idx = torch.randint(C, (1, BATCH_SIZE))
        self.label_true[N, idx] = 1
        
        self.box_pred = torch.rand(BATCH_SIZE, S, S, B+4)
        self.label_pred = torch.rand(BATCH_SIZE, S, S, C)

    def test_class_acc(self):
        acc = class_acc(self.box_true, self.label_true, self.label_pred)
        self.assertIs(type(acc), float)
        self.assertGreaterEqual(acc, 0.)
        self.assertLessEqual(acc,1.)

    def test_MSE(self):
        mse = MSE(self.box_true, self.box_pred)
        self.assertIs(type(mse), float)
        self.assertGreaterEqual(mse, 0.)
        self.assertLessEqual(mse,1.)

    def test_MSE_confidenceScore(self):
        ### TODO
        pass


if __name__ == "__main__":
    unittest.main()
        