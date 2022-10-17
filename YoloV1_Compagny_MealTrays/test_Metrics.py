import unittest
import torch

from metrics import class_acc, MSE, MSE_confidenceScore

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
        
        self.prediction = torch.rand(BATCH_SIZE, S, S, 5*B+C)

    def test_class_acc(self):
        ### TODO : assertEqual()
        acc = class_acc(self.target, self.prediction)
        self.assertIs(type(acc), float)
        self.assertGreaterEqual(acc, 0.)
        self.assertLessEqual(acc,1.)

    def test_MSE(self):
        ### TODO : assertEqual()
        mse = MSE(self.target, self.prediction)
        self.assertIs(type(mse), float)
        self.assertGreaterEqual(mse, 0.)

    def test_MSE_confidenceScore(self):
        ### TODO : assertEqual()
        mse = MSE(self.target, self.prediction)
        self.assertIs(type(mse), float)
        self.assertGreaterEqual(mse, 0.)


if __name__ == "__main__":
    unittest.main()
        
